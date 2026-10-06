"""Zeroshot: one model call per line of a sample, predictions and metrics written next to the raw responses."""

import jinja2
import json
import httpx
import hydra
import openai
import os
import subprocess
import time

from concurrent.futures import ThreadPoolExecutor
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm

from bottled.constants import DATASETS_DIR
from bottled.datasets import zeroshot_model_inputs, zeroshot_pred_parsers, load_from_config, scorers
from bottled.openrouter_models import OPENROUTER_ROUTES
from bottled.utils import (
    calculate_usd_cost, 
    calculate_mtok_cost, 
    load_usd_rates, 
    load_mtok_weights, 
    parse_openai_usage
)


USD_COSTS_CSV = Path(__file__).parent / "charts" / "openrouter_prices.csv"
MTOK_WEIGHTS_CSV = Path(__file__).parent / "charts" / "mtok_weights.csv"
PROMPTS_DIR = Path(__file__).parent / "prompts"


def request_body(cfg: DictConfig, route: tuple[str, str], system_prompt: str, user_content: str) -> dict:
    """The JSON body of one call; common to all zeroshot calls.

    `openai` merges all args into the top level of the dict; `provider` and `reasoning` become siblings of `model`.
    """
    slug, provider = route
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "model": slug,
        "max_tokens": cfg.eval.max_tokens,
    }
    if cfg.model.is_local:
        body["chat_template_kwargs"] = OmegaConf.to_container(cfg.model.chat_template_kwargs) # run_local renders the chat template with these
    else:
        body["provider"] = {"only": [provider], "allow_fallbacks": False}
    if cfg.harness_thinking:
        body["reasoning"] = {"effort": cfg.harness_thinking}
    return body


# `openai` create() accepts only these args; the rest of the body goes through extra_body (see `request_body`)
SDK_FIELDS = ("messages", "model", "max_tokens")


def send(client: openai.OpenAI, body: dict):
    named = {k: body[k] for k in SDK_FIELDS}
    return client.chat.completions.create(**named, extra_body={k: v for k, v in body.items() if k not in SDK_FIELDS})


def submit_batch(cfg: DictConfig, route: tuple[str, str], system_prompt: str, dataset, output_dir: Path) -> str:
    """Send every line of the sample to OpenRouter as one batch, then stop; the batch runs server-side.

    custom_id is the line's index in the sample. 
    """
    requests_list = []
    for i in range(len(dataset)):
        body = request_body(cfg, route, system_prompt, zeroshot_model_inputs[cfg.dataset.name](dataset[i]))
        body["max_tokens"] = cfg.eval.batch_max_tokens
        requests_list.append({"custom_id": str(i), "body": body})
    with (output_dir / "batch_requests.jsonl").open("w", encoding="utf-8") as f:
        for r in requests_list:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # OpenRouter rejects a batch whose JSON does not put endpoint and model before requests: https://openrouter.ai/docs/batch-quickstart
    payload = {
        "endpoint": "/v1/chat/completions", 
        "model": route[0], 
        "requests": requests_list
    }
    resp = httpx.post(
        "https://openrouter.ai/api/v1/batches",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json=payload,
        timeout=600,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter batch submit failed {resp.status_code}: {resp.text}")
    submit = resp.json()
    (output_dir / "batch_submit.json").write_text(json.dumps(submit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Submitted {len(requests_list)} requests as batch {submit['id']} (model {route[0]}, status {submit.get('status')}); requests in {output_dir / 'batch_requests.jsonl'}")
    return submit["id"]


# A batch that has reached one of these will not change again.
BATCH_TERMINAL = ("completed", "failed", "expired", "cancelled")
BATCH_POLL_SECONDS = 60
# Poll this many times before giving up; batch might just be getting ready
BATCH_NOT_FOUND_POLLS = 10


def wait_for_batch(batch_id: str, output_dir: Path) -> dict:
    """Poll the batch until it reaches a terminal status; save its final state, results included, as batch_results.json and return it."""
    not_found = 0
    while True:
        time.sleep(BATCH_POLL_SECONDS)
        resp = httpx.get(
            f"https://openrouter.ai/api/v1/batches/{batch_id}",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            timeout=600,
        )
        if resp.status_code == 404 and not_found < BATCH_NOT_FOUND_POLLS:
            not_found += 1
            print(f"batch {batch_id}: not visible yet (404, poll {not_found} of {BATCH_NOT_FOUND_POLLS})", flush=True)
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter batch poll failed {resp.status_code}: {resp.text}")
        batch = resp.json()
        print(f"batch {batch_id}: {batch['status']} {batch['request_counts']}", flush=True)
        if batch["status"] in BATCH_TERMINAL:
            (output_dir / "batch_results.json").write_text(json.dumps(batch, ensure_ascii=False), encoding="utf-8")
            return batch


def batch_answer(item: dict | None) -> dict | None:
    """The response body of a batch result.

    Keep: an answer, a refusal (finish_reason content_filter, content null) and OpenRouter's moderation 403, treated as final outcomes. 
    Send again: a missing result, any other error, and an answer cut off at eval.batch_max_tokens (finish_reason length).
    Returns None when the line has to be sent again.
    """
    response = (item or {}).get("response") or {}
    body = response.get("body") or {}
    if response.get("status_code") == 403:
        return body if "flagged_input" in ((body.get("error") or {}).get("metadata") or {}) else None
    if response.get("status_code") != 200:
        return None
    if (body.get("choices") or [{}])[0].get("finish_reason") == "length":
        return None
    return body


def finish_batch(
    client: openai.OpenAI, 
    cfg: DictConfig, 
    route: tuple[str, str], 
    system_prompt: str, 
    dataset, 
    output_dir: Path, 
    batch: dict
) -> None:
    """Write out.txt and raw.jsonl from a finished batch.

    A kept result goes through build_record with the body that was submitted for it. 
    A result batch_answer rejects is sent again through evaluate_example, which uses the full eval.max_tokens. 
    """
    usd_rates = load_usd_rates(USD_COSTS_CSV)
    mtok_weights = load_mtok_weights(MTOK_WEIGHTS_CSV)
    requests = {int(r["custom_id"]): r["body"] for r in map(json.loads, (output_dir / "batch_requests.jsonl").open(encoding="utf-8"))}
    responses = {int(r["custom_id"]): r for r in batch.get("results") or []}
    resent = []
    with (output_dir / "out.txt").open("w", encoding="utf-8") as out_f, (output_dir / "raw.jsonl").open("w", encoding="utf-8") as raw_f:
        for i in tqdm(range(len(dataset)), desc="Collecting batch"):
            dumped = batch_answer(responses.get(i))
            if dumped is None:
                resent.append(i)
                pred, record = evaluate_example(client, cfg, route, usd_rates, mtok_weights, dataset, i, system_prompt)
            else:
                pred, record = build_record(cfg, usd_rates, mtok_weights, i, requests[i], dumped, 0.0)
            _write_line(out_f, raw_f, pred, record)
    print(f"batch collected: {len(dataset) - len(resent)} lines from the batch, {len(resent)} sent again: {resent[:20]}{' ...' if len(resent) > 20 else ''}")


def evaluate_example(
    client: openai.OpenAI, 
    cfg: DictConfig, 
    route: tuple[str, str], 
    usd_rates: dict, 
    mtok_weights: dict[str, float], 
    dataset, 
    i: int, 
    system_prompt: str
) -> tuple[str, dict]: # reviewed 15 Sep 2026
    user_content = zeroshot_model_inputs[cfg.dataset.name](dataset[i])
    body = request_body(cfg, route, system_prompt, user_content)
    t0 = time.time()
    try:
        dumped = send(client, body).model_dump()
    except openai.PermissionDeniedError as e:
        dumped = e.response.json() # cases where openrouter "flags" the input are treated as unanswered -> fallback predictions
        if "flagged_input" not in ((dumped.get("error") or {}).get("metadata") or {}):
            raise
    return build_record(cfg, usd_rates, mtok_weights, i, body, dumped, (time.time() - t0) / 60)


def build_record(
    cfg: DictConfig, 
    usd_rates: dict, 
    mtok_weights: dict[str, float], 
    i: int, 
    body: dict, 
    dumped: dict, 
    walltime_min: float
) -> tuple[str, dict]:
    """The out.txt line and raw.jsonl record for one line's response; common to sync and batch.

    A response with no content goes to out.txt whole, which the dataset's parser reads as unparseable and scores as fallback.
    """
    record = {
        "idx": i, 
        "request": body, 
        "cost": {"mtok": 0.0, "usd": 0.0}, 
        "walltime_min": walltime_min, 
        "response": dumped
    }
    if dumped.get("usage") is not None and not cfg.model.is_local: # a local model's calls are not billed
        usage = parse_openai_usage(json.dumps(dumped).encode())
        record["cost"] = {
            "mtok": calculate_mtok_cost(usage, mtok_weights), 
            "usd": calculate_usd_cost(usage, cfg.model.name, usd_rates)
        }
    content = ((dumped.get("choices") or [{}])[0].get("message") or {}).get("content")
    if content is None:
        return json.dumps(dumped, ensure_ascii=False), record
    return zeroshot_pred_parsers[cfg.dataset.name](content), record


def _write_line(out_f, raw_f, pred: str, record: dict, keep_raw: bool = True) -> None:
    """Append one line's prediction and record, the same way for a sync call and a batch result. keep_raw=False writes the prediction alone."""
    out_f.write((json.dumps(pred, ensure_ascii=False) if "\n" in pred else pred) + "\n") # newlines break out.txt parsing
    if keep_raw:
        raw_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_f.flush()
    raw_f.flush()


LOCAL_CHUNK = 50_000 # workload lines per llm.generate call; out.txt grows chunk by chunk


def run_local(
    cfg: DictConfig,
    route: tuple[str, str],
    system_prompt: str,
    dataset,
    out_f,
    raw_f,
    usd_rates: dict,
    mtok_weights: dict[str, float],
) -> None:
    """
    A prompt that does not fit in max_model_len together with eval.max_tokens is not generated; its response is an error with no content and is treated as fallback.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=cfg.model.path) # dtype "auto" reads the student's config.json
    sampling_params = SamplingParams(**OmegaConf.to_container(cfg.model.sampling_params), max_tokens=cfg.eval.max_tokens)
    tokenizer = llm.get_tokenizer()
    chat_template_kwargs = OmegaConf.to_container(cfg.model.chat_template_kwargs)
    max_model_len = llm.model_config.max_model_len
    with tqdm(total=len(dataset), desc="Evaluating") as progress:
        for start in range(0, len(dataset), LOCAL_CHUNK):
            idxs = range(start, min(start + LOCAL_CHUNK, len(dataset)))
            bodies = [request_body(cfg, route, system_prompt, zeroshot_model_inputs[cfg.dataset.name](dataset[i])) for i in idxs]
            rendered = [tokenizer.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True, **chat_template_kwargs) for body in bodies]
            prompt_token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            fits = [len(ids) + cfg.eval.max_tokens <= max_model_len for ids in prompt_token_ids]
            # vllm returns outputs in input order
            outputs = iter(llm.generate([{"prompt_token_ids": ids} for ids, fit in zip(prompt_token_ids, fits) if fit], sampling_params, use_tqdm=True))
            for i, body, ids, fit in zip(idxs, bodies, prompt_token_ids, fits):
                if fit:
                    completion = next(outputs).outputs[0]
                    dumped = {
                        "choices": [{"message": {"role": "assistant", "content": completion.text}, "finish_reason": completion.finish_reason}],
                        "usage": {"prompt_tokens": len(ids), "completion_tokens": len(completion.token_ids), "total_tokens": len(ids) + len(completion.token_ids)},
                    }
                else:
                    dumped = {"error": {"message": f"prompt of {len(ids)} tokens plus max_tokens {cfg.eval.max_tokens} exceeds max_model_len {max_model_len}"}}
                pred, record = build_record(cfg, usd_rates, mtok_weights, i, body, dumped, 0.0)
                _write_line(out_f, raw_f, pred, record, keep_raw=i % cfg.eval.raw_every == 0)
            progress.update(len(idxs))


@hydra.main(version_base=None, config_path="configs", config_name="zeroshot_eval/default")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    # region 1. create or find existing output dir [reviewed 15 Sep 2026]
    if cfg.model.is_local and cfg.resume is not None:
        raise SystemExit("resume is not supported for a local model; start a fresh run")
    if cfg.resume is None:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    else:
        output_dir = Path(cfg.resume).expanduser().resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"resume dir does not exist: {output_dir}")
    # endregion

    # region 2. client and req fields
    if cfg.model.is_local:
        route = (cfg.model.name, None) # no provider to pin
    else:
        if "OPENROUTER_API_KEY" not in os.environ:
            raise SystemExit('OPENROUTER_API_KEY is not set')
        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], max_retries=cfg.eval.max_retries)
        route = OPENROUTER_ROUTES[cfg.model.name]
    # endregion

    # region 3. prompt and data [reviewed 15 Sep 2026]
    system_prompt = jinja2.Environment(loader=jinja2.FileSystemLoader(PROMPTS_DIR), keep_trailing_newline=True).get_template(f"{cfg.prompt}.md").render()
    dataset_config = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_config["root"] = DATASETS_DIR / dataset_config["root"]
    dataset = load_from_config({**dataset_config, "provide_labels": False})
    # endregion

    if cfg.eval.batch and not (output_dir / "raw.jsonl").exists():
        submitted = output_dir / "batch_submit.json"
        batch_id = json.loads(submitted.read_text())["id"] if submitted.is_file() else submit_batch(cfg, route, system_prompt, dataset, output_dir)
        finish_batch(client, cfg, route, system_prompt, dataset, output_dir, wait_for_batch(batch_id, output_dir))

    # region 4. run files: one prediction and one raw record per line, appended in dataset order [reviewed 15 Sep 2026]
    out_path = output_dir / "out.txt"
    raw_path = output_dir / "raw.jsonl"
    n_out = sum(1 for _ in out_path.open(encoding="utf-8")) if out_path.exists() else 0
    n_raw = sum(1 for _ in raw_path.open(encoding="utf-8")) if raw_path.exists() else 0
    assert n_out == n_raw, f"out/raw line mismatch: {n_out} vs {n_raw}, fix manually before resume"
    assert n_out <= len(dataset), f"resume has {n_out} lines but the sample has {len(dataset)}"
    print(f"{'Resuming' if cfg.resume is not None else 'Starting'} at {output_dir}  done={n_out}/{len(dataset)}")
    # endregion

    # region 5. one call per remaining line, appended in dataset order [reviewed 15 Sep 2026]
    usd_rates = load_usd_rates(USD_COSTS_CSV)
    mtok_weights = load_mtok_weights(MTOK_WEIGHTS_CSV)
    budget_mtok = cfg.get("api_budget_mtok")
    spent = sum(json.loads(line)["cost"]["mtok"] for line in raw_path.open(encoding="utf-8")) if n_raw else 0.0
    stopped_early = False
    with out_path.open("a", encoding="utf-8") as out_f, raw_path.open("a", encoding="utf-8") as raw_f:
        if cfg.model.is_local:
            run_local(cfg, route, system_prompt, dataset, out_f, raw_f, usd_rates, mtok_weights)
        else:
            def run(i):
                return evaluate_example(client, cfg, route, usd_rates, mtok_weights, dataset, i, system_prompt)

            with ThreadPoolExecutor(max_workers=cfg.eval.concurrency) as pool:
                # map hands results back in submission order
                for expected_idx, (pred, record) in enumerate(tqdm(pool.map(run, range(n_out, len(dataset))), desc="Evaluating", initial=n_out, total=len(dataset)), start=n_out):
                    assert expected_idx == record["idx"], f"out/raw line mismatch: {expected_idx} vs {record['idx']}"
                    _write_line(out_f, raw_f, pred, record, keep_raw=expected_idx % cfg.eval.raw_every == 0)
                    spent += record["cost"]["mtok"]
                    if budget_mtok is not None and spent >= budget_mtok:
                        pool.shutdown(cancel_futures=True)
                        stopped_early = True
                        print(f"\nbudget reached: {spent:.3f} of {budget_mtok:g} mtok after {expected_idx + 1:,} of {len(dataset):,} lines")
                        break
    # endregion

    # region 6. metrics [reviewed 15 Sep 2026]
    metrics = {} if stopped_early else scorers[cfg.dataset.name](dataset_config, out_path)
    records = [json.loads(line) for line in raw_path.open(encoding="utf-8")]
    metrics["cost"] = {unit: sum(r["cost"][unit] for r in records) for unit in ("mtok", "usd")}
    if not cfg.model.is_local: # a local run keeps only every raw_every-th record
        metrics["walltime_min"] = sum(r["walltime_min"] for r in records)
    metrics["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=True).stdout.strip()
    metrics["launch_command"] = os.environ.get("LAUNCH_COMMAND")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Metrics saved to {metrics_path}")
    # endregion



if __name__ == "__main__":
    main()
