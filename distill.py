"""Distillation baseline: fine-tune a small next-token predictor on a teacher's labels, then run it over the whole workload.

    python distill.py student=qwen3_0p6b task=mave group=mave_distillation_baseline
"""

import gc
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from tqdm import tqdm

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import bottled.constants # sets BOTTLED_ARTIFACTS_ROOT

CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def load_hf_model(model_cfg: DictConfig): # reviewed 19 Sep 2026
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name)
    model = AutoModelForCausalLM.from_pretrained(model_cfg.name, dtype=getattr(torch, model_cfg.dtype))
    return model.cuda(), tokenizer


# region build data # reviewed 19 Sep 2026

def load_teacher_labels(cfg: DictConfig) -> list[dict]: # reviewed 19 Sep 2026
    teacher_labels_dir = Path(cfg.teacher.labels_dir)
    rows, no_teacher_reply = [], 0
    for line in tqdm((Path(teacher_labels_dir) / "raw.jsonl").open(encoding="utf-8"), desc="Loading teacher labels"):
        record = json.loads(line)
        system_turn = next(m for m in record["request"]["messages"] if m["role"] == "system")["content"]
        user_turn = next(m for m in record["request"]["messages"] if m["role"] == "user")["content"]
        teacher_label = record["response"]["choices"][0]["message"]["content"]
        if teacher_label is None: # skip refusals, empty replies, etc.
            no_teacher_reply += 1
            continue
        rows.append({
            "idx": record["idx"],
            "system": system_turn,
            "user": user_turn,
            "teacher": teacher_label,
        })
    print(f"teacher labels {len(rows):,} kept, {no_teacher_reply:,} skipped for an empty reply")
    return rows


def prompt_completion_row(row: dict) -> dict: # reviewed 19 Sep 2026
    """One teacher label -> prompt-completion row for SFTTrainer."""
    return {
        "prompt": [
            {"role": "system", "content": row["system"]}, 
            {"role": "user", "content": row["user"]}
        ],
        "completion": [{"role": "assistant", "content": row["teacher"]}],
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    }


def build_prompt_completion_rows(rows: list[dict], tokenizer: AutoTokenizer, max_length: int) -> list[dict]: # reviewed 19 Sep 2026
    """All teacher's labels -> prompt-completion rows for SFTTrainer.

    We skip rows whose prompt and answer do not fit in max_length.
    """
    kept, dropped = [], 0
    for row in tqdm(rows, desc="Building training rows"):
        prompt_completion = prompt_completion_row(row)
        rendered = tokenizer.apply_chat_template(prompt_completion["prompt"], tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
        n_tokens = len(tokenizer(rendered + row["teacher"], add_special_tokens=False)["input_ids"]) # drop rows that won't fit
        if n_tokens > max_length:
            dropped += 1
            continue
        kept.append(prompt_completion)
    print(f"training rows {len(kept):,} kept, {dropped:,} dropped for exceeding max_length={max_length}")
    return kept


def split_rows(rows: list[dict], eval_fraction: float, seed: int) -> tuple[list[dict], list[dict]]: # reviewed 19 Sep 2026
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_eval = int(len(shuffled) * eval_fraction)
    return shuffled[n_eval:], shuffled[:n_eval]

# endregion

def train_student(
    cfg: DictConfig,
    model,
    tokenizer,
    train_ds: Dataset,
    eval_ds: Dataset,
    learning_rate: float,
    output_dir: Path
) -> tuple[Path, float]: # reviewed 19 Sep 2026
    """Fine-tune the student on the teacher's replies, keeping the epoch with the lowest eval_loss. 
    """
    student_dir = output_dir / "student"
    # https://huggingface.co/docs/trl/v1.13.0/en/sft_trainer#trl.SFTConfig
    args = SFTConfig(
        **{**OmegaConf.to_container(cfg.training), "learning_rate": learning_rate},
        output_dir=str(output_dir / "checkpoints"),
        packing=False, # https://huggingface.co/docs/trl/v1.13.0/en/sft_trainer#trl.SFTConfig.max_length
        bf16=True, 
        eval_strategy="epoch", # https://github.com/huggingface/transformers/blob/da58b8921e57ad1272c6e4a3250ff8a0cfc4c50b/src/transformers/trainer_utils.py#L389-L392
        save_strategy="epoch", # https://github.com/huggingface/transformers/blob/da58b8921e57ad1272c6e4a3250ff8a0cfc4c50b/src/transformers/trainer_utils.py#L395-L399
        save_only_model=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, 
        args=args, 
        train_dataset=train_ds, 
        eval_dataset=eval_ds, 
        processing_class=tokenizer
    )
    trainer.train()
    trainer.save_model(str(student_dir))
    tokenizer.save_pretrained(student_dir)
    print(f"student saved to {student_dir} (best eval_loss {trainer.state.best_metric:.4f})")
    return student_dir, trainer.state.best_metric


def hydra_dict(values: dict) -> str: # reviewed 19 Sep 2026
    """{"a": 1, "b": "x"} -> "{a:1,b:x}"."""
    return "{" + ",".join(f"{key}:{value}" for key, value in values.items()) + "}"


def run_inference(cfg: DictConfig, student_dir: Path, output_dir: Path) -> None:
    """Run zeroshot_eval.py over dataset.sample with the student."""
    inference_start = time.time()
    sampling_params = OmegaConf.to_container(cfg.inference.sampling_params)
    max_tokens = sampling_params.pop("max_tokens") # zeroshot_eval's eval.max_tokens
    subprocess.run(
        [
            sys.executable, str(Path(__file__).parent / "zeroshot_eval.py"),
            "--config-name", f"zeroshot_eval/{cfg.dataset.name}_zeroshot",
            f"group={cfg.group}",
            f"dataset.sample={cfg.dataset.sample}",
            "harness_thinking=null",
            "eval.raw_every=1000",
            "zeroshot_eval/model@model=local_distilled_student",
            f"model.path={student_dir}",
            f"model.sampling_params={hydra_dict(sampling_params)}",
            f"eval.max_tokens={max_tokens}",
            f"hydra.run.dir={output_dir / 'inference'}",
        ],
        check=True,
        env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    )
    metrics_path = output_dir / "inference" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["walltime_min"] = (time.time() - inference_start) / 60
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"inference walltime {metrics['walltime_min']:.1f} min, written to {metrics_path}")


@hydra.main(version_base=None, config_path="configs/distillation_baseline", config_name="default")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    run_start = time.time()
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    if cfg.inference_only is not None:
        selection = json.loads((Path(cfg.inference_only) / "selection.json").read_text())
        print(f"inference only: student {selection['selected']['student_dir']} from {cfg.inference_only}")
        run_inference(cfg, Path(selection["selected"]["student_dir"]), output_dir)
        return

    # region load model and print info # reviewed 17 Sep 2026
    rows = load_teacher_labels(cfg)
    print(f"teacher    {cfg.teacher.labels_dir}")
    print(f"teacher labels     {len(rows):,}")
    print()

    student_model, tokenizer = load_hf_model(cfg.student)
    params = sum(p.numel() for p in student_model.parameters())
    print(f"model      {cfg.student.name}")
    print(f"parameters {params:,}")
    print(f"dtype      {next(student_model.parameters()).dtype}")
    print(f"device     {next(student_model.parameters()).device}")
    print(f"vocab      {len(tokenizer):,}")
    print(f"max length {tokenizer.model_max_length:,}")
    # endregion 

    messages = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
    print(f"\nprompt with add_generation_prompt=True and {CHAT_TEMPLATE_KWARGS}:")
    print(prompt)
    print(f"({len(tokenizer(prompt, add_special_tokens=False)['input_ids'])} tokens)")

    train_rows, eval_rows = split_rows(rows, cfg.eval_fraction, cfg.training.seed)
    # train_ds = Dataset.from_list(build_prompt_completion_rows(train_rows, tokenizer, cfg.training.max_length))
    train_ds = Dataset.from_list([prompt_completion_row(row) for row in train_rows])
    eval_ds = Dataset.from_list([prompt_completion_row(row) for row in eval_rows])
    print(f"train {len(train_ds):,} / eval {len(eval_ds):,}")
    print(f"\none training row:\n{json.dumps(train_ds[0], indent=2)[:700]}")
    # endregion

    # region LR sweep # reviewed 19 Sep 2026
    sweep_results = []
    for learning_rate in cfg.sweep.learning_rates:
        print(f"\n=== learning rate {learning_rate:.1e}")
        sweep_point_start = time.time()
        student_model, _ = load_hf_model(cfg.student)
        student_dir, eval_loss = train_student(
            cfg,
            student_model,
            tokenizer,
            train_ds,
            eval_ds,
            learning_rate,
            output_dir / f"lr_{learning_rate:.1e}"
        )
        sweep_results.append(
            {"learning_rate": learning_rate, 
            "eval_loss": eval_loss,
            "student_dir": str(student_dir),
            "walltime_min": (time.time() - sweep_point_start) / 60}
        )
    # endregion

    # region select the student with the lowest eval_loss
    best = min(sweep_results, key=lambda result: result["eval_loss"])
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True)
    selection = {
        "selected": best,
        "sweep": sweep_results,
        "walltime_min": (time.time() - run_start) / 60, # the whole run
        "commit": git_head.stdout.strip() if git_head.returncode == 0 else None, # None outside a git checkout
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    print(f"\nselected learning rate {best['learning_rate']:.1e} (eval_loss {best['eval_loss']:.4f}): {best['student_dir']}")
    # endregion

    del student_model
    gc.collect() 
    torch.cuda.empty_cache()
    run_inference(cfg, Path(best["student_dir"]), output_dir)

if __name__ == "__main__":
    main()
