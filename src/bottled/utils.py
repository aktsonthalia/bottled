# region imports and constants [reviewed 13 Sep 2026]
import csv
import json
import socket
import subprocess


from pathlib import Path

# Columns that do not contain price info.
_NON_RATE_COLUMNS = {"model", "tag"}

# endregion

# region cost accounting [reviewed 14 Sep 2026]

def load_usd_rates(*paths: Path) -> dict[str, dict[str, float]]: # reviewed 13 Sep 2026
    """{model -> {rate column -> USD per Mtok}}, merged over every file given.
    """
    usd_rates: dict[str, dict[str, float]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                model = row["model"]
                if model in usd_rates:
                    raise ValueError(f"{model} priced in more than one table, last in {path}")
                usd_rates[model] = {
                    k: float(v) for k, v in row.items() if k not in _NON_RATE_COLUMNS and v not in (None, "")
                } # blank cells mean the provider publishes no such rate; such keys are skipped
    return usd_rates


def calculate_usd_cost(usage: dict, model: str, usd_rates: dict[str, dict[str, float]]) -> float: # reviewed 14 Sep 2026
    assert model in usd_rates, f"No rates found for model {model}"
    rates = usd_rates[model]
    counter_rates = {
        "input_tokens": "base_input_per_mtok",
        "output_tokens": "output_per_mtok",
        "cache_read_input_tokens": "cache_hit_refresh_per_mtok",
        "cache_creation_input_tokens": "5m_cache_write_per_mtok", # for simplicity, we always use 5m cache prices
    }
    total_cost = 0.0
    for counter, column in counter_rates.items():
        tokens = usage[counter]
        if column not in rates:
            # If rate is not published but non-zero tokens are returned, something is wrong 
            if tokens > 0:
                raise ValueError(f"{model}: {tokens:,} {counter} but no {column} in the price table")
            continue
        total_cost += tokens * rates[column]
    return total_cost / 1e6


def load_mtok_weights(path: Path) -> dict[str, float]: # reviewed 12 Sep 2026
    """{usage counter -> weight}"""
    with path.open(encoding="utf-8") as f:
        return {row["counter"]: float(row["weight"]) for row in csv.DictReader(f)}


def parse_openai_usage(content): # reviewed 14 Sep 2026
    """Usage of an OpenAI-format response, in the counters stated in mtok_weights.csv. 
    OpenRouter's prompt_tokens counts cache reads and cache writes as well; these are taken out of input_tokens.
    """
    if not content:
        raise ValueError("Empty chat completions response body")
    body = content.decode("utf-8")
    usage = None
    if body.lstrip().startswith("{"): # if body is JSON, simply parse it and get json usage
        usage = json.loads(body).get("usage")
    else: # if body arrives as a stream of chunks, process as follows
        for line in body.splitlines(): # get usage from last chunk: https://openrouter.ai/docs/cookbook/administration/usage-accounting#usage-information
            if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"):
                usage = json.loads(line[6:]).get("usage")
    if not usage:
        raise ValueError("Chat completions response missing usage")
    try:
        # extract keys: https://openrouter.ai/docs/cookbook/administration/usage-accounting#response-format
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens")
        written = details.get("cache_write_tokens")
        # see https://developers.openai.com/api/docs/guides/prompt-caching#monitor-cache-performance
        return {
            "input_tokens": usage["prompt_tokens"] - cached - written, # input tokens that were not cached or read from cache
            "output_tokens": usage["completion_tokens"], # all output tokens, including reasoning tokens
            "cache_read_input_tokens": cached, # input tokens read from cache
            "cache_creation_input_tokens": written, # input tokens written to cache
        }
    except Exception as e:
        raise ValueError(f"Failed to parse usage for {content}: {e}") from e

        
def calculate_mtok_cost(usage: dict, weights: dict[str, float]) -> float: # reviewed 13 Sep 2026
    """Usage in normalised million-token units."""
    return sum((usage.get(counter)) * weight for counter, weight in weights.items()) / 1e6

# endregion

# region scoring [reviewed 13 Sep 2026]

def normalized_gain(metric: float, trivial: float, perfect: float) -> float:
    """A score as a fraction of the headroom between the task's trivial answer and its gold. The two limits come from charts/task_baselines.csv."""
    return (metric - trivial) / (perfect - trivial)

# endregion

# region housekeeping [reviewed 12 Sep 2026]

def allocate_free_port(host: str = "127.0.0.1") -> int: # reviewed 12 Sep 2026
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def stop_procs(procs): # reviewed 12 Sep 2026
    for proc in procs: # if process exists and is running, terminate it
        if proc is not None and proc.poll() is None:
            proc.terminate()
    for proc in procs: # kill processes that didn't terminate, after a 5s wait each, then wait.
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

# endregion