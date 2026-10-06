# region imports and constants [reviewed 12 Sep 2026]

import ipaddress
import json
import os
import time

from mitmproxy import ctx, http
from pathlib import Path

from bottled.openrouter_models import OPENROUTER_ROUTES
from bottled.utils import calculate_mtok_cost, load_mtok_weights, parse_openai_usage


CALL_HISTORY_PATH = Path(os.environ["CALL_HISTORY_PATH"])
PROXY_LOGS_JSONL_PATH = Path(os.environ["PROXY_LOGS_JSONL_PATH"])
ALLOWED_MODELS = set(json.loads(os.environ["API_ALLOWED_MODELS"]))
OPENROUTER_HOST = "openrouter.ai"
OPENROUTER_CHAT_PATH = "/api/v1/chat/completions"
if ALLOWED_MODELS & OPENROUTER_ROUTES.keys() and "OPENROUTER_API_KEY" not in os.environ:
    raise SystemExit('OPENROUTER_API_KEY is not set')
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# endregion

# region URL restrictions [reviewed 12 Sep 2026]

RUN_SECRET = os.environ["RUN_SECRET"]

BLOCKED_URL_PATTERNS = [[s.lower() for s in group] for group in json.loads(os.environ.get("BLOCKED_URL_PATTERNS", "[]"))]

def blocked_url_match(flow):
    url = flow.request.pretty_url.lower()
    for group in BLOCKED_URL_PATTERNS:
        if all(substring in url for substring in group):
            return group
    return None

# endregion

# region budgeting [reviewed 14 Sep 2026]

LEDGER_PATH = Path(os.environ["API_LEDGER_PATH"])
MTOK_WEIGHTS_CSV = Path(os.environ["API_MTOK_WEIGHTS_CSV"])
ENFORCE_BUDGETS = {"1": True, "0": False}[os.environ["ENFORCE_BUDGETS"]]
mtok_weights = load_mtok_weights(MTOK_WEIGHTS_CSV)

# endregion

# region routing, auth and billing [reviewed 14 Sep 2026]

def is_openrouter_chat(flow): # reviewed 12 Sep 2026
    if flow.request.host != OPENROUTER_HOST or flow.request.method != "POST":
        return False
    return flow.request.path.split("?", 1)[0] == OPENROUTER_CHAT_PATH


def _apply_reasoning(payload): # reviewed 13 Sep 2026
    """Put the caller's reasoning request into an OpenRouter-compatible `reasoning` object if needed.
    """
    # if called via OpenAI SDK, needs to be converted into Openrouter format
    effort = payload.pop("reasoning_effort", None) 
    if effort is not None and "reasoning" not in payload:
        payload["reasoning"] = {"effort": effort}


def route_to_openrouter(flow): # reviewed 14 Sep 2026
    """Ready an OpenAI-format call for OpenRouter: 
    - the model name becomes its slug, 
    - the provider is pinned, 
    - usage is requested 
    - the run secret is swapped for the real key."""
    payload = json.loads(flow.request.content)
    slug, provider = OPENROUTER_ROUTES[payload["model"]]
    payload["model"] = slug
    payload["provider"] = {"only": [provider], "allow_fallbacks": False} # https://openrouter.ai/docs/guides/routing/provider-selection
    # payload["usage"] = {"include": True} # deprecated: https://openrouter.ai/docs/cookbook/administration/usage-accounting
    _apply_reasoning(payload)
    flow.request.content = json.dumps(payload).encode()
    flow.request.headers["authorization"] = f"Bearer {OPENROUTER_API_KEY}"


def openai_request_model(flow) -> str: # reviewed 12 Sep 2026
    payload = json.loads(flow.request.content)
    if "model" not in payload:
        raise ValueError("Request payload missing model")
    return payload["model"]


class SwapKey: # reviewed 14 Sep 2026
    def __init__(self): # reviewed 12 Sep 2026
        # self.lock = threading.Lock()
        if not LEDGER_PATH.exists():
            raise FileNotFoundError(f"Ledger must be pre-created: {LEDGER_PATH}")

    def running(self): # reviewed 13 Sep 2026
        # see https://docs.mitmproxy.org/stable/api/events.html#LifecycleEvents.running
        ctx.options.update(stream_large_bodies="3m")
        print("[proxy] stream_large_bodies=3m", flush=True)

    def _bill_call(self, usage_entry): # reviewed 12 Sep 2026
        # with self.lock:
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        ledger["spent_mtok"] += usage_entry["cost_mtok"]
        ledger["calls"].append(usage_entry)
        LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        if ledger["spent_mtok"] >= ledger["budget_mtok"]:
            LEDGER_PATH.with_name(".budget_exhausted").touch()

    def _budget_exhausted(self): # reviewed 12 Sep 2026
        # with self.lock:
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        return ledger["spent_mtok"] >= ledger["budget_mtok"]

    def _log_call(self, flow, model): # reviewed 12 Sep 2026
        entry = {
            "time": time.time(),
            "model": model,
            "path": flow.request.path,
            "status_code": flow.response.status_code,
            "request": flow.request.text,
            "response": flow.response.text,
        }
        # with self.lock:
        with CALL_HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_request(self, flow):
        with PROXY_LOGS_JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"url": flow.request.pretty_url, "status_code": flow.response.status_code}) + "\n")

    def request(self, flow): # reviewed 14 Sep 2026
        """do this to the request before sending to the target"""
        print(f"[req] {flow.request.pretty_url}", flush=True)
        # region blocking [reviewed 12 Sep 2026]
        blocked = blocked_url_match(flow)
        if blocked is not None:
            print(f"[blocked] {flow.request.pretty_url} (pattern: {blocked})", flush=True)
            flow.response = http.Response.make(403, b"This resource is prohibited. Do not attempt to circumvent.")
            return
        # endregion
        # a localhost address here could reach the host machine's other services
        host = flow.request.host
        try:
            address = ipaddress.ip_address(host)
            is_localhost = address.is_loopback or address.is_unspecified
        except ValueError: # not an IP address
            is_localhost = "localhost" in host.lower()
        if is_localhost:
            print(f"[blocked] {flow.request.pretty_url} (localhost)", flush=True)
            flow.response = http.Response.make(403, b"The proxy does not forward requests to localhost. Call localhost without the proxy.")
            return
        if is_openrouter_chat(flow):
            # block if key incorrect
            if flow.request.headers.get("authorization", "") != f"Bearer {RUN_SECRET}":
                flow.response = http.Response.make(403, b"Forbidden")
                return
            # block if budget exhausted
            if ENFORCE_BUDGETS and self._budget_exhausted():
                flow.response = http.Response.make(402, b"API budget exceeded")
                return
            # block if model not allowed
            model = openai_request_model(flow)
            if model not in ALLOWED_MODELS or model not in OPENROUTER_ROUTES:
                flow.response = http.Response.make(403, f"Model not allowed: {model}".encode())
                return
            flow.metadata["billing_model"] = model
            route_to_openrouter(flow)
            return

    def response(self, flow): # reviewed 12 Sep 2026
        """do this to the response before sending it to the caller.
        Usually: log the API call, and bill it in mtoks."""
        model = flow.metadata.get("billing_model")
        if model is None:
            self._log_request(flow)
            return
        if not flow.response:
            return
        self._log_call(flow, model)
        if flow.response.status_code != 200:
            return
        try:
            usage = parse_openai_usage(flow.response.content)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"[proxy] failed to parse usage for {flow.request.path}: {e} and will not bill", flush=True)
            return
        cost_mtok = calculate_mtok_cost(usage, mtok_weights)
        self._bill_call(
            {
                "path": flow.request.path,
                "model": model,
                "usage": usage,
                "cost_mtok": cost_mtok,
            }
        )
        # show the caller what this call was billed: each counter in millions of tokens, and the weighted total that counts against the API budget
        billed = {counter.removesuffix("_tokens") + "_mtok": count / 1e6 for counter, count in usage.items()}
        billed["total_cost_mtok"] = cost_mtok
        body = flow.response.content.decode("utf-8")
        if body.lstrip().startswith("{"):
            data = json.loads(body)
            data["harness_billed_usage"] = billed
            flow.response.content = json.dumps(data).encode("utf-8")
        else:
            # usage arrives in the last data chunk, make that chunk carry the billed usage too
            lines = body.split("\n")
            last = max(i for i, line in enumerate(lines) if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"))
            assert lines[last].startswith("data: "), "data: assumption violated"
            data = json.loads(lines[last][6:])
            data["harness_billed_usage"] = billed
            lines[last] = "data: " + json.dumps(data)
            flow.response.content = "\n".join(lines).encode("utf-8")


addons = [SwapKey()]

# endregion