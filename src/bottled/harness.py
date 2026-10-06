# region imports and constants [reviewed 12 Sep 2026]
import json


from dataclasses import dataclass
from pathlib import Path
from typing import Callable


OPENCODE_MCP_PATH = "/mcp"
_OPENCODE_SYSTEM_PROMPT = """You are an autonomous agent."""
OPENCODE_AGENT = "solver"
# see https://opencode.ai/docs/config#models
# A streamed model response that stalls between chunks holds the turn open indefinitely. 
# let's avoid this
OPENCODE_CHUNK_TIMEOUT_MS = 60_000 # chunkTimeout aborts the request when no chunk arrives within the window,
OPENCODE_REQUEST_TIMEOUT_MS = 300_000 # timeout bounds the whole request. 
# endregion

# region dataclasses [reviewed 12 Sep 2026]

@dataclass(frozen=True)
class Bind: # reviewed 12 Sep 2026

    host: Path
    sandbox: str
    readonly: bool

@dataclass(frozen=True) # reviewed 12 Sep 2026
class Harness:
    binary: str  # resolved inside the sandbox as {sandbox_conda}/bin/<binary>
    resume_loop: str  # bash script; see _OPENCODE_RESUME_LOOP
    model_arg: Callable[[str], str]  # cfg.model.name -> CLI --model value
    format_event: Callable[[dict], str]  # stream-json event -> log line
    write_settings: Callable # scaffold harness config into workspace_dir
    mcp_transport: str  # mcp_server.py --transport value this harness's client expects
    mcp_path: str  

# endregion


def _format_opencode_event(event: dict) -> str: # display opencode events in pretty text
    event_type = event.get("type", "unknown")
    part = event.get("part", {})
    part_type = part.get("type", "")

    if event_type == "step_start":
        return f"[opencode/step_start] session={event.get('sessionID', 'unknown')}"

    if event_type == "step_finish":
        return f"[opencode/step_finish] session={event.get('sessionID', 'unknown')}"

    if part_type == "text":
        text = part.get("text", "").replace("\n", " ")
        return f"[opencode/text] {text}" if text else "[opencode/text]"

    if part_type == "reasoning":
        text = part.get("text", "").replace("\n", " ")
        return f"[opencode/reasoning] {text}" if text else "[opencode/reasoning]"

    if part_type == "tool":
        tool = part.get("tool", "unknown")
        state = part.get("state", {})
        tool_input = state.get("input", {})
        if tool == "bash" and isinstance(tool_input, dict) and tool_input.get("command"):
            return f"[opencode/tool_use] {tool} command={tool_input['command']}"
        return f"[opencode/tool_use] {tool} status={state.get('status', 'unknown')}"

    return f"[opencode/{event_type}] {json.dumps(event, ensure_ascii=False)}"


def opencode_provider(model_name: str) -> str: # reviewed 12 Sep 2026
    # set to "openrouter" for openrouter models; see https://opencode.ai/docs/providers/
    assert model_name.startswith("openrouter/"), "Only OpenRouter models are supported"
    return "openrouter"


def _write_opencode_settings(paths, run_info) -> None: # reviewed 14 Sep 2026
    provider = opencode_provider(run_info.model_name)
    model = f"{provider}/{run_info.model_name}"
    model_entry = {"name": run_info.model_name} # constructed as https://opencode.ai/docs/providers/#example
    if run_info.model_limit is not None:
        model_entry["limit"] = run_info.model_limit # {"context": ..., "output": ...}, see https://opencode.ai/docs/providers/#example
    if provider == "openrouter" and run_info.harness_thinking:
        # see https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#controlling-reasoning-tokens
        model_entry["options"] = {"reasoning": {"effort": run_info.harness_thinking}}
    config = {
        # region provider + small model: https://opencode.ai/docs/config/#models [reviewed 14 Sep 2026]
        "provider": {
            provider: {
                "options": {
                    "chunkTimeout": OPENCODE_CHUNK_TIMEOUT_MS,
                    "timeout": OPENCODE_REQUEST_TIMEOUT_MS,
                },
                "models": {run_info.model_name: model_entry},
            }
        },
        "small_model": model, # for lightweight tasks; see https://opencode.ai/docs/config#models
        # endregion 
        # region agent [reviewed 12 Sep 2026]
        "agent": { # https://opencode.ai/docs/agents#json
            OPENCODE_AGENT: {
                "mode": "primary",
                "description": "Narrates its reasoning while working.",
                "prompt": _OPENCODE_SYSTEM_PROMPT,
                "tools": {"tools_get_time_budget_min": False, "tools_get_api_budget_mtok": False} if run_info.hide_budget_tools else {}, # see https://opencode.ai/docs/config#models
            }
        },
        # endregion 
        # region mcp [reviewed 12 Sep 2026]
        "mcp": { # https://opencode.ai/docs/mcp-servers/#enable
            "tools": {
                "type": "remote",
                "url": f"http://127.0.0.1:{run_info.agent_mcp_port}{OPENCODE_MCP_PATH}",
                "enabled": True,
            }
        },
        # endregion
        # region permissions [added 19 Sep 2026]
        "permission": {"external_directory": "allow"}, # https://opencode.ai/docs/permissions/#external-directories
        # endregion
    }
    (paths.workspace_dir / "opencode.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

# region  final deliverables [reviewed 14 Sep 2026]

_OPENCODE_RESUME_LOOP = r'''
opencode=$1; model=$2 # unpack args
done_flag=/task/.task_complete
tmp="${TMPDIR:-/tmp}/agent_stream.jsonl"
rm -f "$done_flag"
session=""
while :; do # run forever
  if [ -z "$session" ]; then # first time, read the task from stdin
    "$opencode" run --model "$model" --format json --auto --agent __AGENT__ < /task/TASK.md | tee "$tmp"
  else
    "$opencode" run --model "$model" --format json --auto --agent __AGENT__ --session "$session" "Continue the task." < /dev/null | tee "$tmp"
  fi
  [ -f "$done_flag" ] && break # if the task is complete, break the loop
  session=$(grep -o '"sessionID":"[^"]*"' "$tmp" | tail -1 | cut -d'"' -f4) # grab the last session ID
  [ -z "$session" ] && break
done
'''.replace("__AGENT__", OPENCODE_AGENT)


HARNESSES: dict[str, Harness] = { # reviewed 14 Sep 2026
    "opencode": Harness(
        binary="opencode",
        resume_loop=_OPENCODE_RESUME_LOOP,
        model_arg=lambda name: f"{opencode_provider(name)}/{name}",
        format_event=_format_opencode_event,
        write_settings=_write_opencode_settings,
        mcp_transport="streamable-http",
        mcp_path=OPENCODE_MCP_PATH,
    ),
}

# endregion