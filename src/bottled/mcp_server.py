# region imports and args [reviewed 12 Sep 2026]

import argparse
import json
import time
import uvicorn


from mcp.server.fastmcp import FastMCP
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--time-budget-min", type=float, required=True, help="agent's total time budget in minutes")
parser.add_argument("--run-start-time", type=float, required=True, help="time at which the agent was launched")
parser.add_argument("--api-ledger-path", type=Path, required=True)
parser.add_argument("--workspace-dir", type=Path, required=True)
parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="sse")
parser.add_argument("--uds", type=str, default="")
args = parser.parse_args()

mcp = FastMCP("tools")

# endregion

# region tools [reviewed 12 Sep 2026]

@mcp.tool()
def get_time_budget_min() -> dict: # reviewed 12 Sep 2026
    """Returns (all times in minutes):
    - total_time_budget_min: total time budget
    - used_time_min: time used so far
    - remaining_time_min: remaining time
    """
    total_time_budget_min = args.time_budget_min
    used_time_min = (time.time() - args.run_start_time) / 60
    remaining_time_min = total_time_budget_min - used_time_min
    return {
        "total_time_budget_min": total_time_budget_min,
        "used_time_min": used_time_min,
        "remaining_time_min": remaining_time_min,
    }


@mcp.tool()
def get_api_budget_mtok() -> dict: # reviewed 12 Sep 2026
    """Returns (all values in mtok where 1 mtok = 1 million tokens):
    - spent_mtok: mtok spent so far
    - budget_mtok: maximum mtok that can be spent
    - remaining_mtok: mtok still available

    Different token counters weigh differently towards the mtok spend; see `/task/mtok_weights.csv` for the weights.
    """
    ledger = json.loads(args.api_ledger_path.read_text(encoding="utf-8"))
    spent_mtok = ledger["spent_mtok"]
    budget_mtok = ledger["budget_mtok"]
    return {
        "spent_mtok": spent_mtok,
        "budget_mtok": budget_mtok,
        "remaining_mtok": budget_mtok - spent_mtok,
    }


@mcp.tool()
def mark_task_complete() -> dict: # reviewed 12 Sep 2026
    """Call when the task is done, and only then.

    The harness resumes after every turn ends, for any reason, unless this tool has been called.
    """
    (args.workspace_dir / ".task_complete").touch()
    return {"ok": True}

# endregion

# region main [reviewed 12 Sep 2026]
if __name__ == "__main__":
    if args.transport == "streamable-http":
        uvicorn.run(
            mcp.streamable_http_app(),
            uds=args.uds,
            timeout_keep_alive=300,
        )
    else:
        raise ValueError(f"Invalid transport: {args.transport}")
# endregion