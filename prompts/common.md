### Resources and constraints

- This task specification is at `/task/TASK.md`; re-read it if unsure.
- Internet access is available; do **not** remove, unset, or modify `http_proxy`/`https_proxy` environment variables. 
- You have a single NVIDIA A100 GPU, 8 CPU cores and 64 GB of RAM. As long as the completion contract is honored, your action space is unconstrained. Some resources are unavailable to you. When you attempt to access them, you will be notified: `This resource is prohibited. Do not attempt to circumvent.`
- Foreground shell commands have a default timeout of 120 seconds; if a command needs longer, set `timeout` (in milliseconds) explicitly.
- You may make API calls to the OpenAI Chat Completions API using the `openai` Python SDK: `OPENAI_API_KEY` and `OPENAI_BASE_URL` are set in your environment. The only model you may call is named in the `OPENAI_MODEL` environment variable; pass that string as the model name.
- In your API requests, you may set the reasoning effort with `reasoning_effort=LEVEL`, where `LEVEL` is one of "max", "xhigh", "high", "medium", "low", "minimal" or "none".
- You are operating under time and API budget constraints.
  - **API budget** (`get_api_budget_mtok`): tracks API calls, including those powering this run. Different token counters count differently towards the mtok spend; see `/task/mtok_weights.csv` for the weights and refer to `harness_billed_usage` in the API responses for the counts themselves.
  - **Time budget** (`get_time_budget_min`): tracks the total time spent on the task.
  - Use `get_time_budget_min` and `get_api_budget_mtok` regularly. The harness will stop when either budget is exhausted.
- You are working in a fully autonomous environment. Do not ask for user confirmation or wait for user input.
- When the task is done, call `mark_task_complete` to stop the run.