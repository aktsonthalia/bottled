# BOTTLED

Code for the paper, "Agent in a Bottle: Can LLM Agents Turn Their Capabilities Into Cheap, Scalable Artifacts?"

## ⚙️ Setup

### 1. Machine settings

Copy the machine config template:

```bash
cp configs/machine.yaml.example configs/machine.yaml
```

In `configs/machine.yaml`, 

- set `artifacts_dir` (datasets -> `<artifacts_dir>/data`, results -> `<artifacts_dir>/results`) 
- set `conda_envs_root` (the directory where you store your conda environments)

### 2. Main environment

This environment runs `agent_eval.py`, `zeroshot_eval.py` and `distill.py`. It is built at `<conda_envs_root>/bottled`:

```bash
nohup bash scripts/create_main_env.sh > main_env_build.log 2>&1 &
tail -f main_env_build.log
```

Activate it:

```bash
conda activate <PATH>
```



### 3. Agents' environment

This environment is used by agents.

Set `agents_env_name` in `configs/machine.yaml`; the environment is built at `<conda_envs_root>/<agents_env_name>`. Then, with the main environment active, run:

```bash
nohup bash scripts/create_solvers_base.sh > agents_env_build.log 2>&1 &
tail -f agents_env_build.log
```



### 4. API key

Currently, this repo only supports [OpenRouter](https://openrouter.ai) models. Export your key:

```bash
export OPENROUTER_API_KEY=...
```

In BOTTLED runs, agents do not get your key directly. They get a per-run secret, which the proxy swaps for your real key.

### 5. Data

Task workloads are built under `<artifacts_dir>/data/<task>/views`.

#### ESCI

```bash
nohup bash scripts/prepare_data/esci.sh > esci_build.log 2>&1 &
```



#### MAVE

```bash
nohup bash scripts/prepare_data/mave.sh > mave_build.log 2>&1 &
```



#### RAID

```bash
nohup bash scripts/prepare_data/raid.sh > raid_build.log 2>&1 &
```



## ⚡ Running experiments

With the main environment active: 

- Replace `mave` with `esci` or `raid` for the other tasks
- models are listed in `configs/agent_eval/model/` (add yours the same way if needed)

BOTTLED (agent):

```bash
python agent_eval.py --config-name agent_eval/mave_bottled \
  agent_eval/model@model=openrouter_opus_5 \
  harness_thinking=high \
  group=mave_bottled
```

Zero-shot:

```bash
python zeroshot_eval.py --config-name zeroshot_eval/mave_zeroshot \
  zeroshot_eval/model@model=openrouter_opus_5 \
  harness_thinking=high \
  group=mave_zeroshot
```

Obtaining teacher labels:

```bash
python zeroshot_eval.py --config-name zeroshot_eval/mave_teacher_labels \
  zeroshot_eval/model@model=openrouter_z_ai_glm_5p3_flash \
  harness_thinking=high \
  group=mave_teacher_labels
```

Student training:

```bash
python distill.py \
  task=mave \
  student=smollm2_360m \
  teacher.labels_dir=results/mave_teacher_labels/<date>/<time> \
  group=mave_distillation \
  seed=42
```

Runs are written to `<artifacts_dir>/results/<group>/<date>/<time>`.

## Repo structure

```
.
├── charts/
│   ├── mtok_weights.csv            # weight of each usage counter in the mtok budget
│   └── openrouter_prices.csv       # USD per Mtok for each model, per usage counter
├── configs/
│   ├── agent_eval/
│   │   ├── harness/
│   │   │   └── opencode.yaml       # selects the opencode harness
│   │   ├── model/
│   │   │   ├── local_distilled_student.yaml  # local vLLM student for the distillation baseline
│   │   │   └── openrouter_*.yaml   # one per model: OpenRouter model name, context and output token limits
│   │   ├── default.yaml            # agent run settings: time and mtok budgets, harness, allowed models
│   │   └── {esci,mave,raid}_bottled.yaml  # per task: prompt and workload sample
│   ├── distillation_baseline/
│   │   ├── student/
│   │   │   └── {qwen3_0p6b,smollm2_360m}.yaml  # per student: HF model, SFT hyperparameters, sampling, learning rates
│   │   ├── task/
│   │   │   └── {esci,mave,raid}.yaml  # per task: teacher labels dir (set at launch) and workload sample
│   │   └── default.yaml            # distillation settings: seed, eval split, inference-only mode
│   ├── zeroshot_eval/
│   │   ├── default.yaml            # zero-shot run settings: max tokens, retries, concurrency, batch mode
│   │   ├── {esci,mave,raid}_teacher_labels.yaml  # per task: zero-shot labels for the full workload (distillation baseline)
│   │   ├── {esci,mave,raid}_zeroshot.yaml  # per task: prompt and 1,000-line sample
│   │   └── model -> ../agent_eval/model
│   ├── base.yaml                   # Hydra settings shared by all runs: group, reasoning effort, output directory
│   └── machine.yaml.example        # template for configs/machine.yaml: paths and agent conda env
├── prompts/
│   ├── common.md                   # resources and constraints, included by every bottled prompt
│   ├── {esci,mave,raid}_bottled.md # task prompt for the agent (becomes /task/TASK.md in sandboxes)
│   └── {esci,mave,raid}_zeroshot.md  # system prompt for the zero-shot model
├── scripts/
│   ├── prepare_data/
│   │   ├── esci.sh                 # builds the ESCI workload and samples (setup step 5)
│   │   ├── mave.sh                 # builds the MAVE workload and samples (setup step 5)
│   │   └── raid.sh                 # builds the RAID workload and samples (setup step 5)
│   ├── create_main_env.sh          # builds the main environment (setup step 2)
│   └── create_solvers_base.sh      # builds the agents' environment (setup step 3)
├── src/bottled/
│   ├── datasets/
│   │   ├── blocklists/
│   │   │   └── {esci,mave,raid}.py # URL patterns blocked by a proxy during the run
│   │   ├── __init__.py             # per-task registries
│   │   ├── common.py               # code shared by all tasks
│   │   └── {esci,mave,raid}.py     # dataset classes
│   ├── __init__.py
│   ├── constants.py                # repo root, machine settings from configs/machine.yaml, data directory
│   ├── harness.py                  # agent harness: opencode config and resume loop (other harnesses can be added in the future)
│   ├── mcp_server.py               # MCP tools for the agent
│   ├── openrouter_models.py        # supported models: OpenRouter slug and pinned provider
│   ├── run_proxy.py                # mitmproxy addon: URL blocklist, API key swap, mtok budget, call logging
│   └── utils.py                    # USD and mtok cost accounting, usage parsing, process housekeeping
├── agent_eval.py                   # bottled eval: runs the agent in a bwrap sandbox with time and mtok budgets, then scores its out.txt
├── distill.py                      # distillation baseline: fine-tune a small student on a teacher's zero-shot labels, pick the best learning rate, run it over the workload with zeroshot_eval.py
├── pyproject.toml                  # build system (setuptools)
├── requirements_bottled.txt        # pinned pip packages for the main environment
├── setup.py                        # installs the `bottled` package from src/
└── zeroshot_eval.py                # zero-shot eval: one model call per line of a sample (OpenRouter sync / OpenRouter batch / local vLLM), plus scoring
```



## Code for the following will be added soon:

- Contamination judge
- Jev evaluation

