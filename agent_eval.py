# region imports and constants [reviewed 12 Sep 2026]

import hydra
import jinja2
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, fields
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import TextIO

from bottled.constants import *
from bottled.datasets import input_providers, register, scorers
from bottled.harness import Bind, HARNESSES
# from bottled.unit_tests import * # skipped in this version
from bottled.utils import allocate_free_port, calculate_usd_cost, load_usd_rates, stop_procs

LOG_PREFIX = "[agent_eval]"
USD_COSTS_CSV = REPO_ROOT / "charts" / "openrouter_prices.csv"
LOCALHOST_URL = "http://127.0.0.1"
# opencode asks for `min(model.limit.output, OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX)` tokens; we set this to model.limit.output so that our desired output limit is respected
# see https://github.com/anomalyco/opencode/blob/b1fc8113948b518835c2a39ece49553cffe9b30c/packages/opencode/src/provider/transform.ts#L1345-L1347
# also https://github.com/anomalyco/opencode/blob/b1fc8113948b518835c2a39ece49553cffe9b30c/packages/opencode/src/effect/runtime-flags.ts#L52
OPENCODE_OUTPUT_TOKEN_MAX_ENV_VAR = "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"

# endregion

# region DATA CLASSES [reviewed 12 Sep 2026]

@dataclass(frozen=True)
class Paths:
    # region host-side filesystem locations [reviewed 12 Sep 2026]
    home = property(lambda self: Path.home()) # original home directory

    # module-level constants
    project_root     = property(lambda self: REPO_ROOT)
    agents_env       = property(lambda self: CONDA_ENVS_ROOT / AGENTS_ENV_NAME)
    system_ca_bundle = property(lambda self: _find_system_ca_bundle())

    # derived from project_root
    mcp_server             = property(lambda self: self.project_root / "src/bottled/mcp_server.py")
    run_proxy              = property(lambda self: self.project_root / "src/bottled/run_proxy.py")

    # binaries from agents_env
    agents_env_bin = property(lambda self: self.agents_env / "bin")
    fuse_overlayfs = property(lambda self: self.agents_env_bin / "fuse-overlayfs")
    mitmdump       = property(lambda self: self.agents_env_bin / "mitmdump")
    bwrap          = property(lambda self: self.agents_env_bin / "bwrap")

    # derived from output_dir. Only workspace_dir is bound as the sandbox root;
    # harness bookkeeping in output_dir (.hydra, logs, call_history.jsonl, overlay/) stays invisible to the agent.
    workspace_dir     = property(lambda self: self.output_dir / "workspace")
    log_path          = property(lambda self: self.output_dir / "agent_eval.log")
    call_history_path = property(lambda self: self.output_dir / "call_history.jsonl")
    proxy_log_file    = property(lambda self: self.output_dir / "proxy_logs.txt")
    proxy_logs_jsonl  = property(lambda self: self.output_dir / "proxy_logs.jsonl")
    mcp_log_file      = property(lambda self: self.output_dir / "mcp_logs.txt")

    # derived from local_sockets_dir
    proxy_sock_host = property(lambda self: self.local_sockets_dir / "proxy.sock")
    mcp_sock_host   = property(lambda self: self.local_sockets_dir / "mcp.sock")

    # run-specific paths derived from the run's cfg
    mtok_weights_csv: Path # mtok_weights.csv file
    output_dir: Path # this run's main output directory 
    agent_modifiable_conda_env: Path # overlayed conda environment for use by the agents
    ledger_path: Path # api_ledger.json file
    local_sockets_dir: Path # sockets directory; AF_UNIX paths are capped at 108 bytes (sockaddr_un.sun_path). We use a path that's guaranteed short.
    source_dataset_root: Path | None # the data/ directory
    dataset_binds: list[Bind] # binds to all the files corresponding to the dataset
    # endregion

    # region paths as seen from inside the bwrap sandbox [reviewed 12 Sep 2026]
    sandbox_root         = property(lambda self: "/task")  # corresponds to host:workspace_dir                                
    sandbox_sockets      = property(lambda self: "/sockets")
    sandbox_proxy_sock   = property(lambda self: f"{self.sandbox_sockets}/proxy.sock")      
    sandbox_mcp_sock     = property(lambda self: f"{self.sandbox_sockets}/mcp.sock")      
    sandbox_conda        = property(lambda self: "/conda")
    sandbox_local        = property(lambda self: f"{self.sandbox_root}/.local")
    sandbox_mitmproxy    = property(lambda self: "/mitmproxy")
    sandbox_harness      = property(lambda self: "/harness")
    sandbox_mcp_server   = property(lambda self: f"{self.sandbox_harness}/mcp_server.py")
    sandbox_project      = property(lambda self: f"{self.sandbox_harness}/project")
    sandbox_ledger       = property(lambda self: f"{self.sandbox_harness}/api_ledger.json")
    sandbox_mitmproxy_ca = property(lambda self: f"{self.sandbox_mitmproxy}/mitmproxy-ca-cert.pem")
    sandbox_src          = property(lambda self: f"{self.sandbox_project}/src")
    # endregion

@dataclass(frozen=True)
class RunInfo:
    # model
    model_name: str
    model_limit: dict | None
    is_local: bool
    allowed_models: list[str]
    harness_thinking: str | None

    # budgets
    api_budget_mtok: float
    time_budget_min: int
    run_start_time: float
    hide_budget_tools: bool
    enforce_budgets: int

    # ports
    proxy_port: int
    agent_proxy_port: int
    mcp_proxy_port: int
    agent_mcp_port: int

    # access control
    run_secret: str
    blocked_url_patterns: list[list[str]]

    # hardware
    selected_gpu: str

# endregion

# region SYSTEM CA BUNDLE [reviewed 12 Sep 2026]

_SYSTEM_CA_BUNDLE_CANDIDATES = [
    Path("/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem"),  # RHEL / CentOS
    Path("/etc/ssl/certs/ca-certificates.crt"),  # Debian / Ubuntu
]
def _find_system_ca_bundle() -> Path:
    for candidate in _SYSTEM_CA_BUNDLE_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Could not find a system CA bundle in any of "
        f"{[str(c) for c in _SYSTEM_CA_BUNDLE_CANDIDATES]}"
    )

# endregion

# region conda env: overlaying a modifiable conda environment [reviewed 12 Sep 2026]

def _overlay_conda_env(output_dir: Path, agents_env: Path, fuse_overlayfs: Path) -> Path:
    assert USE_FUSE_OVERLAYFS, "only systems with fuse_overlayfs are supported for now."
    overlay_upper = output_dir / "overlay" / "upper"
    overlay_work = output_dir / "overlay" / "work"
    overlay_merged = output_dir / "overlay" / "agent_modifiable"
    for d in [overlay_upper, overlay_work, overlay_merged]:
        d.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(fuse_overlayfs),
        "-o", f"lowerdir={agents_env},upperdir={overlay_upper},workdir={overlay_work}",
        str(overlay_merged),
    ], check=True)
    return overlay_merged

# endregion

# region MCP [reviewed 13 Sep 2026]

def _run_mcp_proc(
    mcp_sock_host: Path, 
    mcp_log_file: TextIO,
    mcp_cmd: list[str], 
    bwrap_env: dict[str, str]
) -> subprocess.Popen: 
    """Start MCP process and wait for socket to appear. [reviewed 12 Sep 2026]"""
    mcp_proc = subprocess.Popen(
        mcp_cmd,
        env=bwrap_env,
        stdout=mcp_log_file,
        stderr=subprocess.STDOUT,
    )
    for _ in range(120):
        if mcp_sock_host.exists():
            break
        if mcp_proc.poll() is not None:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError(f"MCP socket did not appear: {mcp_sock_host}")
    if not mcp_sock_host.exists():
        raise RuntimeError(
            f"MCP process exited before socket appeared (rc={mcp_proc.returncode}): {mcp_sock_host}"
        )
    return mcp_proc


def _build_mcp_cmd(
    bwrap_common_args: list[str],
    paths: Paths,
    run_info: RunInfo,
    mcp_transport: str,
) -> list[str]: # reviewed 13 Sep 2026
    mcp_cmd = bwrap_common_args + [
        "--ro-bind", str(paths.ledger_path), paths.sandbox_ledger,
        "--ro-bind", str(paths.agents_env), str(paths.agents_env),
        "--ro-bind", str(paths.mcp_server), paths.sandbox_mcp_server,
        "--setenv", "http_proxy", f"http://127.0.0.1:{run_info.mcp_proxy_port}",
        "--setenv", "https_proxy", f"http://127.0.0.1:{run_info.mcp_proxy_port}",
        "--chdir", paths.sandbox_root,
        "/usr/bin/bash",
        "-lc",
        (
            f"socat TCP-LISTEN:{run_info.mcp_proxy_port},bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:{paths.sandbox_proxy_sock} & "
            f"exec \"$@\""
        ),
        "_",
        # exec: run everything that follows as a command, but skip the first arg (_ above)        
        f"{paths.agents_env}/bin/python",
        paths.sandbox_mcp_server,
        "--time-budget-min",
        str(run_info.time_budget_min),
        "--run-start-time",
        str(run_info.run_start_time),
        "--api-ledger-path",
        paths.sandbox_ledger,
        "--workspace-dir",
        paths.sandbox_root,
        "--transport",
        mcp_transport,
        "--uds",
        paths.sandbox_mcp_sock,
    ]
    return mcp_cmd

# endregion

# region path building [reviewed 13 Sep 2026]

def _bwrap_bind_args(binds: list[Bind]) -> list[str]: # reviewed 12 Sep 2026
    args = []
    for bind in binds:
        args += ["--ro-bind" if bind.readonly else "--bind", str(bind.host), bind.sandbox]
    return args


def _create_ledger(output_dir: Path, api_budget_mtok: float) -> Path: # reviewed 12 Sep 2026
    ledger_path = output_dir / "api_ledger.json"
    ledger_path.write_text(
        json.dumps({"spent_mtok": 0.0, "budget_mtok": api_budget_mtok, "calls": []}, indent=2),
        encoding="utf-8",
    )
    return ledger_path


def _build_paths(cfg: DictConfig) -> Paths: # reviewed 13 Sep 2026

    # region build env, build prompt and housekeeping paths 
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    mtok_weights_csv = REPO_ROOT / cfg.mtok_weights_csv
    ledger_path = _create_ledger(output_dir, cfg.api_budget_mtok)
    workspace_dir = output_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    # prompts may `{% include %}` other files under prompts/; the rendered text is both /task/TASK.md and the first opencode message.
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(REPO_ROOT / "prompts"), keep_trailing_newline=True)
    (workspace_dir / "TASK.md").write_text(env.get_template(f"{cfg.prompt}.md").render(), encoding="utf-8")
    source_dataset_root = DATASETS_DIR / cfg.dataset.root if cfg.dataset.name in register else None
    dataset_binds = input_providers[cfg.dataset.name](cfg.dataset, output_dir)
    agents_env = CONDA_ENVS_ROOT / AGENTS_ENV_NAME  # local only: the Paths.agents_env property does not exist yet
    agent_modifiable_conda_env = _overlay_conda_env(output_dir, agents_env, agents_env / "bin" / "fuse-overlayfs")
    # endregion

    # region grab local sockets dir [reviewed 13 Sep 2026]
    sockets_root = REPO_ROOT / "sockets"
    sockets_root.mkdir(exist_ok=True)
    while True: # grab a unique short path
        local_sockets_dir = sockets_root / secrets.token_hex(8)
        if not local_sockets_dir.exists():
            break
    local_sockets_dir.mkdir()
    # endregion

    (workspace_dir / ".tmp").mkdir(parents=True, exist_ok=True)

    vars_ = locals()
    return Paths(**{f.name: vars_[f.name] for f in fields(Paths)})

# endregion

def _build_bwrap_common_args(paths: Paths, run_info: RunInfo) -> list[str]: # reviewed 14 Sep 2026
    args = [
        str(paths.bwrap),
        # region unshare and binds [reviewed 14 Sep 2026]
        "--unshare-user", 
        "--unshare-ipc", 
        "--unshare-net",
        "--unshare-pid",
        "--die-with-parent", # processes inside the sandbox die when the sandbox process dies
        "--ro-bind", "/bin", "/bin", 
        "--ro-bind", "/sbin", "/sbin",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/usr", "/usr",
        "--dev-bind", "/dev", "/dev",   
        "--ro-bind", "/sys", "/sys",
        "--proc", "/proc",
        "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
        "--ro-bind", "/etc/hosts", "/etc/hosts",
        "--ro-bind", "/etc/alternatives", "/etc/alternatives",
        "--bind", f"{paths.workspace_dir}/.tmp", "/tmp", # to preserve agents' /tmp writes
        "--tmpfs", "/dev/shm", # fresh tmpfs for agent's use, https://www.kernel.org/doc/html/v6.3/filesystems/tmpfs.html
        "--bind", str(paths.workspace_dir), paths.sandbox_root,
        "--bind", str(paths.local_sockets_dir), paths.sandbox_sockets,
        "--ro-bind", str(paths.mtok_weights_csv), f"{paths.sandbox_root}/mtok_weights.csv", # for agent to handle budget
        "--ro-bind", str(paths.home / ".mitmproxy"), paths.sandbox_mitmproxy, # cert store
        # endregion
        # region environment variables
        # region basic environment
        "--setenv", "HOME", paths.sandbox_root,
        "--setenv", "PATH", f"{paths.sandbox_conda}/bin:/usr/local/cuda/bin:/usr/bin:/usr/sbin:/sbin:/bin",
        "--setenv", "CUDA_VISIBLE_DEVICES", run_info.selected_gpu,
        "--setenv", "USER", "task", # The sandbox uid has no /etc/passwd entry
        # endregion
        # region python [reviewed 13 Sep 2026]
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1", # disable .pyc files
        # endregion
        # region proxy [reviewed 13 Sep 2026].
        "--setenv", "http_proxy", f"http://127.0.0.1:{run_info.proxy_port}",
        "--setenv", "https_proxy", f"http://127.0.0.1:{run_info.proxy_port}",
        "--setenv", "CURL_CA_BUNDLE", paths.sandbox_mitmproxy_ca,
        "--setenv", "SSL_CERT_FILE", paths.sandbox_mitmproxy_ca,
        "--setenv", "NODE_EXTRA_CA_CERTS", paths.sandbox_mitmproxy_ca,
        "--setenv", "REQUESTS_CA_BUNDLE", paths.sandbox_mitmproxy_ca,
        # endregion
        # region openrouter [reviewed 13 Sep 2026]. 
        "--setenv", "OPENROUTER_API_KEY", run_info.run_secret, # for opencode to make API calls
        "--setenv", "OPENAI_API_KEY", run_info.run_secret, # for agent to make API calls
        "--setenv", "OPENAI_BASE_URL", "https://openrouter.ai/api/v1",
        "--setenv", "OPENAI_MODEL", run_info.allowed_models[0],
        # endregion
    ]
    if run_info.model_limit is not None:
        args += ["--setenv", OPENCODE_OUTPUT_TOKEN_MAX_ENV_VAR, str(run_info.model_limit["output"])]
    return args


def _build_run_info(cfg: DictConfig) -> RunInfo: # [reviewed 13 Sep 2026]
    run_start_time = time.time() 
    run_secret = secrets.token_urlsafe(48)
    proxy_port = allocate_free_port() # on main machine, listening to agent and mcp
    agent_proxy_port = allocate_free_port() # inside agent sandbox, for agent to access proxy
    agent_mcp_port = allocate_free_port() # inside agent sandbox, for agent to make MCP calls
    mcp_proxy_port = allocate_free_port() # inside mcp sandbox, for mcp to access proxy
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    selected_gpu = cuda_visible if cuda_visible.isdigit() else "0"
    is_local = cfg.model.get("is_local")
    model_name = cfg.model.name
    model_limit = cfg.model.get("limit")
    assert model_limit is not None, "model.limit is required"
    model_limit = OmegaConf.to_container(model_limit, resolve=True)
    time_budget_min = cfg.time_budget_min
    api_budget_mtok = cfg.api_budget_mtok
    enforce_budgets = cfg.enforce_budgets
    hide_budget_tools = cfg.get("hide_budget_tools", False)
    harness_thinking = cfg.get("harness_thinking")
    allowed_models = list(cfg.allowed_models)
    blocked_url_patterns = list(getattr(register.get(cfg.dataset.name), "BLOCKED_URL_PATTERNS", []))
    vars_ = locals()
    return RunInfo(**{f.name: vars_[f.name] for f in fields(RunInfo)})


def _build_agent_base(
    bwrap_common_args: list[str],
    paths: Paths,
    run_info: RunInfo,
) -> list[str]: # reviewed 14 Sep 2026
    agent_base = bwrap_common_args + [
        "--bind", str(paths.agent_modifiable_conda_env), paths.sandbox_conda,
        "--symlink", paths.sandbox_conda, str(paths.agents_env), # if pip's shebang tries to use the base env's absolute path, this symlink redirects it to the sandboxed env
    ] + _bwrap_bind_args(paths.dataset_binds) + [
        "--setenv", "http_proxy", f"{LOCALHOST_URL}:{run_info.agent_proxy_port}",
        "--setenv", "https_proxy", f"{LOCALHOST_URL}:{run_info.agent_proxy_port}",
        "--setenv", "NO_PROXY", f"127.0.0.1:{run_info.agent_mcp_port},localhost:{run_info.agent_mcp_port}",
        "--setenv", "no_proxy", f"127.0.0.1:{run_info.agent_mcp_port},localhost:{run_info.agent_mcp_port}",
        "--chdir", paths.sandbox_root,
        "/usr/bin/bash",
        "-lc",
        (
            "trap 'kill $SOCAT_PROXY $SOCAT_MCP 2>/dev/null; wait' EXIT; " # clean up socat processes on exit
            f"socat TCP-LISTEN:{run_info.agent_proxy_port},bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:{paths.sandbox_proxy_sock} & SOCAT_PROXY=$!; " # forward requests to this port to the proxy socket
            f"socat TCP-LISTEN:{run_info.agent_mcp_port},bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:{paths.sandbox_mcp_sock} & SOCAT_MCP=$!; " # forward requests to this port to the mcp socket
            f"for _ in $(seq 1 50); do " # wait until both ports are accepting connections
            f"(echo > /dev/tcp/127.0.0.1/{run_info.agent_proxy_port}) >/dev/null 2>&1 "
            f"&& (echo > /dev/tcp/127.0.0.1/{run_info.agent_mcp_port}) >/dev/null 2>&1 && break; "
            f"sleep 0.1; "
            f"done; \"$@\""
        ),
        "_",
    ]
    return agent_base

# region main function [reviewed 13 Sep 2026]

@hydra.main(version_base=None, config_path="configs", config_name="agent_eval/default")
def main(cfg: DictConfig) -> None:

    OmegaConf.resolve(cfg)
    # unit tests can leave unintended files (e.g. a downloaded model) in the agent's workspace; used only during debugging
    if cfg.run_unit_tests or cfg.unit_tests_only:
        raise NotImplementedError("run with run_unit_tests=false unit_tests_only=false")
    paths = _build_paths(cfg)
    run_info = _build_run_info(cfg)
    harness = HARNESSES[cfg.harness.name]
    harness.write_settings(paths, run_info)
    
    bwrap_common_args = _build_bwrap_common_args(paths, run_info)
    bwrap_env = {"PATH": "/usr/bin:/bin"} # avoids leaking host secrets.

    # region start proxy [reviewed 13 Sep 2026]
    proxy_log_fp = paths.proxy_log_file.open("w", encoding="utf-8")
    mitmproxy_ca = paths.home / ".mitmproxy" / "mitmproxy-ca-cert.pem"
    proxy_env = {
        **os.environ,
        "CALL_HISTORY_PATH": str(paths.call_history_path),
        "PROXY_LOGS_JSONL_PATH": str(paths.proxy_logs_jsonl),
        "API_ALLOWED_MODELS": json.dumps(run_info.allowed_models),
        "RUN_SECRET": run_info.run_secret,
        "API_LEDGER_PATH": str(paths.ledger_path),
        "API_MTOK_WEIGHTS_CSV": str(paths.mtok_weights_csv),
        "ENFORCE_BUDGETS": "1" if run_info.enforce_budgets else "0",
        # host-side; invisible to the agent sandbox
        "PYTHONPATH": os.pathsep.join([str(paths.project_root / "src"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
        "BLOCKED_URL_PATTERNS": json.dumps(run_info.blocked_url_patterns),
    }
    proxy_cmd = [
        str(paths.mitmdump),
        "--listen-host", "127.0.0.1",
        "--listen-port", str(run_info.proxy_port),
        "-s", str(paths.run_proxy),
        "--set", f"confdir={paths.home / '.mitmproxy'}",
        "--set", "connection_strategy=lazy",
        "--set", "ssl_verify_upstream_cert=true",
        "--set", f"ssl_verify_upstream_trusted_ca={paths.system_ca_bundle}",
    ]
    proxy_proc = subprocess.Popen(
        proxy_cmd,
        env=proxy_env,
        stdout=proxy_log_fp,
        stderr=subprocess.STDOUT,
    )
    # wait until the CA cert appears
    for _ in range(50):
        if mitmproxy_ca.exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"mitmproxy CA cert did not appear: {mitmproxy_ca}")

    # start the socat proxy
    socat_proc = subprocess.Popen(
        [
            "socat",
            f"UNIX-LISTEN:{paths.proxy_sock_host},fork,reuseaddr",
            f"TCP:127.0.0.1:{run_info.proxy_port}",
        ],
        stdout=proxy_log_fp,
        stderr=subprocess.STDOUT,
    )
    # endregion

    # region start and watch agent [reviewed 13 Sep 2026]

    mcp_proc = None
    mcp_log_fp = None
    agent_process = None
    agent_end_time = None

    try:
        # region Unit tests [reviewed 12 Sep 2026]
        if cfg.run_unit_tests:
            pass
        else:
            print("[agent_eval] Skipping unit tests (run_unit_tests=false)", flush=True)
        # endregion

        mcp_cmd = _build_mcp_cmd(bwrap_common_args, paths, run_info, harness.mcp_transport)
        mcp_log_fp = paths.mcp_log_file.open("w", encoding="utf-8")
        mcp_proc = _run_mcp_proc(paths.mcp_sock_host, mcp_log_fp, mcp_cmd, bwrap_env)
        agent_base = _build_agent_base(bwrap_common_args, paths, run_info)
        if cfg.run_unit_tests:
            pass
        if cfg.unit_tests_only:
            print("[agent_eval] unit_tests_only=true; not starting the agent", flush=True)
            return
        time_limit_cmd = ["timeout", str(int(run_info.time_budget_min * 60))] if run_info.enforce_budgets else []
        agent_cmd = time_limit_cmd + agent_base + [
            "/usr/bin/bash", "-c", harness.resume_loop, "_", # run the resume loop with args that follow _
            f"{paths.sandbox_conda}/bin/{harness.binary}",
            harness.model_arg(run_info.model_name),
        ]

        def stop_agent_if_api_budget_exceeded(proc: subprocess.Popen) -> bool: # reviewed 13 Sep 2026
            if paths.ledger_path.with_name(".budget_exhausted").exists():
                print(
                    f"[agent_eval] API budget exceeded ({cfg.api_budget_mtok} mtok); stopping agent",
                    file=sys.stderr,
                    flush=True,
                )
                proc.terminate()
                return True
            return False


        def watch_api_budget(proc: subprocess.Popen) -> None: # reviewed 13 Sep 2026
            # keep checking the budget until the agent is stopped or the budget is exceeded
            while True:
                if proc.poll() is not None:
                    return
                if stop_agent_if_api_budget_exceeded(proc):
                    return
                time.sleep(1.0)

        agent_process = subprocess.Popen(
            agent_cmd,
            env=bwrap_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            errors="replace", # replace bad bytes; prevents them from killing the run
            bufsize=1,
        )
        # region start the budget watcher thread [reviewed 13 Sep 2026]
        if run_info.enforce_budgets:
            budget_thread = threading.Thread(
                target=watch_api_budget, args=(agent_process,), daemon=True
            )
            budget_thread.start()
        # endregion

        # region log and wait on agent process [reviewed 13 Sep 2026]
        with paths.log_path.open("w", encoding="utf-8") as log_file:
            for line in agent_process.stdout:
                try:
                    event = json.loads(line)
                    print(harness.format_event(event), flush=True)
                except json.JSONDecodeError:
                    print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()

            agent_process.wait()
        agent_end_time = time.time()
        if agent_process.returncode != 0:
            print(f"[WARN] Agent process exited with code {agent_process.returncode}", file=sys.stderr)
        # endregion

    # endregion

    finally:
        
        # region Post-run evaluation [reviewed 13 Sep 2026]

        # In case of an exception on the agent path 
        if agent_end_time is None:
            agent_end_time = time.time()
        stop_procs((agent_process,))
        if agent_process is not None:
            out_path = paths.workspace_dir / "out.txt"
            ledger = json.loads(paths.ledger_path.read_text(encoding="utf-8"))
            cost = {"mtok": ledger["spent_mtok"]}
            # raising here would skip the teardown below; the error is recorded in the result.
            try:
                usd_rates = load_usd_rates(USD_COSTS_CSV)
                cost["usd"] = sum(calculate_usd_cost(call["usage"], call["model"], usd_rates) for call in ledger["calls"])
            except Exception as e:
                cost["usd_error"] = f"{type(e).__name__}: {e}"
            post_run_result = {
                "agent_returncode": agent_process.returncode,
                "walltime_min": (agent_end_time - run_info.run_start_time) / 60,
                "cost": cost,
                "out_txt_exists": out_path.exists(),
                "launch_command": os.environ.get("LAUNCH_COMMAND"),
                "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=True).stdout.strip(),
            }
            scorer = scorers.get(cfg.dataset.name)
            assert scorer is not None, f"Scorer not found for dataset {cfg.dataset.name}"
            if not cfg.skip_scoring and out_path.exists():
                try:
                    post_run_result.update(scorer(cfg.dataset, out_path))
                except Exception as e:
                    post_run_result["eval_error"] = f"{type(e).__name__}: {e}"
            post_run_path = paths.output_dir / "post_run_result.json"
            post_run_path.write_text(json.dumps(post_run_result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Post-run results saved to {post_run_path.as_posix()}", flush=True)

        # endregion

        # region Teardown [reviewed 12 Sep 2026]

        stop_procs((mcp_proc,))
        stop_procs((socat_proc, proxy_proc))
        shutil.rmtree(paths.local_sockets_dir, ignore_errors=True)
        subprocess.run(["fusermount", "-u", str(paths.agent_modifiable_conda_env)], check=True)
        shutil.rmtree(paths.agent_modifiable_conda_env, ignore_errors=True)
        if mcp_log_fp is not None:
            mcp_log_fp.close()
        proxy_log_fp.close()

        # endregion

    print(f"Log saved to {paths.log_path.as_posix()}")


if __name__ == "__main__":
    main()

# endregion