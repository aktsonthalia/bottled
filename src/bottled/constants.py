import os
import yaml
from pathlib import Path

# Get the repository root
_here = Path(__file__).resolve().parent
REPO_ROOT = next(d for d in (_here, *_here.parents) if (d / "pyproject.toml").is_file())


_machine = yaml.safe_load((REPO_ROOT / "configs" / "machine.yaml").read_text())
os.environ.setdefault("BOTTLED_ARTIFACTS_ROOT", _machine["artifacts_dir"])
os.environ.setdefault("BOTTLED_CONDA_ENVS_ROOT", _machine["conda_envs_root"])

CONDA_ENVS_ROOT = Path(os.environ["BOTTLED_CONDA_ENVS_ROOT"])
ARTIFACTS_DIR = Path(os.environ["BOTTLED_ARTIFACTS_ROOT"])
AGENTS_ENV_NAME = _machine["agents_env_name"]
USE_FUSE_OVERLAYFS = _machine["use_fuse_overlayfs"]

DATASETS_DIR = ARTIFACTS_DIR / "data"

METRIC_PATH = {
    "mave": ["f1"],
    "esci": ["macro_f1"],
    "raid": ["score_agg", "all", "auroc"],
}

ZEROSHOT_SAMPLE_LINES = 1000  # zeroshot eval labels this many lines of the workload