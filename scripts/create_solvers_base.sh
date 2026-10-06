#!/usr/bin/env bash
# Builds the conda env to be used by BOTTLED agents, at conda_envs_root/agents_env_name from configs/machine.yaml.
#
# Run it with the main environment active, e.g. nohup bash scripts/create_solvers_base.sh > agents_env_build.log 2>&1 &

set -euo pipefail

REPO="$(python -c 'from bottled.constants import REPO_ROOT; print(REPO_ROOT)')"
CONDA_ENVS_ROOT="$(python -c 'from bottled.constants import CONDA_ENVS_ROOT; print(CONDA_ENVS_ROOT)')"
AGENTS_ENV_NAME="$(python -c 'from bottled.constants import AGENTS_ENV_NAME; print(AGENTS_ENV_NAME)')"
ENV="$CONDA_ENVS_ROOT/$AGENTS_ENV_NAME"
REQUIREMENTS_AGENTS_ENV="$REPO/requirements_agents_env.txt"
LOCK="$REPO/$AGENTS_ENV_NAME.lock.txt"

PYTHON_VERSION=3.13.15
NODE_VERSION=22.23.2
SOCAT_VERSION=1.8.1.3
BUBBLEWRAP_VERSION=0.11.2
FUSE_OVERLAYFS_VERSION=1.17
OPENCODE_VERSION=1.17.18

TORCH_INDEX=https://download.pytorch.org/whl/cu130

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$CONDA_ENVS_ROOT/.pip-cache"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "creating env: python=$PYTHON_VERSION nodejs=$NODE_VERSION socat=$SOCAT_VERSION bubblewrap=$BUBBLEWRAP_VERSION fuse-overlayfs=$FUSE_OVERLAYFS_VERSION"
conda create -p "$ENV" -c conda-forge -y \
  "python=$PYTHON_VERSION" "nodejs=$NODE_VERSION" "socat=$SOCAT_VERSION" "bubblewrap=$BUBBLEWRAP_VERSION" "fuse-overlayfs=$FUSE_OVERLAYFS_VERSION"

log "installing packages for the agents' environment (this will take a while)"
"$ENV/bin/pip" install --extra-index-url "$TORCH_INDEX" -r "$REQUIREMENTS_AGENTS_ENV"


log "installing agent CLIs: opencode-ai@$OPENCODE_VERSION"
PATH="$ENV/bin:$PATH" "$ENV/bin/npm" install -g \
  "opencode-ai@$OPENCODE_VERSION"

log "verifying binaries agent_eval.py resolves from this env"
for b in python pip node npm socat bwrap fuse-overlayfs mitmdump opencode; do
  test -e "$ENV/bin/$b" || { echo "MISSING: $ENV/bin/$b"; exit 1; }
done

log "verifying the torch stack"
"$ENV/bin/python" -c "
import torch, vllm
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('vllm', vllm.__version__)
"

log "dependency consistency"
"$ENV/bin/pip" check

"$ENV/bin/pip" freeze > "$LOCK"
log "done. pins written to $LOCK"
log "agent env ready at $ENV"