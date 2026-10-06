#!/usr/bin/env bash
# Builds the main environment (runs agent_eval.py, zeroshot_eval.py and distill.py) at <conda_envs_root>/bottled, with conda_envs_root from configs/machine.yaml.
#
# Run it from anywhere, e.g. nohup bash scripts/create_main_env.sh > main_env_build.log 2>&1 &

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENVS_ROOT="$(awk '/^conda_envs_root:/ {print $2}' "$REPO/configs/machine.yaml")"
if [ -z "$CONDA_ENVS_ROOT" ]; then
  echo "conda_envs_root is not set in $REPO/configs/machine.yaml"
  exit 1
fi
ENV="$CONDA_ENVS_ROOT/bottled"

PYTHON_VERSION=3.11.15
SOCAT_VERSION=1.8.1.1
RAID_BENCH_VERSION=0.2.0

export PYTHONNOUSERSITE=1

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "creating env at $ENV: python=$PYTHON_VERSION socat=$SOCAT_VERSION"
conda create -p "$ENV" -c conda-forge -y "python=$PYTHON_VERSION" "socat=$SOCAT_VERSION"

log "installing packages for the main environment"
"$ENV/bin/pip" install -r "$REPO/requirements_bottled.txt"

log "dependency consistency"
"$ENV/bin/pip" check

# raid-bench pins numpy<2, which conflicts with vllm
log "installing raid-bench==$RAID_BENCH_VERSION without its dependencies"
"$ENV/bin/pip" install --no-deps "raid-bench==$RAID_BENCH_VERSION"

log "installing bottled"
"$ENV/bin/pip" install -e "$REPO"

log "verifying imports"
"$ENV/bin/python" -c "import bottled.datasets, raid, vllm, trl; print('imports ok')"

log "main env ready at $ENV"
