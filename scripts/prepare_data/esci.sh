#!/usr/bin/env bash
# Builds the ESCI workload under <artifacts_dir>/data/esci from the tasksource/esci dataset on Hugging Face.
#
# Run it with the main environment active, e.g. nohup bash scripts/prepare_data/esci.sh > esci_build.log 2>&1 &

set -euo pipefail

# 1. paths, from configs/machine.yaml through bottled.constants
DATASETS_DIR="$(python -c 'from bottled.constants import DATASETS_DIR; print(DATASETS_DIR)')"
ESCI_DIR="$DATASETS_DIR/esci"
mkdir -p "$ESCI_DIR/raw"
echo "building ESCI in $ESCI_DIR"

# 2. the dataset
hf download tasksource/esci --repo-type dataset --revision 8113b17a5d4099e20243282c926f1bc1a08a4d13 --local-dir "$ESCI_DIR/raw"

# 3. merge the shards into the workload
python -c "from bottled.datasets.esci import build_full; build_full()"
N_LINES="$(wc -l < "$ESCI_DIR/views/full.jsonl")"
if [ "$N_LINES" -ne 2621288 ]; then
  echo "views/full.jsonl has $N_LINES lines, expected 2621288"
  exit 1
fi

# 4. sampling
python -c "
from bottled.datasets.esci import VIEWS, _sample_dataset
_sample_dataset(VIEWS / 'full.jsonl', VIEWS, 1789578089)
_sample_dataset(VIEWS / 'full.jsonl', VIEWS, 1789578129, 1000)
"

# 5. one fixed random label per example, which ESCI scoring uses for a missing or unparseable prediction; writes views/random_labels.json
python -c "from bottled.datasets.esci import build_random_labels; build_random_labels()"
echo "ESCI ready in $ESCI_DIR/views"
