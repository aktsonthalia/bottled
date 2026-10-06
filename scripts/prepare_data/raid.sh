#!/usr/bin/env bash
# Builds the RAID workload under <artifacts_dir>/data/raid from the liamdugan/raid dataset on Hugging Face.
#
# Run it with the main environment active, e.g. nohup bash scripts/prepare_data/raid.sh > raid_build.log 2>&1 &

set -euo pipefail

# 1. paths, from configs/machine.yaml through bottled.constants
DATASETS_DIR="$(python -c 'from bottled.constants import DATASETS_DIR; print(DATASETS_DIR)')"
RAID_DIR="$DATASETS_DIR/raid"
mkdir -p "$RAID_DIR/raw"
echo "building RAID in $RAID_DIR"

# 2. the dataset: only train.csv 
hf download liamdugan/raid train.csv --repo-type dataset --revision 865cac74188466cb0c3b7574a10204007b57a459 --local-dir "$RAID_DIR/raw"

# 3. build views/full.jsonl
python -c "from bottled.datasets.raid import build_full; build_full()"
N_LINES="$(wc -l < "$RAID_DIR/views/full.jsonl")"
if [ "$N_LINES" -ne 5615820 ]; then
  echo "views/full.jsonl has $N_LINES lines, expected 5615820"
  exit 1
fi

# 4. sampling
python -c "
from bottled.datasets.raid import VIEWS, _sample_dataset
_sample_dataset(VIEWS / 'full.jsonl', VIEWS, 1789669695)
_sample_dataset(VIEWS / 'full.jsonl', VIEWS, 1789669790, 1000)
"
