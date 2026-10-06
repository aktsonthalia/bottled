#!/usr/bin/env bash
# Builds the MAVE workload under <artifacts_dir>/data/mave from the public MAVE labels and the Amazon product metadata.
#
# Run it with the main environment active, e.g. nohup bash scripts/prepare_data/mave.sh > mave_build.log 2>&1 &

set -euo pipefail

# 1. paths, from configs/machine.yaml through bottled.constants
DATASETS_DIR="$(python -c 'from bottled.constants import DATASETS_DIR; print(DATASETS_DIR)')"
CONDA_ENVS_ROOT="$(python -c 'from bottled.constants import CONDA_ENVS_ROOT; print(CONDA_ENVS_ROOT)')"
MAVE_DIR="$DATASETS_DIR/mave"
mkdir -p "$MAVE_DIR"
echo "building MAVE in $MAVE_DIR"

# 2. a Python 3.8 env for MAVE's cleaning script and the benchmark's split script
MAVE_ENV="$CONDA_ENVS_ROOT/mave-clean"
conda create -p "$MAVE_ENV" -c conda-forge -y python=3.8.20
# https://github.com/google-research-datasets/MAVE/blob/f84cefbd71578d4a84a70664ebdfc8f09b62910e/requirements.txt
"$MAVE_ENV/bin/pip" install absl-py==2.3.1 apache-beam==2.60.0 beautifulsoup4==4.15.0

# 3. the MAVE repo
MAVE_REPO="$MAVE_DIR/MAVE_repo"
git clone https://github.com/google-research-datasets/MAVE.git "$MAVE_REPO"
git -C "$MAVE_REPO" checkout f84cefbd71578d4a84a70664ebdfc8f09b62910e

# the MAVE benchmark's split script, from google-research
SPLIT_SCRIPT="$MAVE_DIR/split_json_lines_main.py"
wget -O "$SPLIT_SCRIPT" https://raw.githubusercontent.com/google-research/google-research/08a8d6736475776f42ffac23b2c13111a28e5795/mave/benchmark/data/split_json_lines_main.py

# 4. downloads (inputs for the cleaning script)
wget -O "$MAVE_REPO/labels/mave_positives_labels.jsonl" https://github.com/google-research-datasets/MAVE/raw/f84cefbd71578d4a84a70664ebdfc8f09b62910e/labels/mave_positives_labels.jsonl
wget -O "$MAVE_REPO/labels/mave_negatives_labels.jsonl" https://github.com/google-research-datasets/MAVE/raw/f84cefbd71578d4a84a70664ebdfc8f09b62910e/labels/mave_negatives_labels.jsonl
# the Amazon product metadata (https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), compressed twice
wget -c -O "$MAVE_REPO/All_Amazon_Meta.json.gz" https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles/All_Amazon_Meta.json.gz
gunzip -c "$MAVE_REPO/All_Amazon_Meta.json.gz" | gunzip -c > "$MAVE_REPO/All_Amazon_Meta.json"

# 5. clean the metadata and join it with the labels (~2-3 hours); writes reproduce/mave_{positives,negatives}.jsonl in the MAVE repo
(cd "$MAVE_REPO" && PATH="$MAVE_ENV/bin:$PATH" bash clean_amazon_product_metadata_main.sh)

# 6. split the cleaned examples into the benchmark's train/eval/test sets; writes raw/splits/PRODUCT/{train,eval,test}/00_All/mave_{positives,negatives}.jsonl
for part in positives negatives; do
  "$MAVE_ENV/bin/python" "$SPLIT_SCRIPT" \
    --input_json_lines_filename="$MAVE_REPO/reproduce/mave_$part.jsonl" \
    --output_json_lines_dir="$MAVE_DIR/raw/splits"
done

# 7. merge all six splits into the workload, views/full.jsonl
SPLITS="$MAVE_DIR/raw/splits/PRODUCT"
mkdir -p "$MAVE_DIR/views"
cat "$SPLITS"/train/00_All/mave_positives.jsonl "$SPLITS"/train/00_All/mave_negatives.jsonl \
    "$SPLITS"/eval/00_All/mave_positives.jsonl "$SPLITS"/eval/00_All/mave_negatives.jsonl \
    "$SPLITS"/test/00_All/mave_positives.jsonl "$SPLITS"/test/00_All/mave_negatives.jsonl \
    > "$MAVE_DIR/views/full.jsonl"
N_LINES="$(wc -l < "$MAVE_DIR/views/full.jsonl")"
if [ "$N_LINES" -ne 4767579 ]; then
  echo "views/full.jsonl has $N_LINES lines, expected 4767579"
  exit 1
fi

# 8. sample files
python -c "
from bottled.datasets.mave import VIEWS, _sample_dataset
_sample_dataset(VIEWS / 'full.jsonl', VIEWS, 1789474317)
_sample_dataset(VIEWS / 'full.jsonl', VIEWS, 1789474389, 1000)
"
echo "MAVE ready in $MAVE_DIR/views"
