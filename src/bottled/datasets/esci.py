"""ESCI (Amazon Shopping Queries): judge how well a product answers a search query."""

import glob
import orjson
import random

from functools import cache, partial
from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import f1_score



from bottled.constants import DATASETS_DIR
from bottled.datasets.blocklists.esci import BLOCKED_URL_PATTERNS


from . import common

RAW = DATASETS_DIR / "esci" / "raw" / "data"
VIEWS = DATASETS_DIR / "esci" / "views"

PRODUCT_FIELDS = ("product_title", "product_brand", "product_color", "product_description", "product_bullet_point")
ID_FIELDS = ("example_id", "query_id", "product_id")

PRED_TO_LABEL = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}
LETTERS = tuple(PRED_TO_LABEL)

RANDOM_LABELS = VIEWS / "random_labels.json"
RANDOM_LABELS_SEED = 0


def build_random_labels(seed: int = RANDOM_LABELS_SEED) -> Path:
    """Write views/random_labels.json: one uniformly drawn letter per example_id in full.jsonl (for fallbacks).

    python -c "from bottled.datasets.esci import build_random_labels; build_random_labels()"
    """
    src = VIEWS / "full.jsonl"
    rng = random.Random(seed)
    labels = {}
    with src.open("rb") as f, tqdm(total=src.stat().st_size, unit="B", unit_scale=True, desc=f"read {src.name}") as pbar:
        while line := f.readline():
            labels[orjson.loads(line)["example_id"]] = rng.choice(LETTERS)
            pbar.update(len(line))

    RANDOM_LABELS.write_bytes(orjson.dumps(labels))
    print(f"wrote {len(labels):,} random labels to {RANDOM_LABELS}")
    return RANDOM_LABELS


class ESCI(common.JsonlDataset):

    BLOCKED_URL_PATTERNS = BLOCKED_URL_PATTERNS

def build_full() -> Path:
    """Write views/full.jsonl: one line per (query, product) pair, in shard order, holding the model's fields and the gold fields in one object for the sampler to split.

    python -c "from bottled.datasets.esci import build_full; build_full()"
    """
    import pandas as pd

    shards = sorted(glob.glob(str(RAW / "*.parquet")))
    assert shards, f"no *.parquet under {RAW}"
    out_path = VIEWS / "full.jsonl"
    VIEWS.mkdir(parents=True, exist_ok=True)

    seen: set[int] = set()  # example_ids written so far
    n_read = 0
    with out_path.open("wb") as out:
        for shard in tqdm(shards, desc="reading ESCI", unit="shard"):
            split = Path(shard).name.split("-", 1)[0]
            df = pd.read_parquet(shard, columns=[*ID_FIELDS, "query", *PRODUCT_FIELDS, "product_locale", "esci_label", "large_version"])
            n_read += len(df)
            for row in df.itertuples(index=False):
                if row.example_id in seen:
                    continue
                seen.add(row.example_id)
                product = {f: getattr(row, f) for f in PRODUCT_FIELDS if getattr(row, f) is not None}
                record = {
                    **{f: str(getattr(row, f)) for f in ID_FIELDS},
                    "query": row.query,
                    **product,
                    "label": row.esci_label,
                    "locale": row.product_locale,
                    "large_version": int(row.large_version),
                    "split": split,
                }
                out.write(orjson.dumps(record) + b"\n")

    print(f"read {n_read:,} rows, dropped {n_read - len(seen):,} duplicates")
    print(f"wrote {len(seen):,} lines to {out_path}")
    return out_path

def _sample_dataset(
    src: Path, 
    out_dir: Path, 
    seed: int, 
    num: int | None = None
) -> tuple[Path, Path, Path]: # reviewed 15 Sep 2026
    """Shuffle `src` with `seed`, keep the first `num` lines (all if None), and write each kept line as a question line (query + the product fields the row has), a gold line (the ids, label, locale, large_version and split), and an offsets line (line index and byte offset in `src`)."""
    offsets = []  # (line index, byte offset) per source line
    with src.open("rb") as f, tqdm(total=src.stat().st_size, unit="B", unit_scale=True, desc=f"index {src.name}") as pbar:
        while line := f.readline():
            offsets.append((len(offsets), f.tell() - len(line)))
            pbar.update(len(line))
    if num is not None and num > len(offsets):
        raise ValueError(f"asked for {num} lines but {src} has {len(offsets)}")
    n_lines = len(offsets)
    random.Random(seed).shuffle(offsets)
    offsets = offsets[:num]
    wanted = {line_index for line_index, _ in offsets}
    lines = {} 
    with src.open("rb") as f:
        for line_index, line in enumerate(tqdm(f, total=n_lines, unit=" lines", unit_scale=True, desc=f"read {src.name}")):
            if line_index in wanted:
                lines[line_index] = line

    tag = f"n{num}_seed{seed}" if num is not None else f"full_seed{seed}"
    q_path = out_dir / f"{tag}_questions.jsonl"
    gt_path = out_dir / f"{tag}_gt.jsonl"
    offsets_path = out_dir / f"{tag}_offsets.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with q_path.open("wb") as q, gt_path.open("wb") as gt, offsets_path.open("wb") as off:
        for line_index, offset in tqdm(offsets, desc=f"write {tag}"):
            item = orjson.loads(lines[line_index])
            question = {"query": item.pop("query"), **{f: item.pop(f) for f in PRODUCT_FIELDS if f in item}}
            q.write(orjson.dumps(question) + b"\n")
            gt.write(orjson.dumps(item) + b"\n")
            off.write(orjson.dumps({"line": line_index, "offset": offset}) + b"\n")
    return q_path, gt_path, offsets_path


sample = partial(common.sample, _sample_dataset, VIEWS)

sandbox_input = common.jsonl_sandbox_input

zeroshot_model_input = common.zeroshot_model_input

GOLD_FIELDS = ("example_id", "label", "locale", "large_version", "split")

zeroshot_gold_row = partial(common.zeroshot_gold_row, gold_fields=GOLD_FIELDS)


@cache
def random_labels() -> dict[str, str]:
    """views/random_labels.json."""
    return orjson.loads(RANDOM_LABELS.read_bytes())


def handle_missing(example_id: str) -> str:
    """handles an example with no usable prediction."""
    return random_labels()[example_id]


def parse_zeroshot_pred(text: str | None) -> str:
    """The reply should be a bare E, S, C or I. Anything else -> empty string."""
    value = (text or "").strip().upper()
    if value not in PRED_TO_LABEL:
        return ""
    return value



def evaluate_preds(gold_rows: list[dict], preds: list[str]) -> dict:
    assert len(preds) <= len(gold_rows), f"{len(preds)} predictions for {len(gold_rows)} lines"
    letters = [p if p in PRED_TO_LABEL else handle_missing(g["example_id"]) for g, p in zip(gold_rows, preds)]
    letters += [handle_missing(g["example_id"]) for g in gold_rows[len(preds):]]
    y_true = [g["label"] for g in gold_rows]
    y_pred = [PRED_TO_LABEL[p] for p in letters]
    # see https://github.com/amazon-science/esci-data/blob/7916cdf6ab75a462e77f20ab40428a10923998d5/classification_identification/inference.py#L125
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def score_run(dataset_config, out_path: Path) -> dict:
    """Score one run's out file against the sample's gold."""
    ds = ESCI(DATASETS_DIR / dataset_config["root"], dataset_config["sample"])
    gold_rows = [zeroshot_gold_row(ds[i]) for i in range(len(ds))]
    with out_path.open(encoding="utf-8") as f:
        preds = [parse_zeroshot_pred(line) for line in f]
    return evaluate_preds(gold_rows, preds)

