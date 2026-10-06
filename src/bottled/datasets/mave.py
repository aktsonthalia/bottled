import orjson
import random
import re
import string

from collections import Counter
from functools import partial
from pathlib import Path
from tqdm import tqdm


from bottled.constants import DATASETS_DIR
from bottled.datasets.blocklists.mave import BLOCKED_URL_PATTERNS

from . import common


VIEWS = DATASETS_DIR / "mave" / "views"

EVAL_SAMPLE_DEFAULT_N = 1000
EVAL_SAMPLE_SEED = 0


# region copied from MAVE: https://github.com/google-research/google-research/blob/08a8d6736475776f42ffac23b2c13111a28e5795/mave/benchmark/run_inference.py
# Copyright 2026 The Google Research Authors.
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
# (comments added)

_A_A = 'a_a'
_A_B = 'a_b'
_A_N = 'a_n'
_N_A = 'n_a'
_N_N = 'n_n'
_MEASURE_STRS = (_A_A, _A_B, _A_N, _N_A, _N_N)

def _normalize_text_squad(answer_text):
  """Text normalization from SQuAD 2.0 eval script."""

  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, u' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(answer_text))))


def get_measure_str(true_evidences,
                    top_prediction):
  """Returns a string representing the eval result."""
  normalized_true_evidences = set()
  for evidence in true_evidences:
    normalized_evidence = _normalize_text_squad(evidence['value'])  # pyrefly: ignore[bad-argument-type]
    if normalized_evidence:
      normalized_true_evidences.add(normalized_evidence)

  normalized_top_prediction = _normalize_text_squad(top_prediction)  # pyrefly: ignore[bad-argument-type]

  if not normalized_true_evidences and not normalized_top_prediction:
    # A means "Value"; paper uses V for this
    # B means "incorrect value"; paper uses W for this
    return _N_N # true negative, NN in paper
  elif not normalized_true_evidences:
    return _N_A # false positive, NV in paper
  elif not normalized_top_prediction:
    return _A_N # false negative, VN in paper
  elif normalized_top_prediction in normalized_true_evidences:
    return _A_A # true positive, correct, VC in paper
  else:
    return _A_B # true positive, incorrect, VW in paper

# endregion



class MAVE(common.JsonlDataset):

    BLOCKED_URL_PATTERNS = BLOCKED_URL_PATTERNS


def _sample_dataset(
    src: Path, 
    out_dir: Path, 
    seed: int, 
    num: int | None = None
) -> tuple[Path, Path, Path]: # reviewed 15 Sep 2026
    """Shuffle `src` with `seed`, keep the first `num` lines (all if None), and write each kept line as a question line (attribute key + paragraphs), a gold line (id, category, evidences), and an offsets line (line index and byte offset in `src`)."""
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
            attributes = item.pop("attributes")
            assert len(attributes) == 1, "expected exactly one attribute"
            attribute = attributes[0]
            paragraphs = item.pop("paragraphs")
            q.write(orjson.dumps({"attribute": attribute["key"], "paragraphs": paragraphs}) + b"\n")
            gt.write(orjson.dumps({**item, "evidences": attribute["evidences"]}) + b"\n")
            off.write(orjson.dumps({"line": line_index, "offset": offset}) + b"\n")
    return q_path, gt_path, offsets_path


sample = partial(common.sample, _sample_dataset, VIEWS)


sandbox_input = common.jsonl_sandbox_input


zeroshot_model_input = common.zeroshot_model_input


GOLD_FIELDS = ("id", "category", "attribute", "evidences")

zeroshot_gold_row = partial(common.zeroshot_gold_row, gold_fields=GOLD_FIELDS)

NO_VALUE = "<NO_VALUE>"


def parse_zeroshot_pred(text: str) -> str:
    """The reply should be the bare attribute value, or NO_VALUE when the text states no value."""
    value = text.strip()
    return "" if value == NO_VALUE else value


# region scoring # reviewed 15 Sep 2026
MEASURES = ("a_a", "a_b", "a_n", "n_a", "n_n")


def evaluate_preds(gold_rows: list[dict], preds: list[str]) -> dict:
    """Precision, recall and F1 as defined in the MAVE paper, from the per-line measure counts."""
    assert len(gold_rows) == len(preds), f"line count mismatch: {len(gold_rows)} gold vs {len(preds)} pred"
    counts = Counter(get_measure_str(g["evidences"], p) for g, p in zip(gold_rows, preds))
    
    n_predicted = counts["n_a"] + counts["a_a"] + counts["a_b"]  # lines where a value was predicted
    n_gold = counts["a_n"] + counts["a_a"] + counts["a_b"]  # lines where gold has a value
    precision = counts["a_a"] / n_predicted if n_predicted else 0.0
    recall = counts["a_a"] / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n_scored": len(preds), "counts": {m: counts[m] for m in MEASURES}}


def score_run(dataset_config, out_path: Path) -> dict:
    """Score one run's out file against the sample's gold."""
    ds = MAVE(DATASETS_DIR / dataset_config["root"], dataset_config["sample"])
    gold_rows = [zeroshot_gold_row(ds[i]) for i in range(len(ds))]
    with out_path.open(encoding="utf-8") as f:
        preds = [parse_zeroshot_pred(line) for line in f]
    return evaluate_preds(gold_rows, preds)
# endregion


if __name__ == "__main__":
    ds = MAVE()
    print(len(ds))
    print(ds[0])
