import orjson
import time
from pathlib import Path

from bottled.harness import Bind
from bottled.constants import DATASETS_DIR


def sample(sampling_fn, views_dir, num: int | None = None) -> tuple[Path, Path, Path]: # reviewed 15 Sep 2026
    """sample_dataset on views/full.jsonl."""
    seed = int(time.time())
    print(f"seed: {seed}")
    return sampling_fn(views_dir / "full.jsonl", views_dir, seed, num)


class JsonlDataset: # reviewed 15 Sep 2026

    def __init__(self, root: Path, sample: str, provide_labels: bool = True):
        """`sample` = a file sample() wrote, e.g. n1000_seed1789475000."""
        views = root / "views"
        self.questions = list((views / f"{sample}_questions.jsonl").open("rb"))
        self.gold = list((views / f"{sample}_gt.jsonl").open("rb")) if provide_labels else None

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> dict:
        record = orjson.loads(self.questions[idx])
        if self.gold is not None:
            record.update(orjson.loads(self.gold[idx]))
        return record
    

def jsonl_sandbox_input(dataset_cfg, output_dir: Path) -> list[Bind]: # reviewed 15 Sep 2026
    """Agent's view."""
    host = DATASETS_DIR / dataset_cfg.root / "views" / f"{dataset_cfg.sample}_questions.jsonl"
    return [Bind(host=host, sandbox="/task/input.jsonl", readonly=True)]


def zeroshot_model_input(example: dict) -> str:
    """user turn for one question."""
    return orjson.dumps(example).decode()


def zeroshot_gold_row(item: dict, gold_fields: tuple) -> dict:
    """gold for one record"""
    return {k: item[k] for k in gold_fields}