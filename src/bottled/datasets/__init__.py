from pathlib import Path

from . import mave, esci, raid


register = {
    "mave": mave.MAVE,
    "esci": esci.ESCI,
    "raid": raid.RAID,
}

input_providers = {
    "mave": mave.sandbox_input,
    "esci": esci.sandbox_input,
    "raid": raid.sandbox_input,
}

# instance -> user turn
zeroshot_model_inputs = {
    "mave": mave.zeroshot_model_input,
    "esci": esci.zeroshot_model_input,
    "raid": raid.zeroshot_model_input,
}

# instance -> gold
zeroshot_gold_rows = {
    "mave": mave.zeroshot_gold_row,
    "esci": esci.zeroshot_gold_row,
    "raid": raid.zeroshot_gold_row,
}

# reply -> pred
zeroshot_pred_parsers = {
    "mave": str.strip,
    "esci": esci.parse_zeroshot_pred,
    "raid": str.strip,
}

scorers = {
    "mave": mave.score_run,
    "esci": esci.score_run,
    "raid": raid.score_run,
}

def load_from_config(config: dict):
    config = dict(config)
    name = config.pop("name")
    root = Path(config.pop("root"))
    dataset = register[name](root=root, **config)
    return dataset
