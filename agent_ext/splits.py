"""70:30 split of Spider train; the 30% "design" slice is the dev set we used to design System B.

    python -m agent_ext.splits            # (re)writes agent_ext/splits/train_design_seed0.json

The split is a seeded shuffle of train.json indices (the "id" every run records), so it is fully determined
by SEED and DESIGN_FRAC. The Spider dev set (dev.json) stays the held-out evaluation set.
"""
import json
import random

from . import EXT_ROOT
from eval.evalpipeline.common import TRAIN_JSON, load_train

SEED = 0
DESIGN_FRAC = 0.3
SPLIT_JSON = EXT_ROOT / "splits" / f"train_design_seed{SEED}.json"


def make_split(n: int, seed: int = SEED, frac: float = DESIGN_FRAC) -> tuple[list[int], list[int]]:
    """(fit_ids, design_ids), each sorted: a seeded shuffle of range(n), with the first round(frac * n) as design."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = round(frac * n)
    return sorted(idx[k:]), sorted(idx[:k])


def fit_ids() -> list[int]:
    """train.json indices of the 70% fit part, the only examples System B may retrieve."""
    return json.loads(SPLIT_JSON.read_text())["fit_ids"]


def load_design() -> list[dict]:
    """The design slice of Spider train, with "id" = index in train.json (as in load_train)."""
    design = set(json.loads(SPLIT_JSON.read_text())["design_ids"])
    return [ex for ex in load_train() if ex["id"] in design]


def main() -> None:
    n = len(load_train())
    fit, design = make_split(n)
    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON.write_text(json.dumps({
        "source": TRAIN_JSON.name, "n_train": n, "seed": SEED, "design_frac": DESIGN_FRAC,
        "method": "random.Random(seed).shuffle(list(range(n_train))); first round(design_frac * n_train) = design",
        "n_fit": len(fit), "n_design": len(design), "fit_ids": fit, "design_ids": design}))
    print(f"fit {len(fit)} / design {len(design)} (seed {SEED}) -> {SPLIT_JSON}")


if __name__ == "__main__":
    main()
