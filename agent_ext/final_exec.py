"""Executes each variant's final SQL on the (schema-only) Spider dev databases and plots the failure rate by error kind.

    python -m agent_ext.final_exec [--run-id full]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .tools import execute

RESULTS = Path(__file__).parent / "results"
VARIANTS = ["base", "rag", "react", "react_parse", "rag_react", "rag_react_parse"]
KINDS = ["no such column", "no such table", "syntax error", "other"]
KIND_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def kind(obs: str) -> str | None:
    if obs.startswith("OK"):
        return None
    m = obs.lower()
    for k in KINDS[:3]:
        if re.search(k.replace("syntax error", r"syntax error|incomplete input"), m):
            return k
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full")
    args = ap.parse_args()
    res = {}
    for variant in VARIANTS:
        rows = [json.loads(l) for l in (RESULTS / variant / args.run_id / "generations.jsonl").read_text().splitlines() if l.strip()]
        res[variant] = (len(rows), Counter(k for r in rows if (k := kind(execute(r["db_id"], r["pred"])))))

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    left = [0.0] * len(VARIANTS)
    for k, col in zip(KINDS, KIND_COLORS):
        pct = [100 * res[a][1][k] / res[a][0] for a in VARIANTS]
        ax.barh(range(len(VARIANTS)), pct, 0.55, left=left, color=col, edgecolor=SURFACE, linewidth=2, label=k)
        left = [l + p for l, p in zip(left, pct)]
    for y, l in enumerate(left):
        ax.text(l + 0.15, y, f"{l:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(VARIANTS)), VARIANTS, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of final SQL that fails to execute in SQLite (n=1034 per variant)", color=MUTED, fontsize=9)
    ax.set_title("Final SQL execution failures", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    out = RESULTS / "figs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "final_exec_errors.png", dpi=150)
    for a in VARIANTS:
        n, c = res[a]
        print(f"{a:16s} {sum(c.values()):4d}/{n} = {100*sum(c.values())/n:5.1f}%  {dict(c)}")


if __name__ == "__main__":
    main()
