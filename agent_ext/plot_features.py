"""Figures of the two RQ2 trajectory features of System B (rag_react_parse), as defined in report/RQ2.tex.

    python -m agent_ext.plot_features [--run-id full]

traj_repair.png: repair success rate per tool and difficulty. A problem is a (question, tool) pair with an error
    observation from that tool; it is repaired if re-running the tool on the final submitted query no longer gives the error.
traj_distance.png: semantic distance 1 - cos(first, last SQL-parsable draft) with all-MiniLM-L6-v2, by difficulty and
    exact match; "unchanged" is the share of trajectories with distance 0.
Writes both to report/ by default (the figures \\includegraphics'd by RQ2.tex) and prints the numbers behind them.
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import REPO
from .plot_trajectories import INK, MUTED, SURFACE, style
from .report_tables import B, _ERRORS, _parse, load, scored

LEVELS = ["easy", "medium", "hard", "extra", "all"]
LABELS = ["Easy", "Medium", "Hard", "Extra", "All"]
TOOL_COLORS = {"execute": "#2a78d6", "spider_parse": "#eb6834"}  # categorical slots 1-2
EM_COLORS = {True: "#1baf7a", False: "#eda100"}  # slots 3-4, so they never read as the tool colours of the repair figure
OUT_DIR = REPO / "report"
REPAIR_PNG, DISTANCE_PNG = "traj_repair.png", "traj_distance.png"


def repair(rows: list[dict], sc: dict) -> dict:
    """(tool, level) -> [problems flagged, repaired]."""
    out = {(t, l): [0, 0] for t in _ERRORS for l in LEVELS}
    for r in rows:
        for tool, (marker, fn) in _ERRORS.items():
            if any(t["action"] == tool and (t["observation"] or "").startswith(marker) for t in r["trace"]):
                fixed = not fn(r["db_id"], r["pred"]).startswith(marker)
                for l in (sc[r["id"]]["hardness"], "all"):
                    out[tool, l][0] += 1
                    out[tool, l][1] += fixed
    return out


def distances(rows: list[dict], sc: dict, models_path: Path) -> list[tuple[str, bool, float]]:
    """(hardness, exact match, distance) for every trajectory with at least one SQL-parsable draft."""
    from .retriever import Retriever  # only for its MiniLM embed(); the retrieval pool is irrelevant here
    emb = Retriever(5, models_path, True)
    pairs = []
    for r in rows:
        drafts = [t["sql"] for t in r["trace"] if t["sql"] and _parse(r["db_id"], t["sql"]) is not None]
        if drafts:
            pairs.append((r, drafts[0], drafts[-1]))
    e = emb.embed([x for _, a, b in pairs for x in (a, b)]).reshape(len(pairs), 2, -1)
    d = 1 - (e[:, 0] * e[:, 1]).sum(1)
    return [(sc[r["id"]]["hardness"], bool(sc[r["id"]]["exact"]), max(float(x), 0.0)) for (r, _, _), x in zip(pairs, d)]


def plot_repair(ax, rep: dict) -> None:
    w = 0.38
    for i, tool in enumerate(_ERRORS):
        for j, l in enumerate(LEVELS):
            n, k = rep[tool, l]
            x = j + (i - 0.5) * w
            if n:
                p = 100 * k / n
                ax.bar(x, p, w * 0.9, color=TOOL_COLORS[tool])
                ax.text(x, p + 1.5, f"{p:.0f}%\n{k}/{n}", ha="center", fontsize=7, color=INK, linespacing=1.1)
            else:
                ax.text(x, 1.5, "n=0", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(LEVELS)), LABELS)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 25))
    style(ax, "Repair success rate of System B, by tool and difficulty", ylabel="problems repaired (%)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in TOOL_COLORS.values()], list(TOOL_COLORS),
              frameon=False, fontsize=8, loc="upper left", ncol=2)


def plot_distance(ax, recs: list[tuple[str, bool, float]]) -> None:
    rng = random.Random(0)
    w = 0.38
    top = max(d for *_, d in recs) * 1.25
    for i, em in enumerate((True, False)):
        for j, l in enumerate(LEVELS):
            ds = [d for h, e, d in recs if e == em and (l == "all" or h == l)]
            x = j + (i - 0.5) * w
            if not ds:
                continue
            xs = [x + rng.uniform(-w * 0.35, w * 0.35) for _ in ds]
            ax.scatter(xs, ds, s=9, color=EM_COLORS[em], alpha=0.45, linewidths=0)
            mean = sum(ds) / len(ds)
            ax.plot([x - w * 0.4, x + w * 0.4], [mean, mean], color=INK, lw=2)
            unchanged = 100 * sum(d < 1e-6 for d in ds) / len(ds)
            ax.text(x, top * 0.93, f"{unchanged:.0f}%\nunchanged", ha="center", fontsize=7, color=MUTED, linespacing=1.1)
    ax.set_xticks(range(len(LEVELS)), LABELS)
    ax.set_ylim(-top * 0.03, top)
    style(ax, "Semantic distance between first and final parsable draft of System B", ylabel="1 − cos(first, final)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, alpha=0.8) for c in EM_COLORS.values()]
    handles.append(plt.Line2D([], [], color=INK, lw=2))
    ax.legend(handles, ["exact match", "not exact match", "mean"], frameon=False, fontsize=8, loc="center left",
              bbox_to_anchor=(0, 0.72))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default="full")
    ap.add_argument("--models-path", type=Path, default=REPO / "a1_lib" / "models")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help=f"default: report/, where RQ2.tex reads {REPAIR_PNG} and {DISTANCE_PNG}")
    args = ap.parse_args()
    rows, _ = load(B, args.run_id)
    sc = scored(rows)
    rep = repair(rows, sc)
    recs = distances(rows, sc, args.models_path)

    for name, draw, data in ((REPAIR_PNG, plot_repair, rep), (DISTANCE_PNG, plot_distance, recs)):
        fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
        draw(ax, data)
        fig.tight_layout()
        fig.savefig(args.out_dir / name, dpi=200)
        plt.close(fig)
        print(f"wrote {args.out_dir / name}")
    for (tool, l), (n, k) in rep.items():
        print(f"repair {tool:12s} {l:6s} {k}/{n}")
    for l in LEVELS:
        for em in (True, False):
            ds = [d for h, e, d in recs if e == em and (l == "all" or h == l)]
            if ds:
                print(f"distance {l:6s} exact={em!s:5s} n={len(ds):4d} mean={sum(ds)/len(ds):.3f} "
                      f"unchanged {100*sum(d < 1e-6 for d in ds)/len(ds):.0f}%")


if __name__ == "__main__":
    main()
