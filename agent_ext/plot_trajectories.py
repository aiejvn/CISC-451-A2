"""Plots of ReAct trajectory metrics for the four loop variants.

    python -m agent_ext.plot_trajectories [--run-id full]

Writes agent_ext/results/figs/trajectories.png (all panels) and one PNG per panel.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
VARIANTS = ["react", "react_parse", "rag_react", "rag_react_parse"]
COLORS = dict(zip(VARIANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # categorical slots 1-4, fixed per variant
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def norm(sql: str | None) -> str:
    return " ".join((sql or "").lower().replace(";", " ").split())


def metrics(row: dict) -> dict:
    tr = row["trace"]
    obs = [t["observation"] or "" for t in tr]
    drafts = [norm(t["sql"]) for t in tr if t["sql"]]
    return {
        "steps": row["steps"],
        "exec_err": sum(o.startswith("ERROR") for o in obs),
        "reject": sum(o.startswith("PARSE ERROR") for o in obs),
        "malformed": sum(t["observation"] is not None and t["observation"].split(":")[0] not in ("OK", "ERROR", "PARSES", "PARSE ERROR")
                         for t in tr),
        "changed": bool(drafts) and norm(row["pred"]) != drafts[0],
        "stop": "called finish" if row["finished"] else "ran out of steps",
    }


def load(run_id: str) -> dict[str, list[dict]]:
    out = {}
    for variant in VARIANTS:
        path = RESULTS / variant / run_id / "generations.jsonl"
        out[variant] = [metrics(json.loads(l)) for l in path.read_text().splitlines() if l.strip()]
    return out


def style(ax, title, ylabel="% of questions"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def grouped(ax, data, key, cats, labels, variants, cap=None):
    """Grouped bars: x = category, one bar per variant, y = % of that variant's questions."""
    w = 0.8 / len(variants)
    for i, variant in enumerate(variants):
        vals = [min(m[key], cap) if cap is not None else m[key] for m in data[variant]]
        c = Counter(vals)
        pct = [100 * c[k] / len(vals) for k in cats]
        xs = [j + (i - (len(variants) - 1) / 2) * w for j in range(len(cats))]
        ax.bar(xs, pct, w * 0.88, color=COLORS[variant], label=variant)
        for x, p in zip(xs, pct):
            if p >= 0.5:
                ax.text(x, p + 0.3, f"{p:.0f}" if p >= 3 else f"{p:.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(cats)), labels)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)  # headroom for the value labels


def panels(data):
    def p_steps(ax):
        grouped(ax, data, "steps", [1, 2, 3, 4], ["1", "2", "3", "4"], VARIANTS)
        style(ax, "Steps per question"); ax.set_xlabel("steps taken", color=MUTED, fontsize=9)

    def p_exec(ax):
        grouped(ax, data, "exec_err", [1, 2, 3], ["1", "2", "3+"], VARIANTS, cap=3)
        style(ax, "SQLite execution errors (questions with 0 omitted)"); ax.set_xlabel("errors per question", color=MUTED, fontsize=9)

    def p_reject(ax):
        variants = ["react_parse", "rag_react_parse"]
        grouped(ax, data, "reject", [1, 2, 3], ["1", "2", "3+"], variants, cap=3)
        style(ax, "Verifier rejections, parse variants (0 omitted)"); ax.set_xlabel("rejections per question", color=MUTED, fontsize=9)

    def p_changed(ax):
        pct = [100 * sum(m["changed"] for m in data[a]) / len(data[a]) for a in VARIANTS]
        ax.bar(range(4), pct, 0.55, color=[COLORS[a] for a in VARIANTS])
        for x, p in enumerate(pct):
            ax.text(x, p + 0.4, f"{p:.1f}%", ha="center", fontsize=9, color=INK)
        ax.set_xticks(range(4), VARIANTS, fontsize=8)
        style(ax, "Final SQL differs from first draft")

    def p_stop(ax):
        reasons = ["called finish", "ran out of steps"]
        left = [0.0] * 4
        for r, col in zip(reasons, ["#2a78d6", "#eb6834"]):
            pct = [100 * sum(m["stop"] == r for m in data[a]) / len(data[a]) for a in VARIANTS]
            ax.barh(range(4), pct, 0.55, left=left, color=col, edgecolor=SURFACE, linewidth=2, label=r)
            for y, (l, p) in enumerate(zip(left, pct)):
                if p >= 4:
                    ax.text(l + p / 2, y, f"{p:.1f}%", ha="center", va="center", fontsize=8, color="white")
            left = [l + p for l, p in zip(left, pct)]
        ax.set_yticks(range(4), VARIANTS, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 100)
        style(ax, "Stop reason", ylabel=""); ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_xlabel("% of questions", color=MUTED, fontsize=9)

    def p_malformed(ax):
        grouped(ax, data, "malformed", [1, 2, 3], ["1", "2", "3+"], VARIANTS, cap=3)
        style(ax, "Malformed actions (0 omitted)"); ax.set_xlabel("per question (unknown tool, no SQL)", color=MUTED, fontsize=9)

    return {"steps": p_steps, "exec_errors": p_exec, "verifier_rejections": p_reject,
            "changed_from_draft": p_changed, "stop_reason": p_stop, "malformed_actions": p_malformed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full")
    args = ap.parse_args()
    data = load(args.run_id)
    fns = panels(data)
    out = RESULTS / "figs"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), facecolor=SURFACE)
    for ax, fn in zip(axes.flat, fns.values()):
        fn(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[a]) for a in VARIANTS]
    fig.legend(handles, VARIANTS, loc="upper right", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f"ReAct trajectories on Spider dev (n={len(data[VARIANTS[0]])} per variant)", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "trajectories.png", dpi=150)

    for name, fn in fns.items():
        f, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        fn(ax)
        if name not in ("changed_from_draft", "stop_reason"):
            ax.legend(frameon=False, fontsize=8)
        elif name == "stop_reason":
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        f.tight_layout()
        f.savefig(out / f"{name}.png", dpi=150)
        plt.close(f)

    print(f"wrote {out}")
    for a in VARIANTS:
        d = data[a]
        n = len(d)
        print(f"{a:16s} steps {sum(m['steps'] for m in d)/n:.2f}  exec_err/q {sum(m['exec_err'] for m in d)/n:.3f}  "
              f"reject/q {sum(m['reject'] for m in d)/n:.3f}  changed {100*sum(m['changed'] for m in d)/n:.1f}%  "
              f"out-of-steps {100*sum(m['stop']!='called finish' for m in d)/n:.1f}%")


if __name__ == "__main__":
    main()
