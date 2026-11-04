"""Compare ablation arms: exact match by difficulty, paired McNemar tests, and per-habit diagnostics.

    python -m agent_ext.compare                    # latest run of every arm
    python -m agent_ext.compare --run-id full      # run "full" of every arm that has one

Exact match is recomputed per example with spider/evaluation.py's own functions (matching metrics.json), so arms
can be paired question by question. Execution accuracy is not used: the local databases are schema-only.
"""
import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

from . import RESULTS_DIR
from .run import ARMS
from .scoring import score_row

LEVELS = ["easy", "medium", "hard", "extra", "all"]
PAIRS = [("rag", "base"), ("react", "base"), ("react_parse", "react"), ("rag_react", "rag"),
         ("rag_react_parse", "rag_react"), ("rag_react", "base"), ("rag_react_parse", "base")]


def latest_run(arm: str, run_id: str | None) -> Path | None:
    arm_dir = RESULTS_DIR / arm
    if not arm_dir.exists():
        return None
    runs = [d for d in arm_dir.iterdir() if (d / "pred.txt").exists() and (run_id is None or d.name == run_id)]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=None, help="use this run id for every arm (default: latest per arm)")
    args = ap.parse_args()

    arms, scored, rows = {}, {}, {}
    for arm in ARMS:
        run = latest_run(arm, args.run_id)
        if run is None:
            continue
        arms[arm] = run
        rows[arm] = {r["id"]: r for r in map(json.loads, (run / "generations.jsonl").read_text().splitlines())}
        scored[arm] = {r["id"]: score_row(r) for r in rows[arm].values()}
    if not arms:
        raise SystemExit(f"no finished runs under {RESULTS_DIR}")
    common = set.intersection(*(set(s) for s in scored.values()))
    print(f"{len(arms)} arms, {len(common)} questions common to all\n")

    print("Exact match (%) by difficulty\n")
    print("| arm | run | " + " | ".join(LEVELS) + " |\n|---|---|" + "---|" * len(LEVELS))
    for arm, s in scored.items():
        cells = []
        for level in LEVELS:
            part = [s[i]["exact"] for i in common if level == "all" or s[i]["hardness"] == level]
            cells.append(f"{pct(sum(part) / len(part))} (n={len(part)})" if part else "-")
        print(f"| {arm} | {arms[arm].name} | " + " | ".join(cells) + " |")

    print("\nPaired comparisons (exact match). McNemar exact test on the questions where the two arms disagree.\n")
    print("| A vs B | A | B | gained (B wrong, A right) | lost (B right, A wrong) | p |\n|---|---|---|---|---|---|")
    for a, b in PAIRS:
        if a in scored and b in scored:
            gained = sum(scored[a][i]["exact"] > scored[b][i]["exact"] for i in common)
            lost = sum(scored[a][i]["exact"] < scored[b][i]["exact"] for i in common)
            p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
            ea, eb = (sum(scored[x][i]["exact"] for i in common) / len(common) for x in (a, b))
            print(f"| {a} vs {b} | {pct(ea)} | {pct(eb)} | {gained} | {lost} | {p:.3g} |")

    print("\nHabit diagnostics (%)\n")
    print("| arm | parses (Spider) | set op used, when gold has one | more SELECT columns than gold | "
          "ReAct: avg steps | ReAct: finished | ReAct: SQL changed after an observation |\n|---|---|---|---|---|---|---|")
    for arm, s in scored.items():
        ids = sorted(common)
        setop = [s[i]["pred_setop"] for i in ids if s[i]["gold_setop"]]
        parsed = [i for i in ids if s[i]["parses"]]
        cells = [pct(len(parsed) / len(ids)), pct(sum(setop) / len(setop)) if setop else "-",
                 pct(sum(s[i]["extra_cols"] for i in parsed) / len(parsed)) if parsed else "-"]
        if ARMS[arm][1]:
            rr = [rows[arm][i] for i in ids]
            changed = [len({t["sql"] for t in r["trace"] if t["sql"]}) > 1 for r in rr]
            cells += [f"{sum(r['steps'] for r in rr) / len(rr):.2f}", pct(sum(r["finished"] for r in rr) / len(rr)),
                      pct(sum(changed) / len(rr))]
        else:
            cells += ["-", "-", "-"]
        print(f"| {arm} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
