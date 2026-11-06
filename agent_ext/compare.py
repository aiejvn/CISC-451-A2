"""Compare ablation variants: exact match by difficulty, paired McNemar tests, and per-habit diagnostics.

    python -m agent_ext.compare                    # latest run of every variant
    python -m agent_ext.compare --run-id full      # run "full" of every variant that has one

Exact match is recomputed per example with spider/evaluation.py's own functions (matching metrics.json), so variants
can be paired question by question. Execution accuracy is not used: the local databases are schema-only.
"""
import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

from . import RESULTS_DIR
from .run import VARIANTS
from .scoring import score_row

LEVELS = ["easy", "medium", "hard", "extra", "all"]
PAIRS = [("rag", "base"), ("react", "base"), ("react_parse", "react"), ("rag_react", "rag"),
         ("rag_react_parse", "rag_react"), ("rag_react", "base"), ("rag_react_parse", "base")]


def latest_run(variant: str, run_id: str | None) -> Path | None:
    variant_dir = RESULTS_DIR / variant
    if not variant_dir.exists():
        return None
    runs = [d for d in variant_dir.iterdir() if (d / "pred.txt").exists() and (run_id is None or d.name == run_id)]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=None, help="use this run id for every variant (default: latest per variant)")
    args = ap.parse_args()

    variants, scored, rows = {}, {}, {}
    for variant in VARIANTS:
        run = latest_run(variant, args.run_id)
        if run is None:
            continue
        variants[variant] = run
        rows[variant] = {r["id"]: r for r in map(json.loads, (run / "generations.jsonl").read_text().splitlines())}
        scored[variant] = {r["id"]: score_row(r) for r in rows[variant].values()}
    if not variants:
        raise SystemExit(f"no finished runs under {RESULTS_DIR}")
    common = set.intersection(*(set(s) for s in scored.values()))
    print(f"{len(variants)} variants, {len(common)} questions common to all\n")

    print("Exact match (%) by difficulty\n")
    print("| variant | run | " + " | ".join(LEVELS) + " |\n|---|---|" + "---|" * len(LEVELS))
    for variant, s in scored.items():
        cells = []
        for level in LEVELS:
            part = [s[i]["exact"] for i in common if level == "all" or s[i]["hardness"] == level]
            cells.append(f"{pct(sum(part) / len(part))} (n={len(part)})" if part else "-")
        print(f"| {variant} | {variants[variant].name} | " + " | ".join(cells) + " |")

    print("\nPaired comparisons (exact match). McNemar exact test on the questions where the two variants disagree.\n")
    print("| A vs B | A | B | gained (B wrong, A right) | lost (B right, A wrong) | p |\n|---|---|---|---|---|---|")
    for a, b in PAIRS:
        if a in scored and b in scored:
            gained = sum(scored[a][i]["exact"] > scored[b][i]["exact"] for i in common)
            lost = sum(scored[a][i]["exact"] < scored[b][i]["exact"] for i in common)
            p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
            ea, eb = (sum(scored[x][i]["exact"] for i in common) / len(common) for x in (a, b))
            print(f"| {a} vs {b} | {pct(ea)} | {pct(eb)} | {gained} | {lost} | {p:.3g} |")

    print("\nHabit diagnostics (%)\n")
    print("| variant | parses (Spider) | set op used, when gold has one | more SELECT columns than gold | "
          "ReAct: avg steps | ReAct: finished | ReAct: SQL changed after an observation |\n|---|---|---|---|---|---|---|")
    for variant, s in scored.items():
        ids = sorted(common)
        setop = [s[i]["pred_setop"] for i in ids if s[i]["gold_setop"]]
        parsed = [i for i in ids if s[i]["parses"]]
        cells = [pct(len(parsed) / len(ids)), pct(sum(setop) / len(setop)) if setop else "-",
                 pct(sum(s[i]["extra_cols"] for i in parsed) / len(parsed)) if parsed else "-"]
        if VARIANTS[variant][1]:
            rr = [rows[variant][i] for i in ids]
            changed = [len({t["sql"] for t in r["trace"] if t["sql"]}) > 1 for r in rr]
            cells += [f"{sum(r['steps'] for r in rr) / len(rr):.2f}", pct(sum(r["finished"] for r in rr) / len(rr)),
                      pct(sum(changed) / len(rr))]
        else:
            cells += ["-", "-", "-"]
        print(f"| {variant} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
