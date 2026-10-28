"""Print a run's outputs for eyeballing: outcome counts, scores, and per-example raw/pred/gold.

    python -m eval.evalpipeline.inspect_run eval/results/Qwen__Qwen3-0.6B/diag1
    python -m eval.evalpipeline.inspect_run eval/results/Qwen__Qwen3-0.6B            # newest run of the model
    python -m eval.evalpipeline.inspect_run <run> --only cut --show 5 --full
    python -m eval.evalpipeline.inspect_run <run> --id 12 345

Outcomes (from breakdown.outcome): ok = finished with SQL | nosql = finished, no usable SQL | cut = hit
max_new_tokens or loop-stopped.
"""
import argparse
import json
import statistics
from pathlib import Path

from .breakdown import load, outcome
from .evaluate import find_runs

LEVELS = ["easy", "medium", "hard", "extra", "all"]
SHORT = {"finished with SQL": "ok", "finished, no usable SQL": "nosql",
         "cut while answering": "cut", "cut while thinking": "cut"}


def header(run: Path, rows: list[dict], limit: int) -> None:
    toks = [r["new_tokens"] for r in rows]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    print(f"run: {run}")
    print(f"n = {len(rows)}   max_new_tokens = {limit}   new_tokens median = {statistics.median(toks):.0f}, "
          f"max = {max(toks)}   empty preds = {sum(not r['pred'].strip() for r in rows)}")
    print("outcomes: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))

    metrics, raw = run / "metrics.json", run / "eval_raw.txt"
    if metrics.exists():
        m = json.loads(metrics.read_text())
        for name, section, row in (("exact", "exact matching accuracy", "exact match"),
                                   ("exec ", "execution accuracy", "execution")):
            s = m.get(section, {}).get(row)
            if s:
                print(f"{name}: " + "  ".join(f"{l} {s.get(l, float('nan')):.3f}" for l in LEVELS))
    elif raw.exists():
        print(f"no metrics.json; evaluation output tail ({raw}):")
        print("\n".join(raw.read_text().splitlines()[-20:]))
    else:
        print("not evaluated yet (python -m eval.evalpipeline.evaluate <run>)")


def show(r: dict, full: bool, tail: int) -> None:
    raw = r["raw"] if full or len(r["raw"]) <= tail else "..." + r["raw"][-tail:]
    think = f"   thinking: {len(r['thinking'])} chars" if r.get("thinking") else ""
    print(f"\n=== id {r['id']}  [{r['db_id']}]  {r['outcome']}  new_tokens={r['new_tokens']}{think}")
    print(f"Q:    {r['question']}")
    print(f"gold: {r['gold']}")
    print(f"pred: {r['pred']}")
    print(f"raw:\n{raw}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect one run's generations.")
    ap.add_argument("path", type=Path, help="a run dir (has generations.jsonl) or a model dir (newest run is used)")
    ap.add_argument("--show", type=int, default=10, help="rows to print after filtering (0 = header only)")
    ap.add_argument("--only", choices=["ok", "nosql", "cut"], help="filter rows by outcome")
    ap.add_argument("--id", type=int, nargs="*", help="only these example ids")
    ap.add_argument("--full", action="store_true", help="print the whole raw generation")
    ap.add_argument("--tail", type=int, default=400, help="raw chars to print without --full")
    args = ap.parse_args()

    runs = find_runs(args.path)
    if not runs:
        raise SystemExit(f"no runs with generations.jsonl under {args.path}")
    run = max(runs, key=lambda d: (d / "generations.jsonl").stat().st_mtime)
    rows, limit = load(run)
    for r in rows:
        r["outcome"] = outcome(r, limit)
    header(run, rows, limit)

    sel = [r for r in rows if (not args.only or SHORT[r["outcome"]] == args.only)
           and (not args.id or r["id"] in args.id)]
    for r in sel[: args.show]:
        show(r, args.full, args.tail)
    if len(sel) > args.show:
        print(f"\n({len(sel) - args.show} more; raise --show)")


if __name__ == "__main__":
    main()
