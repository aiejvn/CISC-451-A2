"""Save n random output traces each from the successes and failures (Spider exact match) of a run.

    python -m agent_ext.traces agent_ext/results/rag_react/<run_id> --n 10 --seed 0

Writes <run_dir>/traces/{success,failure}.jsonl (full rows, with the scoring flags) and traces/sample.md (readable).
run.py calls this automatically with --save-traces n.
"""
import argparse
import json
import random
from pathlib import Path

from .scoring import score_row
from eval.evalpipeline.common import load_train


def _md(row: dict, train: list[dict] | None) -> str:
    s = row["score"]
    out = [f"### dev #{row['id']} ({row['db_id']}, {s['hardness']})",
           f"**Question:** {row['question']}", "",
           f"**Gold:** `{' '.join(row['gold'].split())}`", "",
           f"**Pred:** `{row['pred']}`", "",
           f"Parses in Spider: {s['parses']}. Gold uses a set op: {s['gold_setop']}; pred does: {s['pred_setop']}."]
    if row.get("fewshot_ids") and train:
        out += ["", "<details><summary>Retrieved few-shot examples</summary>", ""]
        out += [f"- {train[i]['question']} → `{' '.join(train[i]['query'].split())}`" for i in row["fewshot_ids"]]
        out += ["", "</details>"]
    if row.get("trace"):
        out += ["", f"Steps: {row['steps']}, finished: {row['finished']}", ""]
        for t in row["trace"]:
            out += [f"**Step {t['step']}**", "````", t["output"], "````"]
            if t["observation"]:
                out += [f"> Observation: {t['observation']}".replace("\n", "\n> "), ""]
    elif row.get("raw") is not None:
        out += ["", "**Raw output:**", "````", row["raw"].strip(), "````"]
    return "\n".join(out) + "\n"


def save_traces(run_dir: Path, n: int, seed: int = 0) -> dict[str, int]:
    """Samples min(n, available) successes and failures; returns how many of each exist."""
    run_dir = Path(run_dir)
    rows = [json.loads(l) for l in (run_dir / "generations.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r["score"] = score_row(r)
    groups = {"success": [r for r in rows if r["score"]["exact"]], "failure": [r for r in rows if not r["score"]["exact"]]}
    rng = random.Random(seed)
    picked = {k: sorted(rng.sample(v, min(n, len(v))), key=lambda r: r["id"]) for k, v in groups.items()}

    out_dir = run_dir / "traces"
    out_dir.mkdir(exist_ok=True)
    train = load_train() if any(r.get("fewshot_ids") for r in rows) else None
    md = [f"# Sampled traces: {run_dir.parent.name} / {run_dir.name}",
          f"{len(groups['success'])} successes and {len(groups['failure'])} failures out of {len(rows)} "
          f"(Spider exact match); {n} sampled from each with seed {seed}.\n"]
    for kind, sample in picked.items():
        (out_dir / f"{kind}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sample))
        md += [f"## {kind.capitalize()} ({len(sample)})\n"] + [_md(r, train) for r in sample]
    (out_dir / "sample.md").write_text("\n".join(md))
    return {k: len(v) for k, v in groups.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    counts = save_traces(args.run_dir, args.n, args.seed)
    print(f"{counts} -> {args.run_dir / 'traces'}")


if __name__ == "__main__":
    main()
