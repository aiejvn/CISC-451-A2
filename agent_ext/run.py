"""Run one ablation arm of Qwen3-4B on Spider dev, then score it with the official evaluator.

    python -m agent_ext.run --arm rag --local-only --models-path a1_lib/models
    python -m agent_ext.run --arm rag_react_parse --limit 20 --run-id smoke --local-only --models-path a1_lib/models
    bash run_slurm.sh -m agent_ext.run --arm react --local-only --models-path a1_lib/models   # on slurm

Arms: few-shot (k nearest train examples) on/off x ReAct loop off / execute / execute + spider_parse.
Output: agent_ext/results/<arm>/<run_id>/{generations.jsonl, gold.txt, pred.txt, metrics.json, config_used.yaml}
and traces/{success,failure}.jsonl + traces/sample.md (--save-traces N random examples of each).
"""
import argparse
import csv
import gc
import json
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from . import RESULTS_DIR
from .agent import Generator, ReActAgent, single_shot
from .prompts import single_shot_prompt
from eval.evalpipeline.adapters import get_adapter
from eval.evalpipeline.common import get_model_config, load_dev, write_gold
from eval.evalpipeline.evaluate import evaluate_run
from eval.evalpipeline.generate import load_done, load_model

MODEL = "Qwen/Qwen3-4B"
LEVELS = ["easy", "medium", "hard", "extra", "all"]
ARMS = {  # arm -> (few-shot, ReAct tools or None for single-shot)
    "base": (False, None),
    "rag": (True, None),
    "react": (False, ["execute"]),
    "react_parse": (False, ["execute", "spider_parse"]),
    "rag_react": (True, ["execute"]),
    "rag_react_parse": (True, ["execute", "spider_parse"]),
}


def append_summary(arm: str, run_dir: Path, n: int, metrics: dict) -> None:
    exact = metrics.get("exact matching accuracy", {}).get("exact match", {})
    execu = metrics.get("execution accuracy", {}).get("execution", {})
    path = RESULTS_DIR / "summary.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arm", "run", "n"] + [f"exact_{l}" for l in LEVELS] + [f"exec_{l}" for l in LEVELS])
        w.writerow([arm, run_dir.name, n] + [exact.get(l, "") for l in LEVELS] + [execu.get(l, "") for l in LEVELS])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--limit", type=int, default=None, help="subset size (default: full dev, 1034)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5, help="few-shot examples per question")
    ap.add_argument("--max-steps", type=int, default=4, help="ReAct steps per question")
    ap.add_argument("--batch-size", type=int, default=64, help="questions generated together (halved on CUDA OOM)")
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--models-path", type=Path, default=None)
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run with the same --run-id")
    ap.add_argument("--skip-generate", action="store_true", help="only re-evaluate an existing --run-id")
    ap.add_argument("--save-traces", type=int, default=5, metavar="N",
                    help="save N random traces each from successes and failures to <run_dir>/traces (0 = off)")
    args = ap.parse_args()
    if args.local_only and args.models_path is None:
        ap.error("--local-only requires --models-path")

    use_rag, tools = ARMS[args.arm]
    cfg = get_model_config(MODEL)
    examples = load_dev(args.limit, args.seed)
    run_dir = RESULTS_DIR / args.arm / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        fewshots = [None] * len(examples)
        if use_rag:
            from .retriever import Retriever
            retriever = Retriever(args.k, args.models_path, args.local_only)
            fewshots = retriever.topk([ex["question"] for ex in examples])
            del retriever
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        adapter = get_adapter(cfg)
        model, tokenizer = load_model(cfg, args.models_path, args.local_only)
        gen = Generator(model, tokenizer, adapter)
        agent = ReActAgent(gen, tools, args.max_steps) if tools else None

        write_gold(examples, run_dir / "gold.txt")
        (run_dir / "config_used.yaml").write_text(yaml.safe_dump({
            "arm": args.arm, "model": cfg, "generation": gen.gen_kwargs, "rag_k": args.k if use_rag else 0,
            "tools": tools, "max_steps": args.max_steps if tools else None, "batch_size": args.batch_size,
            "n_examples": len(examples),
            "seed": args.seed}))

        done = load_done(run_dir / "generations.jsonl") if args.resume else {}
        preds = {i: r["pred"] for i, r in done.items()}
        todo = [(ex, shots) for ex, shots in zip(examples, fewshots) if ex["id"] not in done]
        bs = args.batch_size
        with open(run_dir / "generations.jsonl", "a" if args.resume else "w") as gens:
            for start in tqdm(range(0, len(todo), bs), desc=f"{args.arm} (batches of {bs})"):
                batch = todo[start : start + bs]
                if agent:
                    outs = agent.run([(ex["question"], ex["db_id"], shots) for ex, shots in batch])
                else:
                    outs = single_shot(gen, [single_shot_prompt(ex["question"], ex["db_id"], shots) for ex, shots in batch],
                                       cfg["max_new_tokens"])
                for (ex, shots), out in zip(batch, outs):
                    preds[ex["id"]] = out["pred"]
                    row = {"id": ex["id"], "db_id": ex["db_id"], "question": ex["question"], "gold": ex["query"],
                           **out, "fewshot_ids": [s["train_id"] for s in shots] if shots else None}
                    gens.write(json.dumps(row) + "\n")
                gens.flush()
        (run_dir / "pred.txt").write_text("".join(f"{preds[ex['id']]}\n" for ex in examples))

    metrics = evaluate_run(run_dir, "all")
    append_summary(args.arm, run_dir, len(examples), metrics)
    if args.save_traces:
        from .traces import save_traces
        counts = save_traces(run_dir, args.save_traces, args.seed)
        print(f"traces: {counts} -> {run_dir / 'traces'}")
    print(f"{args.arm}: exact match (all) = {metrics['exact matching accuracy']['exact match']['all']:.3f} -> {run_dir}")


if __name__ == "__main__":
    main()
