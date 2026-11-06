"""Recomputes the numbers in report/RQ1.tex and report/RQ2.tex from finished runs (agent_ext/results/<variant>/<run_id>).

    python -m agent_ext.report_tables [--run-id full]

System A is the `base` variant and System B is `rag_react_parse`. Each function returns a pandas DataFrame; the CLI prints them
all. Input-token counts and the RQ2 embedding distances load models from a1_lib/models (no GPU needed).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import RESULTS_DIR
from .run import VARIANTS, LEVELS, MODEL
from .scoring import _parse, score_row
from .tools import execute, spider_parse
from eval.evalpipeline.adapters import get_adapter
from eval.evalpipeline.common import build_user_prompt, get_model_config, load_train, resolve_local_path
from .prompts import react_messages

A, B = "base", "rag_react_parse"
COMPONENTS = {"SELECT": "select", "WHERE": "where", "GROUP BY": "group(no Having)", "ORDER BY": "order", "KEYWORDS": "keywords"}
REACT_VARIANTS = [a for a, (_, tools) in VARIANTS.items() if tools]


def run_dir(variant: str, run_id: str) -> Path:
    return RESULTS_DIR / variant / run_id


def load(variant: str, run_id: str) -> tuple[list[dict], dict]:
    d = run_dir(variant, run_id)
    rows = [json.loads(l) for l in (d / "generations.jsonl").read_text().splitlines() if l.strip()]
    return rows, json.loads((d / "metrics.json").read_text())


def scored(rows: list[dict]) -> dict[int, dict]:
    return {r["id"]: score_row(r) for r in rows}


def accuracy(run_id: str) -> pd.DataFrame:
    """RQ1 accuracy table: exact match (%) by difficulty, A vs B."""
    out = {}
    for name, variant in (("System A (base)", A), ("System B (agent)", B)):
        em = load(variant, run_id)[1]["exact matching accuracy"]["exact match"]
        out[name] = {l: 100 * em[l] for l in LEVELS}
    df = pd.DataFrame(out).T
    df.loc["Gain (points)"] = df.loc["System B (agent)"] - df.loc["System A (base)"]
    n = load(A, run_id)[1]["count"]["count"]
    df.columns = [f"{l} (n={int(n[l])})" for l in LEVELS]
    return df.round(1)


def execution_accuracy(run_id: str) -> pd.DataFrame:
    """RQ1 execution accuracy (%) by difficulty, A vs B, against the schema-only databases."""
    out = {}
    for name, variant in (("System A (base)", A), ("System B (agent)", B)):
        ex = load(variant, run_id)[1]["execution accuracy"]["execution"]
        out[name] = {l: 100 * ex[l] for l in LEVELS}
    df = pd.DataFrame(out).T
    df.loc["Gain (points)"] = df.loc["System B (agent)"] - df.loc["System A (base)"]
    n = load(A, run_id)[1]["count"]["count"]
    df.columns = [f"{l} (n={int(n[l])})" for l in LEVELS]
    return df.round(1)


def validity(run_id: str) -> pd.DataFrame:
    """RQ1 output validity: % of outputs that parse under Spider's process_sql, and % that finished cleanly (A stopped
    before max_new_tokens, i.e. was not cut off; B ended with finish rather than running out of steps)."""
    ra, rb = load(A, run_id)[0], load(B, run_id)[0]
    cap = get_model_config(MODEL)["max_new_tokens"]
    return pd.DataFrame({"System A": [100 * np.mean([s["parses"] for s in scored(ra).values()]),
                                      100 * np.mean([r["new_tokens"] < cap for r in ra])],
                         "System B": [100 * np.mean([s["parses"] for s in scored(rb).values()]),
                                      100 * np.mean([r["finished"] for r in rb])]},
                        index=["parses under process_sql (%)", "finishes cleanly (%)"]).round(1)


def components(run_id: str) -> pd.DataFrame:
    """RQ1 component matching: Spider's partial-matching accuracy (precision), recall and F1, over all questions."""
    cols = {}
    for name, variant in (("A", A), ("B", B)):
        m = load(variant, run_id)[1]
        for key, label in (("partial matching accuracy", "Acc."), ("partial matching recall", "Rec."), ("partial matching f1", "F1")):
            cols[(f"System {name}", label)] = {c: 100 * m[key][k]["all"] for c, k in COMPONENTS.items()}
    return pd.DataFrame(cols).round(1)


def whole_query(run_id: str) -> pd.DataFrame:
    """RQ1 exact-match and execution accuracy (%). Execution is against schema-only databases, so it is weak evidence."""
    rows = {}
    for name, variant in (("System A", A), ("System B", B)):
        m = load(variant, run_id)[1]
        rows[name] = {"Exact matching accuracy (%)": 100 * m["exact matching accuracy"]["exact match"]["all"],
                      "Execution accuracy (%)": 100 * m["execution accuracy"]["execution"]["all"]}
    return pd.DataFrame(rows).round(1)


def agreement(run_id: str) -> pd.DataFrame:
    """RQ1 per-question agreement on exact match between A and B, by difficulty (from the gold query) and overall."""
    sa, sb = scored(load(A, run_id)[0]), scored(load(B, run_id)[0])
    ids = sorted(set(sa) & set(sb))
    out = {}
    for l in LEVELS:
        lids = [i for i in ids if l == "all" or sa[i]["hardness"] == l]
        both = sum(sa[i]["exact"] and sb[i]["exact"] for i in lids)
        a_only = sum(sa[i]["exact"] and not sb[i]["exact"] for i in lids)
        b_only = sum(sb[i]["exact"] and not sa[i]["exact"] for i in lids)
        out[l] = {"Solved by both A and B": both, "Solved by A only": a_only, "Solved by B only": b_only,
                  "Solved by neither": len(lids) - both - a_only - b_only, "Total": len(lids)}
    return pd.DataFrame(out)


def _tokenizer(models_path: Path | None):
    from transformers import AutoTokenizer
    cfg = get_model_config(MODEL)
    path = resolve_local_path(MODEL, models_path, models_path is not None)
    return AutoTokenizer.from_pretrained(path, local_files_only=True), get_adapter(cfg).chat_template_kwargs()


def efficiency(run_id: str, models_path: Path | None = None) -> pd.DataFrame:
    """RQ1 mean cost per question. Input tokens are the chat-templated prompt length summed over every LLM call,
    rebuilt from the saved prompts and traces. Time is generation time of the whole run / n (needs timing.json)."""
    tok, kw = _tokenizer(models_path)

    def n_tokens(messages):
        return len(tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw),
                       add_special_tokens=False)["input_ids"])

    train = load_train()
    stats = {}
    for name, variant in (("System A", A), ("System B", B)):
        rows = load(variant, run_id)[0]
        use_rag, tools = VARIANTS[variant]
        in_tok, calls = [], []
        for r in rows:
            if not tools:
                # the original A1 prompt; few-shot examples are never used for A
                in_tok.append(n_tokens([{"role": "user", "content": build_user_prompt(r["question"], r["db_id"])}]))
                calls.append(1)
                continue
            shots = None
            if use_rag:
                shots = [{"question": train[i]["question"], "query": " ".join(train[i]["query"].split())} for i in r["fewshot_ids"]]
            msgs = react_messages(r["question"], r["db_id"], tools, 4, shots)
            total = 0
            for step in r["trace"]:
                total += n_tokens(msgs)
                msgs = msgs + [{"role": "assistant", "content": step["output"]}]
                if step["observation"] is not None:
                    msgs = msgs + [{"role": "user", "content": f"Observation: {step['observation']}"}]
            in_tok.append(total)
            calls.append(len(r["trace"]))
        timing = run_dir(variant, run_id) / "timing.json"
        secs = json.loads(timing.read_text())["generation_seconds"] / len(rows) if timing.exists() else float("nan")
        stats[name] = {"LLM calls": np.mean(calls), "Input tokens": np.mean(in_tok),
                       "Output tokens": np.mean([r["new_tokens"] for r in rows]), "Wall-clock time (s)": secs}
    df = pd.DataFrame(stats)
    df["B / A"] = df["System B"] / df["System A"]
    return df.round(2)


_TABLE_RE = re.compile(r"\b(?:from|join)\s+[\"`]?(\w+)", re.I)
_COL_ALIAS_RE = re.compile(r"\bas\s+\w+", re.I)  # also catches table aliases written with AS
_BARE_ALIAS_RE = re.compile(r"\b(?:from|join)\s+\w+\s+(?!(?:as|where|join|on|group|order|limit|inner|left|natural|union|"
                            r"except|intersect|having|cross)\b)\w+", re.I)  # FROM singer s


def failure_candidates(run_id: str = "design", variant: str = A) -> pd.DataFrame:
    """RQ1 example finder, run on the design slice of train (agent_ext/splits.py) so report examples never come from
    the evaluation set. Among questions the variant gets wrong (exact match), a query is
    over-formatting:    it uses an alias (AS ... or a bare table alias) and reads the same tables as the gold query;
    over-complicating:  it reads every gold table plus at least one more.
    These are candidates for hand-checking, not the report's failure counts."""
    rows = load(variant, run_id)[0]
    sc = scored(rows)
    recs = []
    for r in rows:
        if sc[r["id"]]["exact"]:
            continue
        pred, gold = r["pred"] or "", " ".join(r["gold"].split())
        pt, gt = {t.lower() for t in _TABLE_RE.findall(pred)}, {t.lower() for t in _TABLE_RE.findall(gold)}
        aliased = bool(_COL_ALIAS_RE.search(pred) or _BARE_ALIAS_RE.search(pred))
        kind = "over-formatting" if aliased and pt == gt else "over-complicating" if pt > gt else None
        if kind:
            recs.append({"kind": kind, "train_id": r["id"], "db_id": r["db_id"], "hardness": sc[r["id"]]["hardness"],
                         "question": r["question"], "pred": " ".join(pred.split()), "gold": gold})
    return pd.DataFrame(recs).sort_values(["kind", "train_id"]).reset_index(drop=True)


def summary(run_id: str, models_path: Path | None = None) -> pd.DataFrame:
    """RQ1 summary table: the headline rows of the accuracy, component, whole-query, validity, agreement and
    efficiency tables, A vs B."""
    acc, comp, wq, val = accuracy(run_id), components(run_id), whole_query(run_id), validity(run_id)
    agree, eff = agreement(run_id)["all"], efficiency(run_id, models_path)
    rows = {("Exact match (%)", l.split(" (")[0]): (acc.iloc[0][l], acc.iloc[1][l]) for l in acc.columns}
    rows |= {("Component F1 (%)", c): (comp.loc[c, ("System A", "F1")], comp.loc[c, ("System B", "F1")]) for c in COMPONENTS}
    rows[("Execution", "Execution accuracy (%)")] = tuple(wq.loc["Execution accuracy (%)"])
    rows |= {("Validity", k): tuple(val.loc[k]) for k in val.index}
    rows[("Agreement", "Questions solved only by this system")] = (agree["Solved by A only"], agree["Solved by B only"])
    rows[("Agreement", "Questions solved by both")] = (agree["Solved by both A and B"],) * 2
    rows |= {("Cost per question", k): (eff.loc[k, "System A"], eff.loc[k, "System B"]) for k in eff.index}
    df = pd.DataFrame(rows, index=["System A", "System B"]).T
    df["Change"] = np.where(df.index.get_level_values(0) == "Cost per question", df["System B"] / df["System A"],
                            df["System B"] - df["System A"])
    df.loc[[("Agreement", "Questions solved only by this system"), ("Agreement", "Questions solved by both")], "Change"] = np.nan
    return df.round(2)


# report row names of the ablation variants, in the report's order
VARIANT_LABELS = {"base": "System A: Qwen3-4B alone", "rag": "+Few-Shot RAG", "react": "+ReAct loop, Execute Tool",
                  "react_parse": "+ReAct loop, Execute + Parse Tool", "rag_react": "+Few-Shot RAG + ReAct loop, Execute Tool",
                  "rag_react_parse": "System B: Few-Shot RAG + ReAct loop, Execute + Parse Tool"}


def ablation(run_id: str) -> pd.DataFrame:
    """RQ2 ablation: exact match (%) by difficulty for every variant."""
    out = {}
    for variant in VARIANTS:
        em = load(variant, run_id)[1]["exact matching accuracy"]["exact match"]
        out[VARIANT_LABELS[variant]] = {l: 100 * em[l] for l in LEVELS}
    return pd.DataFrame(out).T.round(1)


def ablation_execution(run_id: str) -> pd.DataFrame:
    """RQ2 ablation: execution accuracy (%) by difficulty for every variant (schema-only databases)."""
    out = {}
    for variant in VARIANTS:
        ex = load(variant, run_id)[1]["execution accuracy"]["execution"]
        out[VARIANT_LABELS[variant]] = {l: 100 * ex[l] for l in LEVELS}
    return pd.DataFrame(out).T.round(1)


def ablation_components(run_id: str) -> pd.DataFrame:
    """RQ2 ablation: component-matching precision (Spider's "accuracy"), recall and F1 (%) for every variant."""
    out = {}
    for variant in VARIANTS:
        m = load(variant, run_id)[1]
        out[VARIANT_LABELS[variant]] = {(c, label): 100 * m[key][k]["all"] for c, k in COMPONENTS.items()
                                        for key, label in (("partial matching accuracy", "P"), ("partial matching recall", "R"),
                                                           ("partial matching f1", "F1"))}
    return pd.DataFrame(out).T.round(1)


# (component, variant = System A + component, comparison) in build-up order; the comparison is System B minus that
# component, and the parse tool is also compared against the execute-only loop
FIXED_BROKE = [("Few-Shot RAG", "rag", "react_parse"), ("ReAct loop", "react", "rag"),
               ("Parse Tool", "react_parse", "rag_react"), ("Parse Tool", "react_parse", "react")]


def fixed_broke(run_id: str) -> pd.DataFrame:
    """RQ2 per-component comparison. Fixed: exact match in the variant but not in the comparison; Broke: the reverse.
    The e/m/h/x columns split each count by difficulty."""
    sc = {v: scored(load(v, run_id)[0]) for v in {v for _, a, b in FIXED_BROKE for v in (a, b)}}
    out = []
    for comp, a, b in FIXED_BROKE:
        ids = sorted(set(sc[a]) & set(sc[b]))
        fixed = [i for i in ids if sc[a][i]["exact"] and not sc[b][i]["exact"]]
        broke = [i for i in ids if sc[b][i]["exact"] and not sc[a][i]["exact"]]
        split = lambda xs: "/".join(str(sum(sc[a][i]["hardness"] == l for i in xs)) for l in LEVELS[:-1])
        out.append({"Component": comp, "Variant": VARIANT_LABELS[a], "Compared to": VARIANT_LABELS[b],
                    "Fixed": len(fixed), "Broke": len(broke), "Net": len(fixed) - len(broke),
                    "Fixed (e/m/h/x)": split(fixed), "Broke (e/m/h/x)": split(broke)})
    return pd.DataFrame(out)


_ERRORS = {"execute": ("ERROR", execute), "spider_parse": ("PARSE ERROR", spider_parse)}


def _norm(sql: str | None) -> str:
    return " ".join((sql or "").lower().replace(";", " ").split())


def trajectory_stats(run_id: str) -> pd.DataFrame:
    """RQ2 descriptive statistics of System B's trajectories, by difficulty and exact match (report Table tab:desc).

    Repair: a trajectory has an error if any execute observation began with ERROR or any spider_parse observation with
    PARSE ERROR; it is repaired if its final query both executes and parses. First draft vs. final (% of trajectories):
    an error that was not repaired puts the trajectory in "Final unparseable"; otherwise the first SQL draft is compared
    with the final query: identical strings, equal up to case/whitespace/semicolons ("Formatting only"), a first draft
    that does not parse under process_sql ("First draft unparseable"), or else a semantic change."""
    rows, _ = load(B, run_id)
    sc = scored(rows)
    cats = ["Identical", "Formatting only", "Semantic change", "First draft unparseable -> final parseable",
            "Final unparseable (not comparable)"]
    recs = []
    for r in rows:
        err = any(t["action"] == tool and (t["observation"] or "").startswith(m) for tool, (m, _) in _ERRORS.items()
                  for t in r["trace"])
        repaired = execute(r["db_id"], r["pred"]).startswith("OK") and _parse(r["db_id"], r["pred"]) is not None
        drafts = [t["sql"] for t in r["trace"] if t["sql"]]
        first = drafts[0] if drafts else None
        if err and not repaired:
            cat = cats[4]
        elif first == r["pred"]:
            cat = cats[0]
        elif first is None or _parse(r["db_id"], first) is None:
            cat = cats[3]
        else:
            cat = cats[1] if _norm(first) == _norm(r["pred"]) else cats[2]
        recs.append({"level": sc[r["id"]]["hardness"], "correct": bool(sc[r["id"]]["exact"]), "err": err,
                     "repaired": err and repaired, "cat": cat})
    df = pd.DataFrame(recs)
    out = {}
    for l in LEVELS:
        for ok in (True, False):
            g = df[(df.correct == ok) & ((df.level == l) if l != "all" else True)]
            n_err, n_rep = int(g.err.sum()), int(g.repaired.sum())
            col = {"n trajectories": len(g), "with an error, n": n_err,
                   "repaired, % (k/n)": f"{100 * n_rep / n_err:.0f} ({n_rep}/{n_err})" if n_err else "-"}
            col |= {f"{c} (%)": round(100 * (g.cat == c).mean()) if len(g) else "-" for c in cats}
            out[(l, "Correct" if ok else "Incorrect")] = col
    return pd.DataFrame(out)


def repair_success(run_id: str) -> pd.DataFrame:
    """RQ2 feature 1. A problem is a (question, tool) pair where some observation of that tool began with its error
    marker. It is persistent if re-running the tool on the final submitted query still gives the marker."""
    out = {}
    for variant in REACT_VARIANTS:
        for r in load(variant, run_id)[0]:
            for tool, (marker, fn) in _ERRORS.items():
                if any(t["action"] == tool and (t["observation"] or "").startswith(marker) for t in r["trace"]):
                    d = out.setdefault((variant, tool), [0, 0])
                    d[0] += 1
                    d[1] += fn(r["db_id"], r["pred"]).startswith(marker)
    df = pd.DataFrame([{"variant": a, "tool": t, "problems flagged": n, "persistent": p, "repair success rate (%)": 100 * (1 - p / n)}
                       for (a, t), (n, p) in out.items()])
    return df.round(1)


def semantic_distance(run_id: str, models_path: Path | None = None) -> pd.DataFrame:
    """RQ2 feature 2: 1 - cos(first, last SQL-parsable draft) with all-MiniLM-L6-v2, split by exact match of the final query."""
    from .retriever import Retriever
    emb = Retriever(5, models_path, models_path is not None)
    recs = []
    for variant in REACT_VARIANTS:
        rows, _ = load(variant, run_id)
        sc = scored(rows)
        pairs = []
        for r in rows:
            drafts = [t["sql"] for t in r["trace"] if t["sql"] and _parse(r["db_id"], t["sql"]) is not None]
            pairs.append((drafts[0], drafts[-1]) if drafts else None)
        have = [p for p in pairs if p]
        e = emb.embed([x for p in have for x in p]).reshape(len(have), 2, -1)
        dist = iter(1 - (e[:, 0] * e[:, 1]).sum(1))
        for r, p in zip(rows, pairs):
            if p:
                recs.append({"variant": variant, "solved": bool(sc[r["id"]]["exact"]), "distance": float(max(next(dist), 0.0))})
    df = pd.DataFrame(recs)
    g = df.groupby(["variant", "solved"])["distance"]
    return pd.DataFrame({"n": g.size(), "mean": g.mean(), "median": g.median(), "% unchanged (d=0)": g.apply(lambda x: 100 * (x < 1e-6).mean())}).round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default="full")
    ap.add_argument("--models-path", type=Path, default=Path(__file__).resolve().parent.parent / "a1_lib/models")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    for title, df in [("RQ1 accuracy (exact match %)", accuracy(args.run_id)), ("RQ1 output validity", validity(args.run_id)),
                      ("RQ1 component matching (%)", components(args.run_id)), ("RQ1 whole-query metrics", whole_query(args.run_id)),
                      ("RQ1 efficiency", efficiency(args.run_id, args.models_path)), ("RQ1 agreement", agreement(args.run_id)),
                      ("RQ2 ablation (exact match %)", ablation(args.run_id)), ("RQ2 repair success", repair_success(args.run_id)),
                      ("RQ2 semantic distance", semantic_distance(args.run_id, args.models_path))]:
        print(f"\n## {title}\n{df.to_string()}")


if __name__ == "__main__":
    main()
