"""Batched single-shot generation and a batched ReAct loop (Thought -> Action -> Observation).

A batch of questions runs the loop in lockstep: each step generates one turn for every question that has not
finished yet, then runs their tools.
"""
import re

from eval.evalpipeline.adapters import FALLBACK_SQL, extract_sql

from .prompts import react_messages
from .tools import TOOLS

ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[?(.*)", re.S)  # brackets optional: the model sometimes drops them
# a step is over once the model closes its Action (fenced query, then "]") or starts inventing an Observation
ACTION_DONE_RE = re.compile(r"Action:\s*\w+\s*\[\s*```.*?```\s*\]", re.S)


class StopOnText:
    """Per-sequence stop once the new text completes an Action or starts an Observation.

    Only the last few tokens are decoded each step; the full text is decoded when they contain "]".
    """

    def __init__(self, tokenizer, prompt_len: int):
        self.tok, self.prompt_len = tokenizer, prompt_len

    def __call__(self, input_ids, scores=None, **kw):
        import torch
        stop = []
        for row in input_ids[:, self.prompt_len :]:
            tail = self.tok.decode(row[-6:], skip_special_tokens=True)
            if "Observation" in tail:
                stop.append(True)
            elif "]" in tail:
                stop.append(bool(ACTION_DONE_RE.search(self.tok.decode(row, skip_special_tokens=True))))
            else:
                stop.append(False)
        return torch.tensor(stop, device=input_ids.device)


class Generator:
    def __init__(self, model, tokenizer, adapter):
        self.model, self.tok, self.adapter = model, tokenizer, adapter
        self.gen_kwargs = adapter.generation_kwargs()

    def __call__(self, batch: list[list[dict]], max_new_tokens: int, stop_on_action: bool = False) -> list[tuple[str, int]]:
        """One generation per message list: [(decoded new text, number of new tokens)]. Halves the batch on OOM."""
        import torch
        from transformers import StoppingCriteriaList

        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                              **self.adapter.chat_template_kwargs()) for m in batch]
        inputs = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        n_prompt = inputs["input_ids"].shape[1]
        extra = {"stopping_criteria": StoppingCriteriaList([StopOnText(self.tok, n_prompt)])} if stop_on_action else {}
        try:
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                          **extra, **self.gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            del inputs
            torch.cuda.empty_cache()
            half = len(batch) // 2
            print(f"OOM at batch {len(batch)}; retrying as {half} + {len(batch) - half}", flush=True)
            return (self(batch[:half], max_new_tokens, stop_on_action)
                    + self(batch[half:], max_new_tokens, stop_on_action))
        new = out[:, n_prompt:]
        res = []
        for row in new:
            text = self.tok.decode(row, skip_special_tokens=self.adapter.skip_special_tokens)
            if not self.adapter.skip_special_tokens and self.tok.pad_token:
                text = text.replace(self.tok.pad_token, "")
            res.append((text, int((row != self.tok.pad_token_id).sum())))
        return res


def single_shot(gen: Generator, prompts: list[str], max_new_tokens: int) -> list[dict]:
    outs = gen([[{"role": "user", "content": p}] for p in prompts], max_new_tokens)
    return [{"raw": raw, "pred": gen.adapter.postprocess(raw)[1], "new_tokens": n} for raw, n in outs]


class ReActAgent:
    def __init__(self, gen: Generator, tools: list[str], max_steps: int = 4, max_new_tokens_per_step: int = 1024):
        self.gen, self.tools, self.max_steps, self.step_tokens = gen, tools, max_steps, max_new_tokens_per_step

    def _observe(self, db_id: str, tool, sql, matched: bool) -> str:
        if tool in self.tools and sql:
            return TOOLS[tool](db_id, sql)
        if not matched:
            return "No Action found. End every step with: Action: <tool>[```sql\n<query>\n```]"
        if not sql:
            return "The Action contains no SQL query."
        return f"Unknown tool {tool!r}. Available: {', '.join(self.tools + ['finish'])}."

    def run(self, items: list[tuple[str, str, list[dict] | None]]) -> list[dict]:
        """items: (question, db_id, few-shot examples or None). Returns one result per item, in order."""
        states = [{"messages": react_messages(q, db, self.tools, self.max_steps, shots), "db_id": db, "trace": [],
                   "last_sql": None, "finished": False, "new_tokens": 0} for q, db, shots in items]
        for step in range(1, self.max_steps + 1):
            active = [s for s in states if not s["finished"]]
            if not active:
                break
            outs = self.gen([s["messages"] for s in active], self.step_tokens, stop_on_action=True)
            for s, (raw, n) in zip(active, outs):
                s["new_tokens"] += n
                _, text = self.gen.adapter.split_thinking(raw)
                text = text.split("Observation:")[0].strip()  # drop any observation the model invented
                m = ACTION_RE.search(text)
                tool, sql = (m.group(1), extract_sql(m.group(2))) if m else (None, None)
                if sql == FALLBACK_SQL:
                    sql = None
                s["last_sql"] = sql or s["last_sql"]
                if tool == "finish" and sql:
                    s["trace"].append({"step": step, "output": text, "action": tool, "sql": sql, "observation": None})
                    s["finished"] = True
                    continue
                obs = self._observe(s["db_id"], tool, sql, m is not None)
                if step == self.max_steps - 1:
                    obs += "\nYou have one step left: use finish."
                s["trace"].append({"step": step, "output": text, "action": tool, "sql": sql, "observation": obs})
                s["messages"] = s["messages"] + [{"role": "assistant", "content": text},
                                                 {"role": "user", "content": f"Observation: {obs}"}]
        return [{"pred": s["last_sql"] or FALLBACK_SQL, "trace": s["trace"], "steps": len(s["trace"]),
                 "finished": s["finished"], "new_tokens": s["new_tokens"]} for s in states]
