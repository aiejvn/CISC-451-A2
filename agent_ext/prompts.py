"""Prompts. Single-shot arms reuse the A1 template unchanged, so `base` reproduces the A1 run."""
from eval.evalpipeline.common import build_user_prompt, schema_text

from .tools import DESCRIPTIONS


def format_fewshot(examples: list[dict]) -> str:
    pairs = "\n\n".join(f"Question: {ex['question']}\nSQL: {ex['query']}" for ex in examples)
    return f"Here are example questions with their SQL queries, from other databases:\n\n{pairs}\n\n"


def single_shot_prompt(question: str, db_id: str, fewshot: list[dict] | None = None) -> str:
    return (format_fewshot(fewshot) if fewshot else "") + build_user_prompt(question, db_id)


REACT_SYSTEM = """You answer questions about a SQLite database by writing one SQL query. Work in steps. Every step has exactly this form:

Thought: <your reasoning about what to do next>
Action: <tool>[```sql
<query>
```]

Tools:
{tools}
- finish: submit the query as your final answer. This ends the task.

After each action other than finish you will receive an Observation with the result. Write one Thought and one Action per step, then stop and wait for the Observation. You have at most {max_steps} steps."""


def react_messages(question: str, db_id: str, tools: list[str], max_steps: int,
                   fewshot: list[dict] | None = None) -> list[dict]:
    system = REACT_SYSTEM.format(tools="\n".join(f"- {t}: {DESCRIPTIONS[t]}" for t in tools), max_steps=max_steps)
    user = ((format_fewshot(fewshot) if fewshot else "")
            + f"Database schema:\n\n{schema_text(db_id)}\n\nQuestion: {question}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
