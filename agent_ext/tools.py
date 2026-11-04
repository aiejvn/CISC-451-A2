"""Actions the ReAct agent can take. Each tool maps (db_id, sql) -> observation string."""
import sqlite3
import time

from eval.evalpipeline.common import DB_DIR
from process_sql import Schema, get_schema, get_sql  # a1_lib/spider, on sys.path via agent_ext/__init__

EXEC_TIMEOUT_S = 5.0


def db_path(db_id: str) -> str:
    return str(DB_DIR / db_id / f"{db_id}.sqlite")


def execute(db_id: str, sql: str) -> str:
    """Run the query read-only. The local databases are schema-only, so this only reports errors and columns."""
    conn = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
    deadline = time.monotonic() + EXEC_TIMEOUT_S
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or []]
        n = len(cur.fetchmany(1000))
        return f"OK: columns [{', '.join(cols)}]; {n} result rows (all tables are empty: the database has the schema only)"
    except sqlite3.Error as e:
        return f"ERROR: {e}"
    finally:
        conn.close()


_schemas: dict[str, Schema] = {}


def spider_parse(db_id: str, sql: str) -> str:
    """Parse with Spider's process_sql, exactly as evaluation.py does before scoring."""
    if db_id not in _schemas:
        _schemas[db_id] = Schema(get_schema(db_path(db_id)))
    try:
        get_sql(_schemas[db_id], sql)
        return "PARSES: the grader can read this query"
    except Exception as e:  # process_sql signals failure with assorted exception types
        return f"PARSE ERROR: the grader cannot read this query ({type(e).__name__}: {e})"


TOOLS = {"execute": execute, "spider_parse": spider_parse}

DESCRIPTIONS = {
    "execute": "runs the query on the database and reports the result columns, or the SQLite error. "
               "The database has the schema but no rows, so results are always empty.",
    "spider_parse": "checks whether the benchmark's SQL grader can parse the query, and reports the parser error "
                    "if it cannot.",
}
