"""Per-example Spider exact match, computed with spider/evaluation.py's own functions (matching metrics.json)."""
import copy
import re

from . import RESULTS_DIR  # noqa: F401  (sets up sys.path for the spider imports below)
from .tools import db_path
from eval.evalpipeline.common import TABLES_JSON
import evaluation as ev  # a1_lib/spider/evaluation.py
from process_sql import Schema, get_schema, get_sql

SETOP_RE = re.compile(r"\b(intersect|union|except)\b", re.I)
EMPTY_SQL = {"except": None, "from": {"conds": [], "table_units": []}, "groupBy": [], "having": [],
             "intersect": None, "limit": None, "orderBy": [], "select": [False, []], "union": None, "where": []}

_kmaps = ev.build_foreign_key_map_from_json(str(TABLES_JSON))
_evaluator = ev.Evaluator()
_schemas: dict[str, Schema] = {}


def _schema(db_id: str) -> Schema:
    if db_id not in _schemas:
        _schemas[db_id] = Schema(get_schema(db_path(db_id)))
    return _schemas[db_id]


def _parse(db_id: str, sql: str):
    try:
        return get_sql(_schema(db_id), sql)
    except Exception:
        return None


def score_row(r: dict) -> dict:
    """Mirrors the per-example body of evaluation.evaluate (etype=match), plus habit flags."""
    schema, kmap = _schema(r["db_id"]), _kmaps[r["db_id"]]
    gold = " ".join(r["gold"].split())
    g_sql = get_sql(schema, gold)
    p_raw = _parse(r["db_id"], r["pred"])
    p_sql = copy.deepcopy(EMPTY_SQL) if p_raw is None else p_raw
    hardness = _evaluator.eval_hardness(g_sql)
    n_gold_cols, n_pred_cols = len(g_sql["select"][1]), (len(p_raw["select"][1]) if p_raw else None)
    g = ev.rebuild_sql_col(ev.build_valid_col_units(g_sql["from"]["table_units"], schema), ev.rebuild_sql_val(g_sql), kmap)
    p = ev.rebuild_sql_col(ev.build_valid_col_units(p_sql["from"]["table_units"], schema), ev.rebuild_sql_val(p_sql), kmap)
    return {"id": r["id"], "hardness": hardness, "exact": int(_evaluator.eval_exact_match(p, g)),
            "parses": p_raw is not None, "gold_setop": bool(SETOP_RE.search(gold)),
            "pred_setop": bool(SETOP_RE.search(r["pred"])),
            "extra_cols": n_pred_cols is not None and n_pred_cols > n_gold_cols}
