"""RAG + ReAct extension for Spider text-to-SQL, built on the A1 pipeline in a1_lib/ (imported, never modified).

    python -m agent_ext.run --variant rag_react --local-only --models-path a1_lib/models
    python -m agent_ext.compare
"""
import os
import sys
from pathlib import Path

EXT_ROOT = Path(__file__).resolve().parent
REPO = EXT_ROOT.parent
A1_LIB = REPO / "a1_lib"

# put a1_lib on sys.path and point NLTK at the repo's nltk_data, so `python -m agent_ext...` runs from the repo root.
if str(A1_LIB) not in sys.path:
    sys.path.insert(0, str(A1_LIB))
os.environ.setdefault("NLTK_DATA", str(REPO / "nltk_data"))

from eval.evalpipeline.common import SPIDER_DIR  # noqa: E402

# spider/evaluation.py imports process_sql as a top-level module
if str(SPIDER_DIR) not in sys.path:
    sys.path.insert(0, str(SPIDER_DIR))

RESULTS_DIR = EXT_ROOT / "results"
CACHE_DIR = EXT_ROOT / "cache"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
