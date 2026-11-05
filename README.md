# CISC 451 A2: RAG + ReAct agent for Spider text-to-SQL

## Layout

| path | what |
|---|---|
| `a1_lib/` | Assignment 1 pipeline (model download, generation, Spider evaluator). Imported by `agent_ext/`, never modified. |
| `agent_ext/` | A2 code: `run.py` (one variant), `compare.py` (paired tests), `report_tables.py` (report numbers), `plot_trajectories.py`, `final_exec.py`, `traces.py` |
| `RQ1_RQ2.ipynb` | Reproduces every table and figure in the report from a live run |
| `agent_ext/results/<variant>/<run_id>/` | Outputs of each run (git-ignored) |

## Setup

**Requirements**

- Python 3.10+ (developed on 3.12) and `git`.
- An NVIDIA GPU is strongly recommended. Qwen3-4B in bf16 needs about 8 GB for weights; the ReAct variants at the default
  `--batch-size 64` need far more for long contexts. On a smaller GPU lower `--batch-size` (e.g. 1 or 4); the runner also halves the batch automatically on CUDA out-of-memory. CPU works only for tiny `--limit` checks and is very slow.
- About 10 GB of disk for the two models, and internet access once, to download them.

**1. Environment**

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch                                        
pip install -r a1_lib/eval/requirements.txt
pip install ipykernel                                     # only for the notebook
python -c "import nltk; nltk.download('punkt_tab', download_dir='nltk_data')"
```

`bitsandbytes` (in the requirements) is only used for quantized A1 runs and may fail to install on macOS/Windows; if so,
remove that line from the requirements file, since the A2 experiments do not use it.

**2. Databases and models (needs internet)**

```bash
(cd a1_lib && python build_spider_dbs.py)                 # schema-only SQLite databases -> a1_lib/spider/database/
python a1_lib/download_models.py                          # Qwen3-4B -> a1_lib/models/Qwen/Qwen3-4B
python a1_lib/download_models.py --models sentence-transformers/all-MiniLM-L6-v2   # retriever
```

**3. Run the notebook.** Register the kernel, open `RQ1_RQ2.ipynb`cand run all cells.

`RQ1_RQ2.ipynb` reproduces all data used in the report end to end. Select the kernel created by:

```bash
source .venv/bin/activate && python -m ipykernel install --user --name cisc451 --display-name "CISC451 (.venv)"
```

Set `RUN_ID` to a fresh name (default `repro`) so the original `full` results are not overwritten, and `LIMIT` to a small
number for a quick check. To only regenerate tables and figures from existing results, set `RUN_ID = "full"` and skip
the run cell (the second code cell).