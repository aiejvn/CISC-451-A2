"""Download every model in eval/configs/models.yaml into models/<author>/<name> (run on a node with internet).

Slurm compute nodes are offline; jobs then load weights from this local tree (see common.resolve_local_path).

    python download_models.py                          # all models in models.yaml
    python download_models.py --models Qwen/Qwen3-4B   # just these
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval.evalpipeline.common import MODELS_DIR, load_model_configs  # noqa: E402

PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", help="HF repo ids from models.yaml (default: all)")
    ap.add_argument("--dest", type=Path, default=MODELS_DIR, help=f"target tree (default: {MODELS_DIR})")
    args = ap.parse_args()

    for repo_id in args.models or list(load_model_configs()):
        dest = args.dest / repo_id
        if (dest / "config.json").is_file():
            print(f"{repo_id}: already at {dest}")
            continue
        print(f"{repo_id}: downloading -> {dest}", flush=True)
        snapshot_download(repo_id, local_dir=dest, allow_patterns=PATTERNS)


if __name__ == "__main__":
    main()
