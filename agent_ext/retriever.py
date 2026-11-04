"""k nearest Spider train examples to a question, by cosine similarity of MiniLM sentence embeddings.

Train and dev share no databases, so the retrieved examples never contain a dev answer.
"""
import hashlib
from pathlib import Path

import numpy as np

from . import CACHE_DIR, EMBEDDER
from eval.evalpipeline.common import TRAIN_JSON, load_train, resolve_local_path


class Retriever:
    def __init__(self, k: int = 5, models_path: Path | None = None, local_only: bool = False, batch_size: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = resolve_local_path(EMBEDDER, models_path, local_only)
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        # tiny (22M params): shares the GPU with the LLM when there is one; the tokenizer always runs on CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval().to(self.device)
        self.torch, self.k, self.batch_size = torch, k, batch_size
        self.train = load_train()
        self.train_emb = self._train_embeddings()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled, L2-normalised embeddings (the all-MiniLM-L6-v2 recipe), so a dot product is the cosine."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i : i + self.batch_size], padding=True, truncation=True, max_length=128,
                               return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    def _train_embeddings(self) -> np.ndarray:
        digest = hashlib.sha1(TRAIN_JSON.read_bytes() + EMBEDDER.encode()).hexdigest()[:12]
        cache = CACHE_DIR / f"train_emb_{digest}.npy"
        if cache.exists():
            return np.load(cache)
        emb = self.embed([ex["question"] for ex in self.train])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        return emb

    def topk(self, questions: list[str]) -> list[list[dict]]:
        """For each question: the k most similar train examples, most similar first, with their cosine score."""
        sims = self.embed(questions) @ self.train_emb.T
        idx = np.argsort(-sims, axis=1)[:, : self.k]
        return [[{"train_id": int(j), "score": round(float(s[j]), 4), "question": self.train[j]["question"],
                  "query": " ".join(self.train[j]["query"].split()), "db_id": self.train[j]["db_id"]}
                 for j in row] for row, s in zip(idx, sims)]
