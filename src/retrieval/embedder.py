from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TextEmbedder(Protocol):
    name: str
    dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@dataclass
class HashingEmbedder:
    dim: int = 384
    name: str = "hashing_fallback"

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype="float32")
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + math.log1p(len(token)))
        return vector

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(np.vstack([self._embed_one(text) for text in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed.") from exc

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class DPRQuestionEmbedder:
    def __init__(self, model_name: str = "facebook/dpr-question_encoder-single-nq-base") -> None:
        try:
            import torch
            from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for DPR retrieval.") from exc

        self.name = model_name
        self.dim = 768
        self._torch = torch
        self.tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_name)
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.model.eval()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_query(text) for text in texts]).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with self._torch.no_grad():
            vector = self.model(**inputs).pooler_output[0].detach().cpu().numpy()
        return vector.astype("float32")


def get_sentence_transformer_or_hashing() -> TextEmbedder:
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return HashingEmbedder()

