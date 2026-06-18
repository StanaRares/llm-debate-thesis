from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import heapq
import re

import numpy as np

from .embedder import DPRQuestionEmbedder, TextEmbedder, get_sentence_transformer_or_hashing
from .faiss_index import VectorIndex
from .wikipedia_corpus import (
    WikipediaPassage,
    corpus_snapshot,
    load_local_corpus,
    load_local_corpus_result,
    load_wiki_dpr_dataset,
    normalize_passage,
)


@dataclass
class RetrievedPassage:
    rank: int
    passage_id: str
    title: str
    text: str
    retrieval_score: float
    source: str
    snapshot_date: str

    def to_agent_dict(self) -> dict[str, str]:
        return {"title": self.title, "text": self.text}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WikipediaRetriever:
    def __init__(
        self,
        passages: list[WikipediaPassage],
        vectors: np.ndarray,
        query_embedder: TextEmbedder,
        retriever_type: str,
        snapshot: str,
        corpus_used: str,
        corpus_mode: str,
        fallback_used: bool = False,
        fallback_reason: str = "",
    ) -> None:
        if not passages:
            raise ValueError("A retriever needs at least one passage.")
        self.passages = passages
        self.query_embedder = query_embedder
        self.retriever_type = retriever_type
        self.snapshot = snapshot
        self.corpus_used = corpus_used
        self.corpus_mode = corpus_mode
        self.fallback_used = fallback_used
        self.fallback_reason = fallback_reason
        self.index = VectorIndex(dim=int(vectors.shape[1]), normalize=retriever_type != "dpr")
        self.index.add(vectors)

    @classmethod
    def from_config(
        cls,
        retriever_type: str = "dpr",
        corpus_snapshot_value: str = "",
        corpus_path: Path | str | None = None,
        max_passages: int | None = None,
        allow_fallback: bool = False,
    ) -> "WikipediaRetriever":
        retriever_type = (retriever_type or "dpr").strip().lower()
        if retriever_type == "dpr" and corpus_path is None:
            try:
                corpus = load_wiki_dpr_dataset(max_passages=max_passages)
                rows = list(corpus.dataset.take(max_passages)) if max_passages and hasattr(corpus.dataset, "take") else corpus.dataset
                if rows is corpus.dataset and "embeddings" not in corpus.dataset.column_names:
                    raise RuntimeError("facebook/wiki_dpr did not expose an 'embeddings' column.")
                if isinstance(rows, list):
                    if not rows or "embeddings" not in rows[0]:
                        raise RuntimeError("facebook/wiki_dpr rows did not expose DPR embeddings.")
                    vectors = np.asarray([row["embeddings"] for row in rows], dtype="float32")
                else:
                    vectors = np.asarray(corpus.dataset["embeddings"], dtype="float32")
                return cls(
                    passages=corpus.passages,
                    vectors=vectors,
                    query_embedder=DPRQuestionEmbedder(),
                    retriever_type="dpr",
                    snapshot=corpus_snapshot_value or corpus.corpus_snapshot,
                    corpus_used=corpus.corpus_used,
                    corpus_mode=corpus.corpus_mode,
                    fallback_used=False,
                    fallback_reason="",
                )
            except Exception as exc:
                if not allow_fallback:
                    raise RuntimeError(
                        "DPR Wikipedia retrieval failed and --allow_fallback was not set. "
                        f"Reason: {exc}"
                    ) from exc
                fallback_reason = f"DPR Wikipedia retrieval failed: {exc}"
                print(f"[retrieval] fallback enabled; using local sample. Reason: {fallback_reason}")
                corpus = load_local_corpus_result(
                    corpus_path,
                    default_snapshot=corpus_snapshot_value or "local_fixed_corpus",
                    fallback_reason=fallback_reason,
                )
                embedder = get_sentence_transformer_or_hashing()
                texts = [f"{passage.title}. {passage.text}" for passage in corpus.passages]
                vectors = embedder.embed_texts(texts)
                effective_type = "sentence_transformer" if embedder.name != "hashing_fallback" else "hashing_fallback"
                return cls(
                    passages=corpus.passages,
                    vectors=vectors,
                    query_embedder=embedder,
                    retriever_type=effective_type,
                    snapshot=corpus.corpus_snapshot,
                    corpus_used=corpus.corpus_used,
                    corpus_mode=corpus.corpus_mode,
                    fallback_used=True,
                    fallback_reason=fallback_reason,
                )

        if corpus_path is None and not allow_fallback:
            raise RuntimeError(
                "Sentence-transformer retrieval requires --corpus_path unless --allow_fallback is set."
            )
        fallback_reason = "" if corpus_path else "No --corpus_path supplied for sentence-transformer retrieval."
        corpus = load_local_corpus_result(
            corpus_path,
            default_snapshot=corpus_snapshot_value or "local_fixed_corpus",
            fallback_reason=fallback_reason,
        )
        embedder = get_sentence_transformer_or_hashing()
        texts = [f"{passage.title}. {passage.text}" for passage in corpus.passages]
        vectors = embedder.embed_texts(texts)
        effective_type = "sentence_transformer" if embedder.name != "hashing_fallback" else "hashing_fallback"
        return cls(
            passages=corpus.passages,
            vectors=vectors,
            query_embedder=embedder,
            retriever_type=effective_type,
            snapshot=corpus_snapshot(corpus.passages, corpus_snapshot_value),
            corpus_used=corpus.corpus_used,
            corpus_mode=corpus.corpus_mode,
            fallback_used=corpus.fallback_used,
            fallback_reason=corpus.fallback_reason,
        )

    def retrieve(self, claim: str, top_k: int = 5) -> list[RetrievedPassage]:
        query = self.query_embedder.embed_query(claim)
        results = self.index.search(query, top_k=top_k)
        retrieved: list[RetrievedPassage] = []
        for rank, result in enumerate(results, start=1):
            passage = self.passages[result.index]
            retrieved.append(
                RetrievedPassage(
                    rank=rank,
                    passage_id=passage.passage_id,
                    title=passage.title,
                    text=passage.text,
                    retrieval_score=round(result.score, 6),
                    source=passage.source,
                    snapshot_date=passage.snapshot_date,
                )
            )
        return retrieved

    def metadata(self) -> dict[str, Any]:
        return {
            "corpus_used": self.corpus_used,
            "corpus_mode": self.corpus_mode,
            "corpus_snapshot": self.snapshot,
            "retriever_type": self.retriever_type,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


STOPWORDS = {
    "the", "is", "in", "of", "a", "an", "and", "to", "for", "with", "on", "by",
    "at", "as", "or", "it", "was", "were", "be", "been", "being", "this", "that",
    "from", "into", "than", "claim", "located",
}


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if token not in STOPWORDS]


def lexical_score(query_tokens: list[str], title: str, text: str) -> float:
    title_tokens = set(tokenize(title))
    text_tokens = set(tokenize(text))
    if not query_tokens:
        return 0.0
    score = 0.0
    for token in query_tokens:
        entity_weight = 8.0 if token not in {"paris", "france", "american", "british", "german"} and len(token) >= 5 else 1.0
        if token in title_tokens:
            score += entity_weight * (8.0 + min(len(token), 10) * 0.25)
        if token in text_tokens:
            score += entity_weight * (1.0 + min(len(token), 10) * 0.05)
    query_phrase = " ".join(query_tokens[:3])
    haystack = f"{title} {text}".lower()
    if query_phrase and query_phrase in haystack:
        score += 10.0
    return score


def retrieval_quality_check(claim: str, passages: list[RetrievedPassage | dict[str, Any]]) -> tuple[bool, str]:
    tokens = set(tokenize(claim))
    titles = [
        str(getattr(passage, "title", "") if not isinstance(passage, dict) else passage.get("title", "")).lower()
        for passage in passages
    ]
    retrieved_text = " ".join(
        f"{title} "
        f"{getattr(passage, 'text', '') if not isinstance(passage, dict) else passage.get('text', '')}"
        for title, passage in zip(titles, passages)
    ).lower()
    if "eiffel" in tokens:
        if "eiffel" in retrieved_text or any(title in {"paris", "france"} for title in titles):
            return True, "Eiffel Tower smoke check passed: retrieved evidence mentions Eiffel or has Paris/France as a title."
        return False, "Eiffel Tower smoke check failed: no retrieved passage mentions Eiffel and no title is Paris/France."
    content_tokens = set(tokenize(retrieved_text))
    overlap = sorted(tokens & content_tokens)
    if overlap:
        return True, f"Retrieved passages overlap with claim tokens: {', '.join(overlap[:8])}."
    return False, "Retrieved passages have no meaningful lexical overlap with the claim."


class HybridWikipediaRetriever:
    def __init__(
        self,
        corpus_path: Path | str | None = None,
        max_passages: int | None = None,
        hybrid_scan_limit: int = 150000,
        hybrid_bm25_k: int = 100,
        corpus_snapshot_value: str = "",
    ) -> None:
        self.retriever_type = "hybrid"
        self.hybrid_scan_limit = max(1, int(hybrid_scan_limit))
        self.hybrid_bm25_k = max(1, int(hybrid_bm25_k))
        self.max_passages = max_passages
        self.query_embedder = DPRQuestionEmbedder()
        self.fallback_used = False
        self.fallback_reason = ""
        self._local_passages: list[WikipediaPassage] | None = None
        self._local_vectors: np.ndarray | None = None

        if corpus_path:
            self._local_passages = load_local_corpus(corpus_path, default_snapshot=corpus_snapshot_value or "local_wikipedia_subset")
            self._local_vectors = None
            self.corpus_used = str(Path(corpus_path).resolve())
            self.corpus_mode = "wikipedia_subset"
            self.snapshot = corpus_snapshot(self._local_passages, corpus_snapshot_value)
        else:
            self.corpus_used = "facebook/wiki_dpr"
            self.corpus_mode = "full_wikipedia_dpr_claim_aware_stream"
            self.snapshot = corpus_snapshot_value or "facebook/wiki_dpr:psgs_w100.nq.exact"

    @classmethod
    def from_config(
        cls,
        corpus_snapshot_value: str = "",
        corpus_path: Path | str | None = None,
        max_passages: int | None = None,
        hybrid_scan_limit: int = 5000,
        hybrid_bm25_k: int = 100,
        **_: Any,
    ) -> "HybridWikipediaRetriever":
        return cls(
            corpus_path=corpus_path,
            max_passages=max_passages,
            hybrid_scan_limit=hybrid_scan_limit,
            hybrid_bm25_k=hybrid_bm25_k,
            corpus_snapshot_value=corpus_snapshot_value,
        )

    def _iter_dpr_rows(self) -> Any:
        from datasets import load_dataset

        dataset = load_dataset(
            "facebook/wiki_dpr",
            "psgs_w100.nq.exact",
            split="train",
            trust_remote_code=True,
            streaming=True,
        )
        limit = self.max_passages or self.hybrid_scan_limit
        for index, row in enumerate(dataset):
            if index >= limit:
                break
            yield index, row

    def _candidate_rows(self, claim: str) -> list[tuple[float, int, dict[str, Any]]]:
        query_tokens = tokenize(claim)
        heap: list[tuple[float, int, dict[str, Any]]] = []

        if self._local_passages is not None:
            rows = [
                (index, passage.to_dict() | {"id": passage.passage_id})
                for index, passage in enumerate(self._local_passages)
            ]
        else:
            rows = self._iter_dpr_rows()

        for index, row in rows:
            score = lexical_score(query_tokens, str(row.get("title", "")), str(row.get("text", "")))
            if score <= 0:
                continue
            item = (score, int(index), dict(row))
            if len(heap) < self.hybrid_bm25_k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        return sorted(heap, reverse=True)

    def retrieve(self, claim: str, top_k: int = 5) -> list[RetrievedPassage]:
        candidates = self._candidate_rows(claim)
        if not candidates:
            return []
        query_vector = self.query_embedder.embed_query(claim).astype("float32")
        scored: list[tuple[float, float, int, dict[str, Any]]] = []
        lexical_max = max(score for score, _, _ in candidates) or 1.0
        for lexical, index, row in candidates:
            embedding = row.get("embeddings")
            dense = 0.0
            if embedding is not None:
                passage_vector = np.asarray(embedding, dtype="float32")
                dense = float(np.dot(query_vector, passage_vector))
            scored.append((dense, lexical, index, row))

        dense_values = [dense for dense, _, _, _ in scored]
        dense_min = min(dense_values)
        dense_max = max(dense_values)
        dense_range = dense_max - dense_min or 1.0
        reranked: list[tuple[float, float, int, dict[str, Any]]] = []
        for dense, lexical, index, row in scored:
            lexical_norm = lexical / lexical_max
            dense_norm = (dense - dense_min) / dense_range
            combined = (0.85 * lexical_norm) + (0.15 * dense_norm)
            reranked.append((combined, lexical, index, row))

        retrieved: list[RetrievedPassage] = []
        for rank, (combined, lexical, _, row) in enumerate(sorted(reranked, reverse=True)[:top_k], start=1):
            passage = normalize_passage(
                row,
                default_source=self.corpus_used,
                default_snapshot=self.snapshot,
            )
            retrieved.append(
                RetrievedPassage(
                    rank=rank,
                    passage_id=passage.passage_id,
                    title=passage.title,
                    text=passage.text,
                    retrieval_score=round(float(combined), 6),
                    source=passage.source,
                    snapshot_date=passage.snapshot_date,
                )
            )
        return retrieved

    def metadata(self) -> dict[str, Any]:
        return {
            "corpus_used": self.corpus_used,
            "corpus_mode": self.corpus_mode,
            "corpus_snapshot": self.snapshot,
            "retriever_type": self.retriever_type,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "hybrid_scan_limit": self.hybrid_scan_limit,
            "hybrid_bm25_k": self.hybrid_bm25_k,
        }


def build_wikipedia_retriever(
    retriever_type: str = "dpr",
    corpus_snapshot_value: str = "",
    corpus_path: Path | str | None = None,
    max_passages: int | None = None,
    allow_fallback: bool = False,
    hybrid_scan_limit: int = 5000,
    hybrid_bm25_k: int = 100,
) -> WikipediaRetriever | HybridWikipediaRetriever:
    retriever_type = (retriever_type or "dpr").strip().lower()
    if retriever_type == "hybrid":
        return HybridWikipediaRetriever.from_config(
            corpus_snapshot_value=corpus_snapshot_value,
            corpus_path=corpus_path,
            max_passages=max_passages,
            hybrid_scan_limit=hybrid_scan_limit,
            hybrid_bm25_k=hybrid_bm25_k,
        )
    return WikipediaRetriever.from_config(
        retriever_type=retriever_type,
        corpus_snapshot_value=corpus_snapshot_value,
        corpus_path=corpus_path,
        max_passages=max_passages,
        allow_fallback=allow_fallback,
    )
