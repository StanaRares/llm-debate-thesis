from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_CORPUS_PATH = PROJECT_ROOT / "data" / "wikipedia_corpus_sample.jsonl"


@dataclass(frozen=True)
class WikipediaPassage:
    passage_id: str
    title: str
    text: str
    source: str
    snapshot_date: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusLoadResult:
    passages: list[WikipediaPassage]
    dataset: Any
    corpus_used: str
    corpus_mode: str
    corpus_snapshot: str
    fallback_used: bool = False
    fallback_reason: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "corpus_used": self.corpus_used,
            "corpus_mode": self.corpus_mode,
            "corpus_snapshot": self.corpus_snapshot,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_passage(raw: dict[str, Any], default_source: str, default_snapshot: str) -> WikipediaPassage:
    passage_id = _clean_text(
        raw.get("passage_id")
        or raw.get("id")
        or raw.get("_id")
        or raw.get("wikipedia_id")
        or raw.get("doc_id")
    )
    title = _clean_text(raw.get("title") or raw.get("page_title") or "Untitled")
    text = _clean_text(raw.get("text") or raw.get("passage") or raw.get("contents"))
    source = _clean_text(raw.get("source") or default_source)
    snapshot_date = _clean_text(raw.get("snapshot_date") or raw.get("snapshot") or default_snapshot)

    if not passage_id:
        passage_id = f"{source}:{abs(hash((title, text))) % 10_000_000}"
    if not text:
        raise ValueError(f"Passage {passage_id!r} is missing text.")

    return WikipediaPassage(
        passage_id=passage_id,
        title=title,
        text=text,
        source=source,
        snapshot_date=snapshot_date,
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number} is not valid JSON.") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path} line {line_number} must contain a JSON object.")
            yield item


def load_local_corpus(
    path: Path | str | None = None,
    default_snapshot: str = "local_fixed_corpus",
) -> list[WikipediaPassage]:
    corpus_path = Path(path) if path else DEFAULT_LOCAL_CORPUS_PATH
    if not corpus_path.exists():
        raise FileNotFoundError(f"Local corpus file not found: {corpus_path}")

    if corpus_path.suffix.lower() == ".jsonl":
        rows = list(iter_jsonl(corpus_path))
    else:
        data = json.loads(corpus_path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("passages", [])
        if not isinstance(rows, list):
            raise ValueError("Corpus JSON must be a list or an object with a 'passages' list.")

    passages = [
        normalize_passage(row, default_source=str(corpus_path), default_snapshot=default_snapshot)
        for row in rows
    ]
    if not passages:
        raise ValueError(f"No passages were loaded from {corpus_path}.")
    return passages


def corpus_snapshot(passages: list[WikipediaPassage], configured_snapshot: str = "") -> str:
    if configured_snapshot:
        return configured_snapshot
    snapshots = sorted({passage.snapshot_date for passage in passages if passage.snapshot_date})
    if not snapshots:
        return "unknown"
    if len(snapshots) == 1:
        return snapshots[0]
    return f"mixed:{','.join(snapshots[:5])}"


def load_wiki_dpr_dataset(max_passages: int | None = None) -> CorpusLoadResult:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for facebook/wiki_dpr.") from exc

    attempts = [
        ("psgs_w100.nq.exact", "train"),
        ("psgs_w100.nq.compressed", "train"),
        ("psgs_w100", "train"),
        (None, "train"),
    ]
    last_error: Exception | None = None
    dataset = None
    config_name = ""
    for config, split in attempts:
        try:
            kwargs: dict[str, Any] = {
                "path": "facebook/wiki_dpr",
                "split": split,
                "trust_remote_code": True,
            }
            if config:
                kwargs["name"] = config
            if max_passages:
                kwargs["streaming"] = True
            dataset = load_dataset(**kwargs)
            if max_passages and hasattr(dataset, "take"):
                preview = list(dataset.take(1))
                if preview and "embeddings" not in preview[0]:
                    raise RuntimeError(f"Config {config_name or config or 'default'} has no DPR embeddings.")
            elif hasattr(dataset, "column_names") and "embeddings" not in dataset.column_names:
                raise RuntimeError(f"Config {config or 'default'} has no DPR embeddings.")
            config_name = config or "default"
            break
        except Exception as exc:
            last_error = exc

    if dataset is None:
        raise RuntimeError(f"Could not load facebook/wiki_dpr: {last_error}") from last_error

    if max_passages and hasattr(dataset, "take"):
        rows = list(dataset.take(max_passages))
    elif max_passages:
        dataset = dataset.select(range(min(max_passages, len(dataset))))
        rows = list(dataset)
    else:
        rows = list(dataset)

    passages: list[WikipediaPassage] = []
    for row in rows:
        passages.append(
            normalize_passage(
                dict(row),
                default_source="facebook/wiki_dpr",
                default_snapshot=f"facebook/wiki_dpr:{config_name}",
            )
        )
    mode = "wikipedia_subset" if max_passages else "full_wikipedia_dpr"
    return CorpusLoadResult(
        passages=passages,
        dataset=dataset,
        corpus_used="facebook/wiki_dpr",
        corpus_mode=mode,
        corpus_snapshot=f"facebook/wiki_dpr:{config_name}",
    )


def load_local_corpus_result(
    path: Path | str | None = None,
    default_snapshot: str = "local_fixed_corpus",
    fallback_reason: str = "",
) -> CorpusLoadResult:
    passages = load_local_corpus(path, default_snapshot=default_snapshot)
    source = str(Path(path).resolve()) if path else "local_wikipedia_sample"
    corpus_mode = "local_fallback_sample" if fallback_reason else "wikipedia_subset"
    return CorpusLoadResult(
        passages=passages,
        dataset=None,
        corpus_used=source,
        corpus_mode=corpus_mode,
        corpus_snapshot=corpus_snapshot(passages, default_snapshot),
        fallback_used=bool(fallback_reason),
        fallback_reason=fallback_reason,
    )


def main() -> None:
    import argparse

    from .retriever import build_wikipedia_retriever, retrieval_quality_check

    parser = argparse.ArgumentParser(description="Smoke test the Wikipedia corpus loader.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--retriever_type", default="hybrid", choices=["dpr", "hybrid", "sentence_transformer"])
    parser.add_argument("--max_passages", type=int, default=None)
    parser.add_argument("--hybrid_scan_limit", type=int, default=150000)
    parser.add_argument("--hybrid_bm25_k", type=int, default=100)
    parser.add_argument("--allow_fallback", action="store_true")
    parser.add_argument("--corpus_path", default=None)
    parser.add_argument("--claim", default="The Eiffel Tower is located in Paris, France.")
    args = parser.parse_args()

    if not args.smoke:
        parser.print_help()
        return

    retriever = build_wikipedia_retriever(
        retriever_type=args.retriever_type,
        corpus_path=args.corpus_path,
        max_passages=args.max_passages,
        allow_fallback=args.allow_fallback,
        hybrid_scan_limit=args.hybrid_scan_limit,
        hybrid_bm25_k=args.hybrid_bm25_k,
    )
    passages = retriever.retrieve(args.claim, top_k=args.top_k)
    quality_passed, quality_reason = retrieval_quality_check(args.claim, passages)
    metadata = retriever.metadata()
    metadata["retrieved_passage_count"] = len(passages)
    metadata["retrieval_quality_check_passed"] = quality_passed
    metadata["retrieval_quality_reason"] = quality_reason
    metadata["retrieved_titles"] = [passage.title for passage in passages]
    print(json.dumps(metadata, indent=2))
    if not passages:
        raise RuntimeError("Smoke test failed: no passages were retrieved.")
    if not quality_passed:
        raise RuntimeError(f"Smoke test failed: {quality_reason}")


if __name__ == "__main__":
    main()
