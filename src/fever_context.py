from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

import requests


MAX_FEVER_EVIDENCE_ITEMS = 5
MIN_EXPANDED_CONTEXT_ITEMS = 0
MAX_EXPANDED_CONTEXT_ITEMS = 5
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def compact_text(value: Any, max_items: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key in ["text", "sentence", "evidence", "content", "page", "title"]:
            if key in value:
                text = compact_text(value[key], max_items=max_items)
                if text:
                    parts.append(text)
        if parts:
            return " ".join(parts)
        return " ".join(compact_text(item, max_items=max_items) for item in value.values())
    if isinstance(value, (list, tuple)):
        parts = [compact_text(item, max_items=max_items) for item in value[:max_items]]
        return " ".join(part for part in parts if part)
    return str(value).strip()


def clean_fever_text(text: Any) -> str:
    cleaned = compact_text(text) if not isinstance(text, str) else text
    for old, new in {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
    }.items():
        cleaned = cleaned.replace(old, new)

    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    repeated_title_pattern = re.compile(
        r"(?<!\w)([A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,8})\s+\d+\s+\1(?=\s|,|\.|$)"
    )
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = repeated_title_pattern.sub(r"\1", cleaned)

    cleaned = re.sub(r"\s+([,.;:!?%)\]])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[])\s+", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def wikipedia_url_for_title(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='')}"


def first_present_text(item: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in item and item[key]:
            text = clean_fever_text(item[key])
            if text:
                return text
    return ""


def extract_evidence_sources(value: Any, limit: int = MAX_FEVER_EVIDENCE_ITEMS) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_source(title: Any, sentence: Any, sentence_id: Any = "") -> None:
        cleaned_title = clean_fever_text(title)
        cleaned_sentence = clean_fever_text(sentence)
        if not cleaned_title or not cleaned_sentence:
            return
        key = (cleaned_title, cleaned_sentence)
        if key in seen:
            return
        seen.add(key)
        sources.append(
            {
                "title": cleaned_title,
                "sentence": cleaned_sentence,
                "url": wikipedia_url_for_title(cleaned_title),
                "sentence_id": str(sentence_id or "").strip(),
            }
        )

    def walk(item: Any) -> None:
        if len(sources) >= limit:
            return
        if isinstance(item, dict):
            title = first_present_text(
                item,
                ["title", "page", "page_title", "wikipedia_title", "wiki_title", "document_title"],
            )
            sentence = first_present_text(
                item,
                ["sentence", "text", "evidence_text", "gold_evidence_text", "content"],
            )
            add_source(title, sentence)
            for child in item.values():
                if len(sources) >= limit:
                    break
                if isinstance(child, (dict, list, tuple)):
                    walk(child)
            return

        if isinstance(item, (list, tuple)):
            if len(item) >= 5 and isinstance(item[2], str) and isinstance(item[4], str):
                add_source(item[2], item[4], item[3] if len(item) > 3 else "")
            elif len(item) >= 3 and isinstance(item[0], str) and isinstance(item[2], str):
                add_source(item[0], item[2], item[1] if len(item) > 1 else "")
            elif len(item) >= 2 and isinstance(item[0], str) and isinstance(item[1], str):
                add_source(item[0], item[1])
            for child in item:
                if len(sources) >= limit:
                    break
                if isinstance(child, (dict, list, tuple)):
                    walk(child)

    walk(value)
    return sources[:limit]


def format_evidence_items_for_context(evidence_sources: list[dict[str, str]], limit: int = MAX_FEVER_EVIDENCE_ITEMS) -> str:
    return "\n".join(
        f"[{index}] {source['title']}: {source['sentence']}"
        for index, source in enumerate(evidence_sources[:limit], start=1)
    )


def evidence_sources_to_passages(
    evidence_sources: list[dict[str, str]],
    evidence_type: str,
    limit: int = MAX_FEVER_EVIDENCE_ITEMS,
) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "title": source["title"],
            "text": source["sentence"],
            "url": source.get("url") or wikipedia_url_for_title(source["title"]),
            "evidence_type": evidence_type,
            "source": "fever_gold" if evidence_type == "gold" else "fever_source_page",
        }
        for index, source in enumerate(evidence_sources[:limit], start=1)
    ]


def tokenize_for_relevance(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in {"the", "and", "for", "with", "that", "this", "from", "into", "was", "were"}
    }


def split_wikipedia_sentences(text: str) -> list[str]:
    cleaned = clean_fever_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [part.strip() for part in parts if len(part.strip()) > 30]


def fetch_wikipedia_page_sentences(title: str) -> list[str]:
    response = requests.get(
        WIKIPEDIA_API_URL,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "format": "json",
            "titles": title,
        },
        headers={"User-Agent": "TezaBachelorThesisExperiment/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    if not pages:
        raise RuntimeError(f"Wikipedia returned no page data for {title!r}.")
    page = next(iter(pages.values()))
    if "missing" in page:
        raise RuntimeError(f"Wikipedia page not found: {title}.")
    sentences = split_wikipedia_sentences(page.get("extract", ""))
    if not sentences:
        raise RuntimeError(f"Wikipedia page {title!r} did not provide extract sentences.")
    return sentences


def is_same_sentence(left: str, right: str) -> bool:
    left_tokens = tokenize_for_relevance(left)
    right_tokens = tokenize_for_relevance(right)
    if not left_tokens or not right_tokens:
        return clean_fever_text(left).lower() == clean_fever_text(right).lower()
    overlap = len(left_tokens & right_tokens)
    return overlap / max(1, min(len(left_tokens), len(right_tokens))) >= 0.85


def select_expanded_context(
    claim: str,
    evidence_sources: list[dict[str, str]],
    count: int,
) -> list[dict[str, str]]:
    count = max(MIN_EXPANDED_CONTEXT_ITEMS, min(MAX_EXPANDED_CONTEXT_ITEMS, int(count)))
    if count == 0:
        return []
    claim_tokens = tokenize_for_relevance(claim)
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for source in evidence_sources:
        title = source.get("title", "")
        if not title:
            continue
        sentences = fetch_wikipedia_page_sentences(title)
        gold_sentence = source.get("sentence", "")
        relevance_tokens = claim_tokens | tokenize_for_relevance(gold_sentence)
        sentence_id_text = source.get("sentence_id", "")
        candidates: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            if is_same_sentence(sentence, gold_sentence):
                continue
            sentence_tokens = tokenize_for_relevance(sentence)
            score = float(len(relevance_tokens & sentence_tokens))
            if sentence_id_text.isdigit():
                distance = abs(index - int(sentence_id_text))
                if 0 < distance <= 2:
                    score += 5.0 - distance
            candidates.append((score, -index, sentence))

        for _, _, sentence in sorted(candidates, reverse=True):
            key = (title, sentence)
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    "title": title,
                    "sentence": sentence,
                    "url": wikipedia_url_for_title(title),
                }
            )
            if len(selected) >= count:
                return selected

    return selected
