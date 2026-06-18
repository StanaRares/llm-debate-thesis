from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

from .prompts import build_agent_messages


load_dotenv()

OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}


class OllamaError(RuntimeError):
    pass


@dataclass
class DebateConfig:
    agent_type: str
    rag_mode: str
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    number_of_turns: int = 2


def is_ollama_running() -> bool:
    try:
        response = requests.get(OLLAMA_HEALTH_URL, timeout=1.5)
        return response.status_code < 500
    except requests.RequestException:
        return False


def call_llm(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    json_mode: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
    except requests.RequestException as exc:
        raise OllamaError("Ollama is not running or is unreachable.") from exc
    if response.status_code >= 400:
        raise OllamaError(response.text.strip() or response.reason)
    try:
        return response.json().get("message", {}).get("content", "")
    except Exception as exc:
        raise OllamaError("Ollama returned an invalid response.") from exc


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().upper().replace("_", " ")
    if text in {"SUPPORTS", "SUPPORTED", "SUPPORT", "TRUE"}:
        return "SUPPORTS"
    if text in {"REFUTES", "REFUTED", "REFUTE", "FALSE"}:
        return "REFUTES"
    return "NOT ENOUGH INFO"


def extract_final_answer(text: str) -> str:
    match = re.search(r"final\s+(?:answer|stance)\s*:\s*(supports|refutes|not enough info)", text, re.I)
    if match:
        return normalize_label(match.group(1))
    lowered = text.lower()
    if "not enough info" in lowered or "insufficient evidence" in lowered:
        return "NOT ENOUGH INFO"
    if "refutes" in lowered or "contradicts" in lowered:
        return "REFUTES"
    if "supports" in lowered or "supported" in lowered:
        return "SUPPORTS"
    return "NOT ENOUGH INFO"


def _limit_words(text: str, max_words: int = 180) -> str:
    words = str(text or "").split()
    return " ".join(words[:max_words])


def run_debate(
    claim: str,
    config: DebateConfig,
    retrieved_passages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []

    for turn_number in range(1, max(0, config.number_of_turns) + 1):
        messages = build_agent_messages(
            claim=claim,
            agent_type=config.agent_type,
            rag_mode=config.rag_mode,
            debate_history=turns,
            retrieved_passages=retrieved_passages,
        )
        text = call_llm(messages, config.model, config.temperature).strip()

        text = _limit_words(text, 180)
        turns.append(
            {
                "turn_number": turn_number,
                "speaker": "Agent",
                "agent_type": config.agent_type,
                "rag_mode": config.rag_mode,
                "text": text,
                "extracted_final_answer": extract_final_answer(text),
            }
        )
    return turns
