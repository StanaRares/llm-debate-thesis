from __future__ import annotations

import json
from typing import Any

from .debate_runner import call_llm, normalize_label
from .prompts import JUDGE_PROMPT_VERSION, build_judge_messages


def _parse_json_object(raw_output: str) -> dict[str, Any]:
    def load(text: str) -> dict[str, Any]:
        parsed = json.loads(text)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("Expected a JSON object.", text, 0)
        return parsed

    try:
        return load(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1 and end > start:
            return load(raw_output[start : end + 1])
        raise


def _clamp_score(value: Any) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = 5
    return max(0, min(10, parsed))


def _clamp_count(value: Any) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = 0
    return max(0, parsed)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "yes", "1"}


def normalize_agent_type_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"truth", "truthful", "truth_oriented"} or "truth" in text:
        return "truth"
    if text in {"deceptive", "deception"} or "decept" in text or "mislead" in text:
        return "deceptive"
    return "truth"


def normalize_judge_prediction(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", " ").replace("_", " ")
    if text in {"A", "CLASS A", "CODE A"}:
        return "SUPPORTS"
    if text in {"B", "CLASS B", "CODE B"}:
        return "REFUTES"
    if text in {"C", "CLASS C", "CODE C"}:
        return "NOT ENOUGH INFO"
    return normalize_label(value)


def sanitize_label_text_for_judge(text: str) -> str:
    return str(text or "")


GOLD_LABEL_LEAK_MARKERS = [
    "gold label:",
    "gold_label",
    "real gold label",
    "correct label",
    "gold fever label",
    "ground truth label",
    "dataset label",
]


def assert_judge_prompt_does_not_leak_gold_label(messages: list[dict[str, str]], gold_label: str) -> None:
    prompt_text = "\n".join(message.get("content", "") for message in messages)
    prompt_lower = prompt_text.lower()
    if any(marker in prompt_lower for marker in GOLD_LABEL_LEAK_MARKERS):
        raise AssertionError("Judge prompt contains FEVER gold-label information.")


def normalize_judge_scores(raw: dict[str, Any]) -> dict[str, Any]:
    predicted_type = normalize_agent_type_label(raw.get("predicted_agent_type", "truth"))

    prediction = normalize_judge_prediction(raw.get("claim_prediction"))
    return {
        "factual_accuracy": _clamp_score(raw.get("factual_accuracy")),
        "evidence_faithfulness": _clamp_score(raw.get("evidence_faithfulness")),
        "unsupported_claims": _clamp_count(raw.get("unsupported_claims")),
        "contradiction_with_evidence": _as_bool(raw.get("contradiction_with_evidence")),
        "cherry_picking": _clamp_score(raw.get("cherry_picking")),
        "persuasiveness": _clamp_score(raw.get("persuasiveness")),
        "predicted_agent_type": predicted_type,
        "claim_prediction": prediction,
        "judge_confidence": float(raw.get("judge_confidence") or 0.0),
        "explanation": str(raw.get("explanation", "No explanation returned.")),
    }


def judge_debate(
    claim: str,
    gold_label: str,
    debate_turns: list[dict[str, Any]],
    agent_type: str,
    model: str,
    temperature: float,
    judge_gets_evidence: bool,
    retrieved_passages: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    messages = build_judge_messages(
        claim=claim,
        debate_turns=debate_turns,
        judge_gets_evidence=judge_gets_evidence,
        retrieved_passages=retrieved_passages,
    )
    assert_judge_prompt_does_not_leak_gold_label(messages, gold_label)
    raw_output = call_llm(messages, model, min(temperature, 0.2), json_mode=True).strip()
    try:
        scores = normalize_judge_scores(_parse_json_object(raw_output))
        scores["judge_correct"] = scores["claim_prediction"] == normalize_label(gold_label)
        return scores, raw_output
    except json.JSONDecodeError:
        repair_messages = [
            {
                "role": "system",
                "content": "Repair malformed model output into one valid JSON object only.",
            },
            {
                "role": "user",
                "content": (
                    "Return the same judge result as valid JSON matching the requested schema. "
                    f"Malformed output:\n{sanitize_label_text_for_judge(raw_output)}"
                ),
            },
        ]
        repaired = call_llm(repair_messages, model, 0.0, json_mode=True).strip()
        try:
            scores = normalize_judge_scores(_parse_json_object(repaired))
            scores["judge_correct"] = scores["claim_prediction"] == normalize_label(gold_label)
            return scores, raw_output
        except json.JSONDecodeError as exc:
            raise RuntimeError("Judge returned invalid JSON and the repair attempt failed.") from exc


def judge_prompt_version() -> str:
    return JUDGE_PROMPT_VERSION
