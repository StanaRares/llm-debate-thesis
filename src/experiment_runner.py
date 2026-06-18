from __future__ import annotations

import argparse
import csv
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .debate_runner import DebateConfig, DEFAULT_MODEL, is_ollama_running, normalize_label, run_debate
from .judge import judge_debate, judge_prompt_version
from .metrics import comparison_tables, compute_run_metrics
from .prompts import normalize_agent_type, normalize_rag_mode
from .retrieval.retriever import build_wikipedia_retriever, retrieval_quality_check


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_PATHS = [
    PROJECT_ROOT / "data" / "evaluation_scenarios.xlsx",
    PROJECT_ROOT / "backend" / "data" / "evaluation_scenarios.xlsx",
]
DEFAULT_CLAIMS_PATH = PROJECT_ROOT / "data" / "claims.xlsx"
DEFAULT_FEVER_LOCAL_PATH = PROJECT_ROOT / "data" / "fever_claims.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WIKIPEDIA_RAG_DATASET_MODE = "Wikipedia RAG"
FEVER_DATASET_MODE = "FEVER sample"
MANUAL_DATASET_MODE = "Manual topic"


FALLBACK_EXAMPLES = [
    {
        "claim": "The Eiffel Tower is located in Berlin.",
        "gold_label": "REFUTES",
        "evidence": "The Eiffel Tower is located on the Champ de Mars in Paris, France.",
        "topic": "geography",
        "difficulty": "easy",
        "source": "fixed_fallback_examples",
    },
    {
        "claim": "Marie Curie won a Nobel Prize in Literature.",
        "gold_label": "REFUTES",
        "evidence": "Marie Curie won Nobel Prizes in Physics and Chemistry.",
        "topic": "biography",
        "difficulty": "easy",
        "source": "fixed_fallback_examples",
    },
    {
        "claim": "The Amazon rainforest is located in South America.",
        "gold_label": "SUPPORTS",
        "evidence": "The Amazon rainforest is a tropical rainforest in South America.",
        "topic": "geography",
        "difficulty": "easy",
        "source": "fixed_fallback_examples",
    },
]


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def resolve_scenario_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    if path_value:
        requested = (PROJECT_ROOT / path_value).resolve() if not Path(path_value).is_absolute() else Path(path_value)
        if requested.exists():
            return requested
        if requested.name == "evaluation_scenarios.xlsx":
            for candidate in DEFAULT_SCENARIO_PATHS:
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(f"Scenario file not found: {requested}")

    return None


def normalize_scenario_row(row: dict[str, Any], row_number: int) -> dict[str, Any]:
    scenario_id = str(row.get("scenario_id") or f"S{row_number}").strip()
    return {
        "scenario_id": scenario_id,
        "scenario_name": str(row.get("scenario_name") or row.get("name") or scenario_id).strip(),
        "description": str(row.get("description") or "").strip(),
        "enabled": str_to_bool(row.get("enabled", True)),
        "number_of_turns": int(float(row.get("number_of_turns") or row.get("turns") or 2)),
        "number_of_claims": int(float(row.get("number_of_claims") or 1)),
        "repeats_per_claim": int(float(row.get("repeats_per_claim") or 1)),
        "random_seed": int(float(row.get("random_seed") or 42)),
        "judge_model": str(row.get("judge_model") or DEFAULT_MODEL).strip(),
        "judge_prompt_type": str(row.get("judge_prompt_type") or "neutral").strip(),
        "temperature": float(row.get("temperature") or 0.3),
        "label_filter": str(row.get("label_filter") or "Any").strip(),
        "dataset_mode": str(row.get("dataset_mode") or "FEVER sample").strip(),
        "topic": str(row.get("topic") or "").strip(),
        "difficulty": str(row.get("difficulty") or "").strip(),
        "claim": str(row.get("claim") or "").strip(),
        "gold_label": normalize_label(row.get("gold_label")),
        "evidence": str(row.get("evidence") or "").strip(),
    }


def load_scenarios(path: Path, include_disabled: bool = False) -> list[dict[str, Any]]:
    dataframe = pd.read_excel(path, dtype=object).fillna("")
    rows = [
        normalize_scenario_row(row.to_dict(), row_number=index + 1)
        for index, row in dataframe.iterrows()
    ]
    scenarios = [row for row in rows if include_disabled or row["enabled"]]
    if not scenarios:
        raise ValueError("No enabled scenarios were found.")
    return scenarios


def resolve_claims_path(path_value: str | None) -> Path:
    requested = (
        (PROJECT_ROOT / path_value).resolve()
        if path_value and not Path(path_value).is_absolute()
        else Path(path_value)
        if path_value
        else DEFAULT_CLAIMS_PATH
    )
    if not requested.exists():
        raise FileNotFoundError(f"Claims file not found: {requested}")
    return requested


def load_claims(path: Path, label_filter: str = "Any", limit: int | None = None) -> list[dict[str, Any]]:
    dataframe = pd.read_excel(path, dtype=object).fillna("")
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    required = ["claim_id", "claim", "gold_label", "topic", "difficulty"]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Claims file is missing required columns: {', '.join(missing)}")

    claims: list[dict[str, Any]] = []
    for index, row in dataframe.iterrows():
        claim = _compact_text(row.get("claim"))
        if not claim:
            raise ValueError(f"Claims file row {index + 2} has an empty claim.")
        gold_label = normalize_label(row.get("gold_label"))
        if label_filter != "Any" and gold_label != label_filter:
            continue
        claims.append(
            {
                "claim_id": str(row.get("claim_id") or f"C{index + 1:03d}").strip(),
                "claim": claim,
                "gold_label": gold_label,
                "topic": str(row.get("topic") or "").strip(),
                "difficulty": str(row.get("difficulty") or "").strip(),
                "evidence": "",
                "source": str(path),
            }
        )
        if limit and len(claims) >= limit:
            break

    if not claims:
        raise ValueError(f"No claims were loaded from {path}.")
    return claims


def load_local_fever_examples(path: Path, label_filter: str, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Local FEVER file not found: {path}")
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number} is not valid JSON.") from exc
            claim = _compact_text(row.get("claim"))
            if not claim:
                raise ValueError(f"{path} line {line_number} has an empty claim.")
            gold_label = normalize_label(row.get("gold_label") or row.get("label"))
            if label_filter != "Any" and gold_label != label_filter:
                continue
            examples.append(
                {
                    "claim_id": str(row.get("claim_id") or row.get("id") or f"FEVER-{line_number:06d}"),
                    "claim": claim,
                    "gold_label": gold_label,
                    "evidence": _compact_text(row.get("evidence") or row.get("gold_evidence") or ""),
                    "topic": str(row.get("topic") or "").strip(),
                    "difficulty": str(row.get("difficulty") or "").strip(),
                    "source": str(path),
                }
            )
            if limit and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No FEVER examples were loaded from {path}.")
    return examples


def _compact_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_compact_text(item) for item in value[:6])
    if isinstance(value, dict):
        parts = []
        for key in ["title", "text", "sentence", "evidence", "content"]:
            if key in value:
                parts.append(_compact_text(value[key]))
        return " ".join(parts)
    return " ".join(str(value or "").split())


def load_fever_examples(
    label_filter: str,
    limit: int,
    seed: int,
    allow_fallback: bool,
    fever_local_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if fever_local_path and fever_local_path.exists():
        return load_local_fever_examples(fever_local_path, label_filter=label_filter, limit=limit), {
            "scenario_source": "local_fever_jsonl",
            "fever_local_path": str(fever_local_path),
            "fallback_used": False,
            "fallback_reason": "",
        }

    try:
        from datasets import load_dataset

        dataset = load_dataset("copenlu/fever_gold_evidence", trust_remote_code=True)
        split_name = "train" if "train" in dataset else list(dataset.keys())[0]
        split = dataset[split_name].shuffle(seed=seed)
        examples = []
        for row in split:
            label = normalize_label(row.get("label") or row.get("gold_label"))
            if label_filter != "Any" and label != label_filter:
                continue
            evidence = ""
            for key in ["evidence", "evidence_text", "gold_evidence", "gold_evidence_text", "evidence_sentence"]:
                if key in row:
                    evidence = _compact_text(row[key])
                    if evidence:
                        break
            examples.append(
                {
                    "claim_id": str(row.get("id") or row.get("claim_id") or f"HF-FEVER-{len(examples) + 1:06d}"),
                    "claim": _compact_text(row.get("claim") or row.get("sentence")),
                    "gold_label": label,
                    "evidence": evidence,
                    "source": "copenlu/fever_gold_evidence",
                }
            )
            if len(examples) >= limit:
                return examples, {
                    "scenario_source": "copenlu/fever_gold_evidence",
                    "fallback_used": False,
                    "fallback_reason": "",
                }
        if not examples:
            raise RuntimeError("No FEVER rows matched the requested label filter.")
    except Exception as exc:
        fallback_reason = f"FEVER loading failed: {exc}"
        if not allow_fallback:
            raise RuntimeError(
                f"{fallback_reason}. Pass --allow_fallback to use fixed fallback examples."
            ) from exc
        print(f"[data] fallback enabled; using fixed examples. Reason: {fallback_reason}")

    filtered = [
        item for item in FALLBACK_EXAMPLES
        if label_filter == "Any" or item["gold_label"] == label_filter
    ]
    return [filtered[index % len(filtered)] for index in range(limit)], {
        "scenario_source": "fixed_fallback_examples",
        "fallback_used": True,
        "fallback_reason": fallback_reason,
    }


def examples_for_scenario(
    scenario: dict[str, Any],
    claims_per_scenario: int | None = None,
    allow_fallback: bool = False,
    claims_path: Path | None = None,
    fever_local_path: Path | None = None,
    dataset_override: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_mode = dataset_override or str(scenario.get("dataset_mode") or FEVER_DATASET_MODE).strip()
    limit = claims_per_scenario or scenario["number_of_claims"]

    if dataset_mode.lower() == WIKIPEDIA_RAG_DATASET_MODE.lower():
        path = claims_path or DEFAULT_CLAIMS_PATH
        examples = load_claims(path, label_filter=scenario["label_filter"], limit=limit)
        return examples, {
            "scenario_source": "excel_scenarios+claims_xlsx",
            "claims_file": str(path),
            "fallback_used": False,
            "fallback_reason": "",
        }

    if dataset_mode.lower() == FEVER_DATASET_MODE.lower():
        examples, metadata = load_fever_examples(
            label_filter=scenario["label_filter"],
            limit=limit,
            seed=scenario["random_seed"],
            allow_fallback=allow_fallback,
            fever_local_path=fever_local_path,
        )
        for example in examples:
            example.setdefault("topic", scenario.get("topic", ""))
            example.setdefault("difficulty", scenario.get("difficulty", ""))
        if metadata["scenario_source"] == "copenlu/fever_gold_evidence":
            metadata["scenario_source"] = "excel_scenarios+copenlu/fever_gold_evidence"
        return examples, metadata

    if scenario.get("claim"):
        return [
            {
                "claim_id": f"{scenario['scenario_id']}-manual",
                "claim": scenario["claim"],
                "gold_label": scenario["gold_label"],
                "evidence": scenario.get("evidence", ""),
                "topic": scenario.get("topic", ""),
                "difficulty": scenario.get("difficulty", ""),
                "source": "scenario_workbook",
            }
        ], {
            "scenario_source": "excel_scenarios",
            "fallback_used": False,
            "fallback_reason": "",
        }

    raise ValueError(
        f"Unsupported dataset_mode {dataset_mode!r}. "
        f"Use '{WIKIPEDIA_RAG_DATASET_MODE}', '{FEVER_DATASET_MODE}', or '{MANUAL_DATASET_MODE}'."
    )


def default_fever_scenario() -> dict[str, Any]:
    return {
        "scenario_id": "FEVER",
        "scenario_name": "Default FEVER scenario",
        "description": "Default scenario generated because no Excel scenario file was provided.",
        "enabled": True,
        "number_of_turns": 2,
        "number_of_claims": 1,
        "repeats_per_claim": 1,
        "random_seed": 42,
        "judge_model": DEFAULT_MODEL,
        "judge_prompt_type": "neutral",
        "temperature": 0.3,
        "label_filter": "Any",
        "dataset_mode": "FEVER sample",
        "topic": "",
        "difficulty": "",
        "claim": "",
        "gold_label": "NOT ENOUGH INFO",
        "evidence": "",
    }


def condition_values(value: str, allowed: list[str]) -> list[str]:
    if value == "all":
        return allowed
    return [value]


def make_retriever(args: argparse.Namespace, needs_rag: bool) -> Any | None:
    if not needs_rag:
        return None
    return build_wikipedia_retriever(
        retriever_type=args.retriever_type,
        corpus_snapshot_value=args.corpus_snapshot,
        corpus_path=args.corpus_path,
        max_passages=args.max_passages,
        allow_fallback=args.allow_fallback,
        hybrid_scan_limit=args.hybrid_scan_limit,
        hybrid_bm25_k=args.hybrid_bm25_k,
    )


def final_agent_answer(debate_turns: list[dict[str, Any]]) -> str:
    if not debate_turns:
        return ""
    return str(debate_turns[-1].get("extracted_final_answer") or "")


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(row)
    for key in ["retrieved_passages", "debate_turns", "judge_scores"]:
        flattened[key] = json.dumps(flattened.get(key, []), ensure_ascii=False)
    return flattened


def write_outputs(payload: dict[str, Any], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = payload["runs"]
    if rows:
        csv_rows = [flatten_for_csv(row) for row in rows]
        fieldnames = list(csv_rows[0].keys())
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    scenario_path = resolve_scenario_path(args.scenarios)
    scenarios = (
        load_scenarios(scenario_path, include_disabled=args.include_disabled)
        if scenario_path
        else [default_fever_scenario()]
    )
    if args.limit_scenarios:
        scenarios = scenarios[: args.limit_scenarios]

    agent_types = condition_values(args.agent_type, ["truth", "deceptive"])
    rag_modes = condition_values(args.rag_mode, ["none", "full_wikipedia"])
    needs_rag = "full_wikipedia" in rag_modes
    retriever = make_retriever(args, needs_rag)
    dataset_mode_override = {
        "fever": FEVER_DATASET_MODE,
        "claims": WIKIPEDIA_RAG_DATASET_MODE,
        "scenario": "",
    }[args.dataset]
    claims_path = resolve_claims_path(args.claims) if (
        dataset_mode_override == WIKIPEDIA_RAG_DATASET_MODE
        or any(
            str(scenario.get("dataset_mode", "")).strip().lower() == WIKIPEDIA_RAG_DATASET_MODE.lower()
            for scenario in scenarios
        )
    ) else None
    fever_local_path = Path(args.fever_local_path).resolve() if args.fever_local_path else None

    mock_mode = str_to_bool(args.mock_mode) if args.mock_mode is not None else False
    if not mock_mode and not is_ollama_running():
        raise RuntimeError("Ollama is not running. Start Ollama or pass --mock_mode true for explicit demo mode.")
    experiment_id = args.run_id or f"rag-exp-{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    runs: list[dict[str, Any]] = []
    scenario_sources: dict[str, dict[str, Any]] = {}
    fallback_events: list[str] = []
    if retriever and retriever.fallback_used:
        fallback_events.append(retriever.fallback_reason)

    for scenario in scenarios:
        examples, scenario_metadata = examples_for_scenario(
            scenario,
            claims_per_scenario=args.claims_per_scenario,
            allow_fallback=args.allow_fallback,
            claims_path=claims_path,
            fever_local_path=fever_local_path,
            dataset_override=dataset_mode_override,
        )
        scenario_sources[scenario["scenario_id"]] = scenario_metadata
        if scenario_metadata.get("fallback_used"):
            fallback_events.append(str(scenario_metadata.get("fallback_reason", "")))
        rng = random.Random(scenario["random_seed"])
        selected_examples = list(examples)
        rng.shuffle(selected_examples)
        for claim_index, example in enumerate(selected_examples[: args.claims_per_scenario or len(selected_examples)], start=1):
            repeat_count = 1 if args.smoke else scenario["repeats_per_claim"]
            for repeat_id in range(1, repeat_count + 1):
                for agent_type in agent_types:
                    agent_type = normalize_agent_type(agent_type)
                    for rag_mode in rag_modes:
                        rag_mode = normalize_rag_mode(rag_mode)
                        retrieved = (
                            [passage.to_dict() for passage in retriever.retrieve(example["claim"], args.top_k)]
                            if rag_mode == "full_wikipedia" and retriever
                            else []
                        )
                        quality_passed, quality_reason = (
                            retrieval_quality_check(example["claim"], retrieved)
                            if retrieved
                            else (rag_mode == "none", "No retrieval in prompt-only condition.")
                        )

                        debate_config = DebateConfig(
                            agent_type=agent_type,
                            rag_mode=rag_mode,
                            model=scenario["judge_model"],
                            temperature=scenario["temperature"],
                            number_of_turns=scenario["number_of_turns"],
                            mock_mode=mock_mode,
                        )
                        debate_turns = run_debate(example["claim"], debate_config, retrieved)
                        judge_scores, raw_judge_output = judge_debate(
                            claim=example["claim"],
                            gold_label=example["gold_label"],
                            debate_turns=debate_turns,
                            agent_type=agent_type,
                            model=scenario["judge_model"],
                            temperature=scenario["temperature"],
                            judge_gets_evidence=args.judge_gets_evidence,
                            retrieved_passages=retrieved,
                            mock_mode=mock_mode,
                        )
                        metric_values = compute_run_metrics(example, debate_turns, retrieved, judge_scores)
                        run_id = (
                            f"{experiment_id}-{scenario['scenario_id']}-C{claim_index:03d}-"
                            f"R{repeat_id}-{agent_type}-{rag_mode}"
                        )
                        row = {
                            "run_id": run_id,
                            "scenario_id": scenario["scenario_id"],
                            "claim": example["claim"],
                            "claim_id": example.get("claim_id", f"C{claim_index:03d}"),
                            "gold_label": normalize_label(example["gold_label"]),
                            "agent_type": agent_type,
                            "rag_mode": rag_mode,
                            "retriever_type": args.retriever_type,
                            "effective_retriever_type": retriever.retriever_type if retriever else "none",
                            "top_k": args.top_k,
                            "corpus_used": retriever.corpus_used if retriever else "",
                            "corpus_mode": retriever.corpus_mode if retriever else "",
                            "corpus_snapshot": retriever.snapshot if retriever else args.corpus_snapshot,
                            "fallback_used": bool((retriever and retriever.fallback_used) or scenario_metadata.get("fallback_used")),
                            "fallback_reason": "; ".join(
                                item
                                for item in [
                                    retriever.fallback_reason if retriever else "",
                                    str(scenario_metadata.get("fallback_reason", "")),
                                ]
                                if item
                            ),
                            "scenario_source": scenario_metadata.get("scenario_source", ""),
                            "retrieved_passages": retrieved,
                            "retrieval_quality_check_passed": quality_passed,
                            "retrieval_quality_reason": quality_reason,
                            "debate_turns": debate_turns,
                            "final_agent_answer": final_agent_answer(debate_turns),
                            "judge_prompt_version": judge_prompt_version(),
                            "judge_scores": judge_scores,
                            "judge_predicted_agent_type": judge_scores.get("predicted_agent_type"),
                            "judge_correct": bool(judge_scores.get("judge_correct")),
                            "raw_judge_output": raw_judge_output,
                            "topic": example.get("topic") or scenario.get("topic", ""),
                            "difficulty": example.get("difficulty") or scenario.get("difficulty", ""),
                            "number_of_turns": scenario["number_of_turns"],
                            "judge_model": scenario["judge_model"],
                            "temperature": scenario["temperature"],
                            "judge_gets_evidence": args.judge_gets_evidence,
                            "mock_mode": mock_mode,
                            **metric_values,
                        }
                        runs.append(row)

    payload = {
        "experiment_id": experiment_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario_file": str(scenario_path) if scenario_path else "",
        "claims_file": str(claims_path) if claims_path else "",
        "fever_local_path": str(fever_local_path) if fever_local_path else "",
        "scenarios_used": [scenario["scenario_id"] for scenario in scenarios],
        "scenario_source": "excel_scenarios" if scenario_path else "fever",
        "scenario_sources": scenario_sources,
        "fallback_used": bool(fallback_events),
        "fallback_reason": "; ".join(item for item in fallback_events if item),
        "config": {
            "agent_type": args.agent_type,
            "rag_mode": args.rag_mode,
            "top_k": args.top_k,
            "retriever_type": args.retriever_type,
            "judge_gets_evidence": args.judge_gets_evidence,
            "corpus_snapshot": args.corpus_snapshot,
            "allow_fallback": args.allow_fallback,
            "mock_mode": mock_mode,
            "claims_file": str(claims_path) if claims_path else "",
            "dataset": args.dataset,
            "fever_local_path": str(fever_local_path) if fever_local_path else "",
            "retrieval": retriever.metadata() if retriever else {},
            "controlled_variables": [
                "scenario list",
                "debate format",
                "judge prompt version",
                "number_of_turns",
                "model settings",
            ],
        },
        "runs": runs,
        "summary_tables": comparison_tables(runs),
    }

    output_json = Path(args.output_json) if args.output_json else OUTPUT_DIR / f"{experiment_id}.json"
    output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
    if args.smoke and not args.output_json:
        output_json = OUTPUT_DIR / "smoke_test_results.json"
        output_csv = OUTPUT_DIR / "smoke_test_results.csv"
    write_outputs(payload, output_json, output_csv)
    payload["output_json"] = str(output_json)
    payload["output_csv"] = str(output_csv)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prompt-only vs full Wikipedia RAG debate experiments.")
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--dataset", default="fever", choices=["fever", "claims", "scenario"])
    parser.add_argument("--fever_local_path", default=str(DEFAULT_FEVER_LOCAL_PATH))
    parser.add_argument("--claims", default=str(DEFAULT_CLAIMS_PATH))
    parser.add_argument("--agent_type", default="all", choices=["truth", "deceptive", "all"])
    parser.add_argument("--rag_mode", default="all", choices=["none", "full_wikipedia", "all"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--retriever_type", default="hybrid", choices=["dpr", "hybrid", "sentence_transformer"])
    parser.add_argument("--judge_gets_evidence", type=str_to_bool, default=True)
    parser.add_argument("--corpus_snapshot", default="")
    parser.add_argument("--corpus_path", default=None)
    parser.add_argument("--max_passages", type=int, default=None)
    parser.add_argument("--hybrid_scan_limit", type=int, default=150000)
    parser.add_argument("--hybrid_bm25_k", type=int, default=100)
    parser.add_argument("--allow_fallback", action="store_true")
    parser.add_argument("--include_disabled", action="store_true")
    parser.add_argument("--limit_scenarios", type=int, default=None)
    parser.add_argument("--claims_per_scenario", type=int, default=None)
    parser.add_argument("--mock_mode", choices=["true", "false"], default=None)
    parser.add_argument("--run_id", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.smoke:
        args.limit_scenarios = args.limit_scenarios or 3
        args.claims_per_scenario = args.claims_per_scenario or 1
        if args.top_k == 5:
            args.top_k = 3
        if args.retriever_type != "hybrid":
            args.max_passages = args.max_passages or 64
    payload = run_experiment(args)
    print(f"Saved JSON: {payload['output_json']}")
    print(f"Saved CSV: {payload['output_csv']}")
    print(f"Runs: {len(payload['runs'])}")


if __name__ == "__main__":
    main()
