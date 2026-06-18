# Bachelor Thesis LLM Debate Experiment System

This repository contains the code for a bachelor thesis experiment system for studying multi-agent LLM debates on FEVER-style fact verification tasks.

The system compares three controlled agent behaviors:

- Truth-Oriented
- Deceptive / Strategically Misleading
- Persuasion-Optimized

It uses local Ollama models, FEVER claims and gold labels, controlled FEVER-based RAG context, and Excel-defined evaluation scenarios.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama and pull the default model:

```bash
ollama pull llama3.2:3b
```

You can set a different local model in `.env`:

```text
OLLAMA_MODEL=llama3.2:3b
```

## Streamlit App

Run the interactive thesis evaluation app:

```bash
python -m streamlit run app.py
```

The app loads FEVER examples, runs agent debates, evaluates behavior with a local judge model, supports scenario execution from Excel, and saves evaluation artifacts locally.

## Experiment Runner

Run the FEVER experiment runner:

```bash
python -m src.experiment_runner --scenarios data/evaluation_scenarios.xlsx --rag_mode fever --top_k 5
```

Experiment outputs are written to `outputs/`.

## Scenario Workbook

Scenario-based evaluations are loaded from:

```text
data/evaluation_scenarios.xlsx
```

Each row defines the agent pairing, dataset mode, claim count, repeats, judge model, prompt focus, and run settings. The Streamlit app also supports uploading and downloading scenario workbooks.

Expected columns:

```text
scenario_id, scenario_name, description, enabled, agent_a_type, agent_b_type,
starting_agent, number_of_turns, dataset_mode, label_filter, number_of_claims,
repeats_per_claim, random_seed, judge_model, judge_prompt_type, temperature, notes
```

## Data Sources

FEVER remains the main thesis dataset. The runner loads `copenlu/fever_gold_evidence` through Hugging Face datasets by default. A local FEVER JSONL file can be provided explicitly when it contains the required gold evidence and Wikipedia source-page metadata.

The thesis uses a controlled FEVER-based RAG setup instead of full Wikipedia retrieval. FEVER RAG gives agents gold FEVER evidence and separately labeled expanded context retrieved from the same FEVER/Wikipedia source pages. This reduces engineering risk and keeps the experiment focused on agent behavior. If required FEVER evidence, source-page context, data files, or the Ollama service are unavailable, the system raises a clear error.

## API

The optional FastAPI service exposes batch and scenario evaluation endpoints:

```bash
uvicorn api:api --reload
```

```text
GET  /api/evaluation/scenarios
GET  /api/evaluation/scenarios/download
POST /api/evaluation/scenarios/upload
POST /api/evaluation/run-batch
POST /api/evaluation/run-scenarios
```
