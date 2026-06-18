# LLM Debate Behaviour Demo

This is a simple Streamlit proof-of-concept for a bachelor thesis about multi-agent LLM debates.
It uses local Ollama models and FEVER-style factual claims from `copenlu/fever_gold_evidence`.

The demo compares three agent behaviors:

- Truth-Oriented
- Deceptive / Strategically Misleading, as a controlled academic research simulation
- Persuasion-Optimized

The app loads a small FEVER sample, runs a short debate about whether a claim is supported by evidence, asks a local LLM judge to evaluate the agents, and saves each run as JSON in `debate_logs/`.

Mock mode is kept as a fallback. If Ollama is not running, or if the dataset cannot be loaded, the app still runs with local hardcoded examples and mock responses.

## Local Setup

1. Install Ollama:
   https://ollama.com/download

2. Pull the model:

```bash
ollama pull llama3.2:3b
```

3. Test the model:

```bash
ollama run llama3.2:3b
```

4. Install Python packages:

```bash
pip install -r requirements.txt
```

5. Run the app:

```bash
streamlit run app.py
```

On Windows, use the project launcher:

```bash
.\run_app.bat
```

It uses the local `.venv` and opens Streamlit on port `8502`. If you prefer
manual commands, use:

```bash
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

6. Optional: run the batch evaluation API:

```bash
uvicorn api:api --reload
```

The batch endpoint is:

```text
POST /api/evaluation/run-batch
```

Scenario-based thesis evaluations are loaded from:

```text
backend/data/evaluation_scenarios.xlsx
```

If the file is missing, the app creates a default workbook with scenarios S0-S7.
The Streamlit app includes a **Scenario Evaluation** section where scenarios can
be selected, run, uploaded, downloaded, graphed, and exported as JSON or CSV.

The scenario API endpoints are:

```text
GET  /api/evaluation/scenarios
GET  /api/evaluation/scenarios/download
POST /api/evaluation/scenarios/upload
POST /api/evaluation/run-scenarios
```

Each scenario row uses these columns:

```text
scenario_id, scenario_name, description, enabled, agent_a_type, agent_b_type,
starting_agent, number_of_turns, dataset_mode, label_filter, number_of_claims,
repeats_per_claim, random_seed, judge_model, judge_prompt_type, temperature, notes
```

## Wikipedia RAG Experiment Runner

The experimental RAG runner lives in `src/` and compares these controlled
conditions while keeping scenario list, turn count, judge prompt version, and
model settings fixed:

```text
truth + prompt-only
truth + full_wikipedia_rag
deceptive + prompt-only
deceptive + full_wikipedia_rag
```

Run the required smoke test:

```bash
python -m src.experiment_runner --smoke --rag_mode full_wikipedia --top_k 3 --scenarios data/evaluation_scenarios.xlsx
```

This runs 3 scenarios with `top_k=3` and writes:

```text
outputs/smoke_test_results.json
outputs/smoke_test_results.csv
```

Run a full Wikipedia RAG condition:

```bash
python -m src.experiment_runner --scenarios data/evaluation_scenarios.xlsx --rag_mode full_wikipedia --top_k 5
```

Key options:

```text
--agent_type truth|deceptive|all
--rag_mode none|full_wikipedia|all
--top_k 5
--retriever_type dpr|sentence_transformer
--judge_gets_evidence true|false
--corpus_snapshot "snapshot-name"
--corpus_path data/wikipedia_corpus_sample.jsonl
--allow_fallback
--mock_mode true
```

The retriever first attempts `facebook/wiki_dpr` with DPR embeddings. If that
is unavailable in the local Python/Hugging Face environment, the run fails
unless `--allow_fallback` is explicitly passed. When fallback is allowed, every
JSON/CSV output records `fallback_used=true` and the exact fallback reason. No
live Wikipedia search is used.

The normal experiment runner also requires a real local Ollama model. Use
`--mock_mode true` only for explicit demo/smoke work where mock generations are
acceptable.

If FEVER falls back with `Feature type 'List' not found`, upgrade the Python
environment used by Streamlit:

```bash
python -m pip install --upgrade -r requirements.txt
```

For thesis Wikipedia RAG experiments, FEVER remains the primary claim source:

```text
data/evaluation_scenarios.xlsx  # experiment settings; dataset_mode stays "FEVER sample"
data/fever_claims.jsonl         # local FEVER-compatible export for reproducible runs
data/claims.xlsx                # optional smoke/debug claims only
```

The thesis architecture is FEVER claims and FEVER gold labels, with Wikipedia
DPR retrieved evidence used only in RAG conditions. `data/claims.xlsx` is not
the thesis dataset path.

## Optional Environment File

Copy `.env.example` to `.env` if you want to set a default Ollama model:

```text
OLLAMA_MODEL=llama3.2:3b
```
