# Agents E2E - Docs-and-Actions Agent

A reliable "Docs-and-Actions Agent" that:
1. Answers questions grounded on a small local corpus
2. Can call tools (calc, calendar mock, web search, simple file ops)
3. Exposes traces, evals, and guardrails

## Stack

- **Orchestration:** LangGraph (graph + state)
- **Model & tools:** OpenAI Responses API + Agents SDK
- **Vector store:** FAISS (local)
- **Tests & evals:** pytest + scenario suite (YAML) + cost/latency logging

## Setup

1. **Activate conda environment:**
   ```bash
   conda activate agents
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. **Add corpus documents:**
   - Place PDFs/markdown files in `data/corpus/`

5. **Ingest corpus:**
   ```bash
   python -m src.rag.ingest
   ```

6. **Run the agent:**
   ```bash
   python -m src.app "your question here"
   ```

## Project Structure

```
agents-e2e/
  pyproject.toml
  .env.example
  data/
    corpus/              # PDFs/markdown you'll ground on
    index/               # FAISS index and metadata
  src/
    app.py              # CLI (and simple FastAPI) entry
    config.py
    tools/
      calculator.py
      calendar_mock.py
      file_ops.py
      web_search.py
    graph/
      state.py          # Typed state (messages, scratchpad, citations)
      nodes.py          # router, rag_node, tool_node, finalize_node
      build_graph.py
    rag/
      ingest.py         # chunk + embed + index
      retriever.py
    evals/
      scenarios.yaml    # golden tasks
      run_evals.py      # runs suite, computes metrics
    guards/
      policy.py         # refuse patterns, pii mask, url allowlist
    observability/
      telemetry.py      # latency, token, tool success logs
  tests/
    test_tools.py
    test_rag.py
    test_flows.py
  README.md
```

## Milestones

- [x] Project setup
- [ ] Milestone 1: RAG answers from corpus (no tools)
- [ ] Milestone 2: Tool use works
- [ ] Milestone 3: Reliability & guardrails
- [ ] Milestone 4: Observability & cost

## Testing

Run tests:
```bash
pytest
```

Run eval scenarios:
```bash
python -m src.evals.run_evals
```

## Capabilities & Limits

_(To be documented after implementation)_

