# Multi-Agent RAG System with Governance

> A production-grade multi-agent RAG system — specialized agents for retrieval, reasoning, validation, and governance working together through LangGraph orchestration.

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│Router Agent │ ──→ Classifies query type (factual / analytical / creative)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│      Specialist Agent Pool       │
│  ┌──────────┐  ┌──────────────┐ │
│  │Retrieval │  │  Reasoning   │ │
│  │  Agent   │  │   Agent      │ │
│  └──────────┘  └──────────────┘ │
│  ┌──────────┐  ┌──────────────┐ │
│  │Citation  │  │  Validation  │ │
│  │  Agent   │  │   Agent      │ │
│  └──────────┘  └──────────────┘ │
└─────────────────────────────────┘
       │
       ▼
┌─────────────┐
│ Aggregator  │ ──→ Combines agent outputs, resolves conflicts
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Guardrail   │ ──→ Factuality + safety + compliance check
└──────┬──────┘
       │
       ▼
Final Response + Citations
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Orchestration | LangGraph |
| Vector DB | ChromaDB |
| LLM | OpenRouter |
| Reranking | Cross-encoder |
| Serving | FastAPI |
| Frontend | Streamlit |
| Testing | pytest |

## Quick Start

```bash
git clone https://github.com/Harisathwik/AgenticRAG.git
cd AgenticRAG
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Set OpenRouter API key
cp .env.example .env
# Edit .env with your key

# Ingest documents
python -m src.retrieval.ingest --dir data/sample_docs

# Run API
uvicorn src.serving.app:app --reload --port 8000

# Open dashboard
streamlit run src/dashboard/app.py
```

## Project Structure

```
AgenticRAG/
├── src/
│   ├── agents/           # 7 specialized agents
│   ├── orchestrator/     # LangGraph state machine
│   ├── retrieval/        # Embedding, vector store, reranking
│   ├── serving/          # FastAPI application
│   ├── evaluation/       # Faithfulness, recall, latency metrics
│   └── dashboard/        # Streamlit monitoring UI
├── tests/
├── configs/
├── data/sample_docs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## License

MIT

## Author

**Harisathwik Veerla** — AI Engineer specializing in agentic systems, RAG architectures, and LLMOps.

- LinkedIn: https://www.linkedin.com/in/harisathwik-veerla/
- GitHub: https://github.com/Harisathwik
- Portfolio: https://harisathwik.github.io/
