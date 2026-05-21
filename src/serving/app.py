"""FastAPI serving layer"""

from fastapi import FastAPI

app = FastAPI(title="AgenticRAG", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(request: dict):
    """Submit a query to the multi-agent pipeline."""
    raise NotImplementedError


@app.get("/metrics")
def metrics():
    """Return system metrics."""
    raise NotImplementedError
