"""Router Agent — Classifies queries and routes to specialist agents"""


def route(query: str) -> dict:
    """Classify query type and return routing decision."""
    raise NotImplementedError
