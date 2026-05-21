# Sample Document: RAG vs Fine-Tuning

## Overview

Retrieval-Augmented Generation (RAG) and fine-tuning are two fundamental approaches for adapting large language models (LLMs) to domain-specific tasks. Each has distinct tradeoffs in terms of cost, latency, accuracy, and maintenance.

## RAG (Retrieval-Augmented Generation)

RAG combines a retrieval system with a generative model. When a query arrives, the system first retrieves relevant documents from a knowledge base, then passes them as context to the LLM for generation.

**Advantages:**
- No retraining required — update the knowledge base, not the model
- Lower cost — no GPU training time needed
- Real-time knowledge updates
- Full traceability — every claim can be traced to a source document
- Smaller model can be used (cheaper inference)

**Disadvantages:**
- Retrieval quality directly impacts output quality
- Latency overhead from retrieval step
- Context window limits how much knowledge can be injected
- Retrieval can miss relevant documents (recall problem)

## Fine-Tuning

Fine-tuning updates the model's weights on domain-specific data, baking knowledge directly into the model.

**Advantages:**
- No retrieval latency at inference time
- Better performance on domain-specific language patterns
- No context window limitations for learned knowledge
- Can improve reasoning on domain-specific tasks

**Disadvantages:**
- Expensive — requires GPU training time
- Knowledge is frozen at training time — updates require retraining
- No traceability — can't trace claims to sources
- Risk of catastrophic forgetting
- Requires large amounts of training data

## When to Use Which

| Factor | RAG | Fine-Tuning |
|--------|-----|-------------|
| Knowledge freshness | Real-time | Frozen at training |
| Cost | Lower | Higher |
| Latency | Higher (retrieval) | Lower |
| Traceability | Full | None |
| Domain language | Moderate improvement | Strong improvement |
| Data required | Documents | Labeled examples |

## Hybrid Approach

In practice, the best systems combine both: fine-tune the model for domain language understanding, then use RAG for knowledge injection and traceability. This gives you the best of both worlds — domain expertise from training and up-to-date knowledge from retrieval.
