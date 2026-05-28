# 🧠 BRAIN — Bidirectional Reasoning Architecture for Intelligence Networks

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg)
![GUDHI](https://img.shields.io/badge/GUDHI-3.9.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

**BRAIN (Bidirectional Reasoning Architecture for Intelligence Networks)** is a deterministic, topology-driven AI architecture designed to address one of the most critical limitations of Large Language Models (LLMs): **retrieval-induced hallucination**.

Unlike conventional Retrieval-Augmented Generation (RAG) systems that rely on probabilistic token prediction and vector similarity, BRAIN introduces a mathematically verifiable reasoning framework using **Topological Data Analysis (TDA)** and **Persistent Homology**.

The system separates:

* **Syntactic extraction** → handled by language models and biomedical parsers
* **Causal verification** → handled by deterministic geometric reasoning

Instead of “guessing” relationships between concepts, BRAIN validates them through explicit topological structures and measurable evidence gaps.

---

# 🚨 The Problem: Retrieval-Induced Hallucination

Traditional AI pipelines retrieve semantically similar text chunks and rely on an LLM to synthesize answers. However, LLMs fundamentally operate as probabilistic sequence models and lack a formal mechanism for truth verification.

This frequently causes:

* Fabricated causal relationships
* Unsupported medical conclusions
* False reasoning bridges between disconnected facts
* Overconfident responses despite missing evidence

### Example

Query:

> “Does Metformin interact with Atorvastatin?”

A standard LLM may confidently answer “Yes” or “No” based primarily on training priors and token probabilities, even if the retrieved evidence is incomplete or contradictory.

BRAIN avoids this failure mode entirely.

---

# 📐 The Solution: Topological Medical Reasoning

BRAIN transforms biomedical knowledge into a mathematically analyzable geometric structure.

## Core Pipeline

### 1. Syntactic Extraction

Biomedical Subject–Verb–Object (SVO) relationships are extracted from PubMed XML datasets using `sciSpaCy`.

Example:

```text
(Metformin) → (interacts_with) → (Atorvastatin)
```

---

### 2. Topological Ingestion

Extracted relationships are mapped into a `gudhi.SimplexTree`.

| Mathematical Object | Semantic Meaning           |
| ------------------- | -------------------------- |
| 0-simplex           | Entity / Concept           |
| 1-simplex           | Direct Interaction         |
| 2-simplex           | Higher-order Triangulation |
| Betti Numbers       | Measurable Knowledge Gaps  |

This creates a persistent geometric knowledge space rather than a probabilistic embedding space.

---

### 3. Deterministic Verification

When a query is issued:

* If a verified causal edge exists → `VERIFIED`
* If entities exist without causal linkage → `EPISTEMIC_VOID`
* If conflicting evidence exists → `PARTIAL`
* If entities are unknown → `UNKNOWN_ENTITIES`

The system never fabricates missing relationships.

---

### 4. Bottom-Up Evidence Expansion

When an epistemic void is detected, BRAIN autonomously:

1. Searches live PubMed sources
2. Extracts additional biomedical evidence
3. Updates the simplicial topology
4. Re-runs causal verification

This enables adaptive, evidence-driven reasoning.

---

# 🏗 Architecture

## Backend Stack

### FastAPI

Persistent backend architecture holding singleton model instances to eliminate repeated cold-start overhead.

### LangGraph

Bidirectional orchestration engine responsible for:

* State management
* Tool routing
* Web retrieval
* Evidence synthesis
* Workflow control

### GUDHI

C++-accelerated computational topology engine used for:

* Simplicial complex construction
* Persistent homology
* Betti number computation
* Topological verification

### sciSpaCy (`en_core_sci_sm`)

Biomedical NLP layer providing deterministic:

* Named Entity Recognition (NER)
* Dependency parsing
* Relationship extraction

### LiteLLM / Groq

Modular inference routing for:

* Lightweight reasoning
* Syntactic parsing
* Response generation

Optimized for:

```text
llama-3.1-8b-instant
```

### Streamlit

Real-time visualization dashboard and interactive reasoning interface.

---

# 📊 Mathematical Reasoning Metrics

The dashboard exposes the internal topological state of the system in real time.

## Betti-1 Holes

Represents measurable one-dimensional topological voids in the knowledge graph.

These correspond to:

* Missing causal pathways
* Incomplete evidence chains
* Unverified interaction regions

---

## System Status Definitions

| Status             | Meaning                                           |
| ------------------ | ------------------------------------------------- |
| `VERIFIED`         | A direct geometric causal path exists             |
| `PARTIAL`          | Conflicting or uncertain causal evidence detected |
| `EPISTEMIC_VOID`   | Entities exist without verified causal geometry   |
| `UNKNOWN_ENTITIES` | Query entities are outside known vocabulary       |

---

# ⚙️ Installation

## 1. Environment Setup

> **Note:** `gudhi` depends heavily on native C++ compilation. Running under WSL2 or Linux is strongly recommended.

```bash
# Create virtual environment
python3 -m venv brain_env

# Activate environment
source brain_env/bin/activate

# Install dependencies
pip install fastapi uvicorn streamlit requests pydantic
pip install langgraph langchain-core litellm biopython gudhi
pip install spacy==3.7.6 scispacy==0.5.5 click

# Install sciSpaCy biomedical model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

---

## 2. Configuration

Export API credentials for inference routing and optional PubMed enrichment.

```bash
export GROQ_API_KEY="gsk_your_api_key_here"

# Optional: Live biomedical retrieval
export NCBI_EMAIL="your_email@example.com"
```

---

## 3. Running the Stack

```bash
chmod +x start.sh
./start.sh
```

Access the application at:

```text
http://localhost:8501
```

---

# 🧪 Current Limitations (MVP)

## 1. Undirected Simplicial Structures

`gudhi.SimplexTree` natively models undirected topology.

The current implementation can verify that relationships exist but requires fallback SVO parsing to infer directional causality.

---

## 2. Simplex Explosion

Dense higher-order biomedical interactions generate combinatorial growth in simplicial volumes.

To preserve memory stability:

* Native computation is currently capped at 2-simplices
* Higher-order interaction compression is under active exploration

---

## 3. Biomedical Scope Constraints

The current pipeline is optimized primarily for:

* Drug interaction analysis
* Biomedical causal reasoning
* PubMed-derived knowledge extraction

Cross-domain reasoning remains experimental.

---

# 🔬 Research Direction

Future work includes:

* Directed simplicial complexes
* Dynamic persistent homology updates
* Temporal reasoning layers
* Uncertainty-weighted topology
* Real-time distributed topology computation
* Semantic compression for large-scale knowledge persistence

---

# 📄 License

Released under the MIT License.
