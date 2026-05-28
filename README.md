# 🧠 BRAIN: Bidirectional Reasoning Architecture for Intelligence Networks

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg)
![GUDHI](https://img.shields.io/badge/GUDHI-3.9.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**BRAIN** is a deterministic, memory-safe AI architecture designed to solve Retrieval-Induced Hallucination in Large Language Models (LLMs). By decoupling syntactic extraction from causal verification, it replaces standard vector databases with a mathematically rigorous **Topological Data Analysis (TDA)** engine. 

Instead of guessing relationships based on latent token probabilities, BRAIN mathematically proves causality by traversing higher-order geometries (Simplicial Complexes) and explicitly calculating Epistemic Voids (Betti Numbers).

## 🛑 The Problem: Retrieval-Induced Hallucination
Standard AI systems (like RAG) retrieve text blocks and pass them to an LLM to formulate an answer. Because LLMs are probabilistic engines lacking a formal concept of truth, they frequently hallucinate causal bridges across disconnected islands of data. 

For example, if a standard LLM is asked *"Does Metformin interact with Atorvastatin?"*, clinical pre-training will often cause it to confidently hallucinate a "Yes" or "No" based entirely on conversational token weights, regardless of the actual retrieved data.

## 📐 The Solution: Topological Medical Reasoning
BRAIN eliminates this by treating the LLM strictly as a UI translator and syntactic parser. The actual "thinking" happens inside a geometric engine.

1. **Syntactic Extraction:** `sciSpaCy` extracts biomedical Subject-Verb-Object (SVO) relationships from raw PubMed XMLs.
2. **Topological Ingestion:** These SVO triples are mapped into a `gudhi.SimplexTree`. 
    * **Entities** = 0-simplices (Vertices)
    * **Interactions** = 1-simplices (Edges)
    * **Triangulations** = 2-simplices (Volumes)
3. **Deterministic Verification:** When a user queries the system, it queries the geometry. If the nodes exist but the causal edge does not, the system returns an `EPISTEMIC_VOID`.
4. **Bottom-Up Evidence Engine:** Upon detecting a void, the system autonomously searches the live PubMed API, extracts new facts on the fly, updates the geometric topology, and re-verifies.

---

## 🏗 Architecture & Core Technologies

* **FastAPI:** Persistent backend holding Singletons of heavy models to eliminate cold starts.
* **LangGraph:** The bidirectional state machine handling routing, web search, and synthesis.
* **GUDHI:** The C++ backend (via Python bindings) computing the Simplicial Complexes and Persistent Homology.
* **sciSpaCy (`en_core_sci_sm`):** Deterministic biomedical Named Entity Recognition (NER).
* **LiteLLM / Groq:** Modular LLM routing for synthetic parsing and final response formatting (optimized for `llama-3.1-8b-instant`).
* **Streamlit:** Real-time metrics dashboard and chat interface.

---

## 📊 Live Metrics & Math

The frontend dashboard exposes the exact mathematical state of the knowledge graph per query:

* **Betti-1 Holes:** The exact number of 1-dimensional "loops" (topological voids) in the dataset, representing mathematically proven missing data.
* **Status:** * `VERIFIED`: A hard geometric line exists between the queried entities.
  * `PARTIAL`: Conflicting causal vectors were detected (Aleatoric Uncertainty).
  * `EPISTEMIC_VOID`: Entities exist, but no causal geometry connects them.
  * `UNKNOWN_ENTITIES`: The terminology is completely out of vocabulary.

---

## ⚙️ Installation & Usage

### 1. Environment Setup
*Note: `gudhi` relies heavily on C++ compilers. Running this in WSL2 or native Linux bypasses Windows build tool issues.*

```bash
# Create and activate virtual environment
python3 -m venv brain_env
source brain_env/bin/activate

# Install Core Dependencies
pip install fastapi uvicorn streamlit requests pydantic
pip install langgraph langchain-core litellm biopython gudhi
pip install spacy==3.7.6 scispacy==0.5.5 click

# Install sciSpaCy Biomedical Model (v0.5.4 fallback)
pip install [https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz)


### 2. Configuration

The system relies on LiteLLM for routing. Export your preferred provider's API key.


export GROQ_API_KEY="gsk_your_api_key_here"
#### Optional for live web searches
export NCBI_EMAIL="your_email@example.com"


### 3. Running the Stack

Use the included startup script to boot the persistent FastAPI backend and the Streamlit frontend simultaneously.
Bash

chmod +x start.sh
./start.sh

Navigate to http://localhost:8501 to access the BRAIN interface.


## 🧪 Current Limitations (MVP State)

    The Undirected Flaw: gudhi.SimplexTree natively builds undirected geometries. Currently, the system knows that a causal link exists between A and B, but determining the directionality requires a fallback to the raw SVO extraction data.

    Simplex Explosion: Handling highly dense 5-way or 6-way drug interactions currently scales poorly in standard RAM due to combinatorial hyper-volumes. The system is capped to compute up to 2-simplices natively to protect host memory.

📄 License

MIT License.