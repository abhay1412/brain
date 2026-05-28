"""
BRAIN — backend/api.py
═════════════════════════════════════════════════════════════════════════════
FastAPI application with persistent model singletons.

On startup:
    1. Loads en_core_sci_sm (sciSpaCy) → DeterministicExtractor  (singleton)
    2. Initialises TopologicalBrain (GUDHI SimplexTree)           (singleton)
    3. Initialises LLMValidator (litellm)
    4. Compiles the LangGraph workflow

After startup the models live in RAM — every POST /query executes in
milliseconds (minus LLM network latency), with ZERO cold-start penalty.
═════════════════════════════════════════════════════════════════════════════

    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FMT = "%(asctime)s │ %(levelname)-7s │ %(name)-18s │ %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("brain.api")


# ============================================================================
# 0.  GLOBAL SINGLETONS  (populated at startup)
# ============================================================================

# These are module-level references populated in the lifespan handler.
# Using globals (not dependency injection) because sciSpaCy + GUDHI
# must be loaded exactly ONCE on a RAM-constrained host.
_workflow = None           # BrainWorkflow instance
_brain = None              # TopologicalBrain singleton
_extractor = None          # DeterministicExtractor singleton

CHECKPOINT_PATH = Path("brain_data/brain_checkpoint.pkl")


# ============================================================================
# 1.  LIFESPAN — load models once, persist until shutdown
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Runs ONCE at startup — loads sciSpaCy, GUDHI, and compiles LangGraph.
    Saves brain checkpoint at shutdown.
    """
    global _workflow, _brain, _extractor

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║          BRAIN Backend — Initialising Singletons           ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")

    t0 = time.perf_counter()

    # Import engine module (heavy imports happen here)
    from backend.engine import (
        BrainWorkflow,
        DeterministicExtractor,
        LLMValidator,
        TopologicalBrain,
    )

    # 1. sciSpaCy extractor — SINGLETON, loaded once
    _extractor = DeterministicExtractor.get_instance()

    # 2. TopologicalBrain — SINGLETON, try to resume from checkpoint
    if CHECKPOINT_PATH.exists():
        _brain = TopologicalBrain.load_from_disk(CHECKPOINT_PATH)
        TopologicalBrain._instance = _brain
    else:
        _brain = TopologicalBrain.get_instance()

    # 3. LLM validator
    validator = LLMValidator()

    # 4. LangGraph workflow
    _workflow = BrainWorkflow(
        extractor=_extractor,
        brain=_brain,
        validator=validator,
    )

    elapsed = time.perf_counter() - t0
    log.info("✅ All singletons loaded in %.1f seconds", elapsed)
    log.info("   sciSpaCy : en_core_sci_sm  (loaded once)")
    log.info("   GUDHI    : %d simplices in memory", _brain.stree.num_simplices())
    log.info("   LangGraph: compiled and ready")

    yield   # ── application runs here ──

    # Shutdown: save brain state
    log.info("Shutting down — saving brain checkpoint …")
    _brain.save_to_disk(CHECKPOINT_PATH)
    log.info("Goodbye.")


# ============================================================================
# 2.  FASTAPI APP
# ============================================================================

app = FastAPI(
    title="BRAIN API",
    description="Bidirectional Reasoning Architecture for Intelligence Networks",
    version="0.2.0",
    lifespan=lifespan,
)

# Allow Streamlit (port 8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 3.  REQUEST / RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    user_query: str = Field(..., min_length=3, description="Medical question")
    llm_model: str = Field(
        default="groq/llama-3.1-8b-instant",
        description="LiteLLM model string (e.g. groq/llama-3.1-8b-instant, ollama/mistral)",
    )


class QueryResponse(BaseModel):
    final_answer: str
    betti_1_holes: int
    topology_status: str
    search_attempted: bool
    simplices_count: int
    execution_time_ms: float
    entities_found: list[str] = []
    error: str = ""


class HealthResponse(BaseModel):
    status: str
    scispacy_loaded: bool
    simplices_count: int
    triples_ingested: int
    betti_1: int


class TopologyResponse(BaseModel):
    betti_0: int
    betti_1: int
    betti_2: int
    total_simplices: int
    num_vertices: int
    triples_ingested: int


# ============================================================================
# 4.  ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Liveness + readiness probe."""
    if _brain is None or _extractor is None:
        raise HTTPException(503, "Models not loaded yet")

    topo = _brain.compute_topology()
    return HealthResponse(
        status="ok",
        scispacy_loaded=True,
        simplices_count=topo.get("total_simplices", 0),
        triples_ingested=topo.get("triples_ingested", 0),
        betti_1=topo.get("betti_1", 0),
    )


@app.get("/topology", response_model=TopologyResponse, tags=["brain"])
async def get_topology():
    """Return current topological invariants of the SimplexTree."""
    if _brain is None:
        raise HTTPException(503, "Brain not initialised")

    topo = _brain.compute_topology()
    return TopologyResponse(
        betti_0=topo.get("betti_0", 0),
        betti_1=topo.get("betti_1", 0),
        betti_2=topo.get("betti_2", 0),
        total_simplices=topo.get("total_simplices", 0),
        num_vertices=topo.get("num_vertices", 0),
        triples_ingested=topo.get("triples_ingested", 0),
    )


@app.post("/query", response_model=QueryResponse, tags=["brain"])
async def query_brain(req: QueryRequest):
    """
    Main endpoint: run a medical query through the full LangGraph pipeline.

    Accepts a dynamic llm_model string so the frontend dropdown can
    switch between Groq, Ollama, Gemini, etc. per request.
    """
    if _workflow is None:
        raise HTTPException(503, "Workflow not initialised — server still loading")

    log.info("POST /query  model=%s  query=%.80s…", req.llm_model, req.user_query)

    result = _workflow.run(
        user_query=req.user_query,
        llm_model=req.llm_model,
    )

    # Checkpoint the brain periodically (every 10 queries)
    triples = _brain.stats.get("triples_ingested", 0)
    if triples > 0 and triples % 10 == 0:
        _brain.save_to_disk(CHECKPOINT_PATH)

    return QueryResponse(
        final_answer=result.get("final_answer", ""),
        betti_1_holes=result.get("betti_1_holes", 0),
        topology_status=result.get("topology_status", "UNKNOWN"),
        search_attempted=result.get("search_attempted", False),
        simplices_count=result.get("simplices_count", 0),
        execution_time_ms=result.get("execution_time_ms", 0.0),
        entities_found=result.get("entities", []),
        error=result.get("error", ""),
    )


@app.post("/save", tags=["system"])
async def save_checkpoint():
    """Manually trigger a brain checkpoint save."""
    if _brain is None:
        raise HTTPException(503, "Brain not initialised")
    _brain.save_to_disk(CHECKPOINT_PATH)
    return {"status": "saved", "path": str(CHECKPOINT_PATH)}
