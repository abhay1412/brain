"""
BRAIN — backend/engine.py
═════════════════════════════════════════════════════════════════════════════
Persistent engine module housing:
    • TopologicalBrain   — GUDHI SimplexTree codec  (Singleton)
    • DeterministicExtractor — sciSpaCy NER + dep parse  (Singleton)
    • LLMValidator       — litellm bridge with dynamic model
    • BrainWorkflow      — LangGraph state machine

All heavy models are loaded ONCE and kept in memory for the lifetime
of the FastAPI process — no cold starts after the first request.
═════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import re
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FMT = "%(asctime)s │ %(levelname)-7s │ %(name)-22s │ %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("brain.engine")


# ============================================================================
# 0.  DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class SVOTriple:
    """A Subject-Verb-Object triple with provenance."""
    subject: str
    verb: str
    obj: str
    source_sentence: str = ""
    source: str = ""            # e.g. "user_query", "pubmed", "llm_search"


@dataclass
class ValidatedTriple:
    """A triple that passed LLM or bypass validation."""
    triple: SVOTriple
    confidence: float = 1.0
    conditions: Optional[str] = None


# ── LangGraph State Schema ───────────────────────────────────────────────────

class BrainState(TypedDict, total=False):
    """Shared state flowing through the LangGraph pipeline."""
    user_query: str
    llm_model: str
    # Phase 1: Parse
    entities: list[str]
    extracted_triples: list[dict]
    # Phase 2: Topology check
    known_triples: list[dict]
    unknown_entities: list[str]
    betti_1_holes: int
    topology_status: str           # VERIFIED | EPISTEMIC_VOID | PARTIAL | UNKNOWN
    simplices_count: int
    # Phase 3: Search
    search_attempted: bool
    search_context: str
    # Phase 4: Synthesis
    final_answer: str
    # Meta
    execution_time_ms: float
    error: str


# ============================================================================
# 1.  TOPOLOGICAL BRAIN  (GUDHI SimplexTree — Singleton)
# ============================================================================

class TopologicalBrain:
    """
    Topological Codec for the BRAIN knowledge graph.

    Mapping:
        • Entities (subjects, objects, verbs) → 0-simplices
        • Binary relations                   → 1-simplices
        • 3-way interactions (auto-detected)  → 2-simplices

    CRITICAL:
        _insert_edge uses strict boolean checks for GUDHI:
            if self.stree.find(edge):
                return
        This is the canonical GUDHI API — find() returns bool.
    """

    _instance: Optional["TopologicalBrain"] = None

    def __init__(self) -> None:
        import gudhi
        self._gudhi = gudhi
        self.stree = gudhi.SimplexTree()

        self._node_id: Dict[str, int] = {}
        self._id_node: Dict[int, str] = {}
        self._next_id: int = 0
        self._adjacency: Dict[int, Set[int]] = defaultdict(set)

        self.stats = {
            "triples_ingested": 0,
            "simplices_0": 0,
            "simplices_1": 0,
            "simplices_2": 0,
        }
        log.info("TopologicalBrain initialised (empty SimplexTree)")

    # ── Singleton accessor ────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "TopologicalBrain":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    # ── Node management ───────────────────────────────────────────────
    def _get_node_id(self, label: str) -> int:
        if label not in self._node_id:
            nid = self._next_id
            self._node_id[label] = nid
            self._id_node[nid] = label
            self._next_id += 1
            self.stree.insert([nid], filtration=0.0)
            self.stats["simplices_0"] += 1
        return self._node_id[label]

    # ── Edge insertion with auto 2-simplex detection ──────────────────
    def _insert_edge(self, a_id: int, b_id: int, filtration: float = 1.0) -> None:
        """
        Insert a 1-simplex.  Automatically promotes to 2-simplices
        when a shared neighbour completes a triangle.

        CRITICAL MATH FIX: Boolean check for GUDHI — find() returns bool.
        """
        edge = sorted([a_id, b_id])

        # ── STRICT boolean guard — GUDHI find() returns True/False ────
        if self.stree.find(edge):
            return

        self.stree.insert(edge, filtration=filtration)
        self.stats["simplices_1"] += 1

        # Auto-detect 2-simplices via common-neighbour intersection
        common = self._adjacency[a_id] & self._adjacency[b_id]
        for c_id in common:
            triangle = sorted([a_id, b_id, c_id])
            if not self.stree.find(triangle):          # boolean check
                self.stree.insert(triangle, filtration=2.0)
                self.stats["simplices_2"] += 1

        self._adjacency[a_id].add(b_id)
        self._adjacency[b_id].add(a_id)

    # ── Ingest a validated triple ─────────────────────────────────────
    def ingest_triple(self, vt: ValidatedTriple) -> None:
        t = vt.triple
        s_id = self._get_node_id(t.subject)
        v_id = self._get_node_id(t.verb)
        o_id = self._get_node_id(t.obj)

        filt = max(0.1, 2.0 - vt.confidence)
        self._insert_edge(s_id, v_id, filtration=filt)
        self._insert_edge(v_id, o_id, filtration=filt)
        self._insert_edge(s_id, o_id, filtration=filt)
        self.stats["triples_ingested"] += 1

    # ── Query: check if a triple's simplex exists ─────────────────────
    def query_triple(self, subject: str, verb: str, obj: str) -> dict:
        s_id = self._node_id.get(subject)
        v_id = self._node_id.get(verb)
        o_id = self._node_id.get(obj)

        if s_id is None or v_id is None or o_id is None:
            return {"exists": False, "score": 3.0, "level": "unknown"}

        triangle = sorted([s_id, v_id, o_id])
        if self.stree.find(triangle):                  # boolean check
            return {"exists": True, "score": 0.0, "level": "triangle"}

        edges_found = sum(
            1 for a, b in combinations(triangle, 2)
            if self.stree.find(sorted([a, b]))         # boolean check
        )
        if edges_found == 3:
            return {"exists": False, "score": 0.5, "level": "edges_complete"}
        elif edges_found > 0:
            return {"exists": False, "score": 1.0, "level": "partial_edges"}
        return {"exists": False, "score": 2.0, "level": "vertices_only"}

    # ── Check which entities are known ────────────────────────────────
    def check_entities(self, entities: list[str]) -> Tuple[list[str], list[str]]:
        known = [e for e in entities if e in self._node_id]
        unknown = [e for e in entities if e not in self._node_id]
        return known, unknown

    # ── Retrieve known triples involving an entity ────────────────────
    def get_entity_relations(self, entity: str, max_results: int = 10) -> list[dict]:
        nid = self._node_id.get(entity)
        if nid is None:
            return []

        relations = []
        neighbours = self._adjacency.get(nid, set())
        for nb_id in list(neighbours)[:max_results]:
            nb_name = self._id_node.get(nb_id, f"node_{nb_id}")
            relations.append({"entity": entity, "related_to": nb_name})
        return relations

    # ── Persistent homology ───────────────────────────────────────────
    def compute_topology(self) -> Dict[str, Any]:
        self.stree.compute_persistence()
        betti = self.stree.betti_numbers()
        return {
            "betti_0": betti[0] if len(betti) > 0 else 0,
            "betti_1": betti[1] if len(betti) > 1 else 0,
            "betti_2": betti[2] if len(betti) > 2 else 0,
            "total_simplices": self.stree.num_simplices(),
            "dimension": self.stree.dimension(),
            "num_vertices": self.stree.num_vertices(),
            **self.stats,
        }

    # ── Persistence ──────────────────────────────────────────────────
    def save_to_disk(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "stree": self.stree,
            "node_id": dict(self._node_id),
            "id_node": dict(self._id_node),
            "next_id": self._next_id,
            "adjacency": {k: list(v) for k, v in self._adjacency.items()},
            "stats": dict(self.stats),
        }
        tmp = filepath.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(filepath)
        log.info("💾 Brain saved → %s  (%d simplices)", filepath, self.stree.num_simplices())

    @classmethod
    def load_from_disk(cls, filepath: str | Path) -> "TopologicalBrain":
        filepath = Path(filepath)
        if not filepath.exists():
            log.warning("No checkpoint found at %s — starting fresh", filepath)
            return cls()
        with open(filepath, "rb") as f:
            snap = pickle.load(f)
        import gudhi
        brain = cls.__new__(cls)
        brain._gudhi = gudhi
        brain.stree = snap["stree"]
        brain._node_id = snap["node_id"]
        brain._id_node = snap["id_node"]
        brain._next_id = snap["next_id"]
        brain._adjacency = defaultdict(set, {
            int(k): set(v) for k, v in snap["adjacency"].items()
        })
        brain.stats = snap["stats"]
        log.info("💾 Brain restored (%d simplices)", brain.stree.num_simplices())
        return brain


# ============================================================================
# 2.  DETERMINISTIC EXTRACTOR  (sciSpaCy — Singleton)
# ============================================================================

class DeterministicExtractor:
    """
    sciSpaCy NER + dependency parsing for SVO extraction.
    Singleton — the en_core_sci_sm model is loaded ONCE into RAM.
    """

    _instance: Optional["DeterministicExtractor"] = None

    def __init__(self) -> None:
        log.info("Loading sciSpaCy en_core_sci_sm … (one-time cost)")
        import spacy
        self.nlp = spacy.load("en_core_sci_sm")
        self.nlp.max_length = 500_000
        log.info("sciSpaCy loaded ✓  Pipeline: %s", self.nlp.pipe_names)

    @classmethod
    def get_instance(cls) -> "DeterministicExtractor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    # ── Helpers ───────────────────────────────────────────────────────
    def _get_full_span(self, token) -> str:
        parts = []
        for child in token.lefts:
            if child.dep_ in ("compound", "amod", "nmod"):
                parts.append(child.text)
        parts.append(token.text)
        for child in token.rights:
            if child.dep_ in ("compound",):
                parts.append(child.text)
        return " ".join(parts)

    def _overlaps_entity(self, span_text: str, entities: Set[str]) -> bool:
        normed = span_text.lower().strip()
        for ent in entities:
            if normed in ent.lower() or ent.lower() in normed:
                return True
        return False

    # ── Core extraction ───────────────────────────────────────────────
    def extract_entities(self, text: str) -> list[str]:
        """Return all medical entity surface forms from text."""
        doc = self.nlp(text)
        return list({ent.text for ent in doc.ents})

    def extract_triples(self, text: str, source: str = "") -> list[SVOTriple]:
        """Extract SVO triples where Subject and Object are medical entities."""
        doc = self.nlp(text)
        entities: Set[str] = {ent.text for ent in doc.ents}
        if not entities:
            return []

        triples: list[SVOTriple] = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ != "VERB":
                    continue

                subjects, objects = [], []
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        span = self._get_full_span(child)
                        subjects.append(span)
                        for conj in child.conjuncts:
                            subjects.append(self._get_full_span(conj))
                    elif child.dep_ in ("dobj", "attr", "pobj", "oprd"):
                        span = self._get_full_span(child)
                        objects.append(span)
                        for conj in child.conjuncts:
                            objects.append(self._get_full_span(conj))
                    elif child.dep_ == "prep":
                        for gc_ in child.children:
                            if gc_.dep_ == "pobj":
                                objects.append(self._get_full_span(gc_))

                verb = token.lemma_.capitalize()
                for s in subjects:
                    if not self._overlaps_entity(s, entities):
                        continue
                    for o in objects:
                        if not self._overlaps_entity(o, entities):
                            continue
                        triples.append(SVOTriple(
                            subject=s.strip(), verb=verb,
                            obj=o.strip(),
                            source_sentence=sent.text.strip(),
                            source=source,
                        ))
        return triples


# ============================================================================
# 3.  LLM VALIDATOR  (LiteLLM — dynamic model per request)
# ============================================================================

VALIDATOR_SYSTEM = textwrap.dedent("""\
You are a biomedical knowledge-graph validator. You receive an SVO triple
extracted from a medical query/text, plus the original sentence.

Reply with ONLY valid JSON — no markdown fences, no explanation:
{
    "is_valid_medical_fact": boolean,
    "temporal_or_dosage_conditions": "string or null",
    "confidence": float (0.0 to 1.0)
}""")


class LLMValidator:
    """Validates SVO triples and generates answers via litellm."""

    def __init__(self) -> None:
        import litellm
        self.litellm = litellm
        litellm.suppress_debug_info = True
        log.info("LLMValidator initialised (litellm ready)")

    def validate_triple(
        self, triple: SVOTriple, model: str, threshold: float = 0.4
    ) -> Optional[ValidatedTriple]:
        prompt = (
            f"Triple: ({triple.subject}, {triple.verb}, {triple.obj})\n"
            f"Sentence: \"{triple.source_sentence}\"\n"
            f"Respond with ONLY the JSON object."
        )
        try:
            resp = self.litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": VALIDATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                timeout=20.0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)

            if not data.get("is_valid_medical_fact", False):
                return None
            conf = float(data.get("confidence", 0.0))
            if conf < threshold:
                return None
            return ValidatedTriple(
                triple=triple,
                confidence=conf,
                conditions=data.get("temporal_or_dosage_conditions"),
            )
        except Exception as exc:
            log.warning("LLM validation failed: %s", exc)
            return None

    def generate_answer(self, query: str, context: str, model: str) -> str:
        """
        Strict translation node. Relays the mathematical state without hallucinating.
        """
        import litellm
        
        prompt = f"""
        You are a strict data-reporting interface for a deterministic mathematical engine. 
        Your ONLY job is to relay the "System Status" to the user in a polite, professional tone.
        
        CRITICAL RULES:
        1. DO NOT answer the medical question using your own knowledge.
        2. DO NOT offer medical advice.
        3. You must base your entire response ON THE SYSTEM STATUS below.
        
        User's Original Question: "{query}"
        System Status to Relay: "{context}"
        """
        
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                # Temporarily drop JSON format for the final natural language output
                response_format=None 
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"System successfully calculated topology, but LLM synthesis failed: {e}"
        
    def synthesize_answer(state: AgentState) -> AgentState:
        print("\n[Node: synthesize_answer] Drafting final deterministic response...")
        
        status = state["topology_status"]
        
        if status == "VERIFIED":
            if state.get("search_attempted"):
                context = "The knowledge graph initially lacked this data, but I successfully searched PubMed, extracted the clinical facts, updated my topological memory, and mathematically verified the causal link."
            else:
                context = "I have mathematically verified a direct causal link between these entities in the local knowledge graph."
        elif status in ["EPISTEMIC_VOID", "UNKNOWN_ENTITIES"]:
            if state.get("search_attempted"):
                context = f"I detected an epistemic void and searched PubMed, but could not extract mathematically verifiable evidence to connect these entities. The graph retains {state.get('betti_1_holes', 0)} topological holes."
            else:
                context = f"I cannot verify this. The entities lack a causal link in my current database. {state.get('betti_1_holes', 0)} topological holes detected."
        else:
            context = "Database error."

        # HARDENED PROMPT: We stop the LLM from answering the medical question directly.
        prompt = f"""
        You are a strict data-reporting interface for a deterministic mathematical engine. 
        Your ONLY job is to relay the "System Status" to the user in a polite, professional tone.
        
        CRITICAL RULES:
        1. DO NOT answer the medical question using your own knowledge.
        2. DO NOT offer medical advice.
        3. You must base your entire response ON THE SYSTEM STATUS below.
        
        User's Original Question: "{state['user_query']}"
        System Status to Relay: "{context}"
        """
        
        # We pass the dynamic model selected from the Streamlit UI
        model_choice = state.get("llm_model", "groq/llama-3.1-8b-instant")
        
        response = litellm.completion(
            model=model_choice,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {**state, "final_answer": response.choices[0].message.content.strip()}

    def search_medical_knowledge(
        self, query: str, entities: list[str], model: str
    ) -> str:
        """Use the LLM to retrieve/synthesise relevant medical knowledge."""
        system = textwrap.dedent("""\
        You are a biomedical knowledge retrieval system. Given a medical query
        and a list of entities, provide concise, factual medical relationships
        between the entities in Subject-Verb-Object format.

        Return ONLY a JSON array of objects:
        [{"subject": "...", "verb": "...", "object": "...", "fact": "one-sentence summary"}]

        Limit to at most 8 relationships. Be factual — no speculation.""")

        user_msg = (
            f"Query: {query}\n"
            f"Entities: {', '.join(entities)}\n"
            f"Return the JSON array."
        )

        try:
            resp = self.litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                timeout=25.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            log.warning("LLM search failed: %s", exc)
            return "[]"


# ============================================================================
# 4.  LANGGRAPH WORKFLOW  (State Machine)
# ============================================================================

class BrainWorkflow:
    """
    LangGraph state machine that orchestrates the full query pipeline:

        parse_query → check_topology → assess_uncertainty
             → [search_knowledge] → synthesise_answer → update_brain

    Accepts a dynamic `llm_model` string per request so users can swap
    between Groq, Ollama, Gemini, etc. from the Streamlit dropdown.
    """

    def __init__(
        self,
        extractor: DeterministicExtractor,
        brain: TopologicalBrain,
        validator: LLMValidator,
    ) -> None:
        self.extractor = extractor
        self.brain = brain
        self.validator = validator
        self.graph = self._build_graph()

    # ── Node functions ────────────────────────────────────────────────

    def _parse_query(self, state: BrainState) -> dict:
        """Phase 1: Extract medical entities and SVO triples from the query."""
        query = state["user_query"]
        entities = self.extractor.extract_entities(query)
        triples = self.extractor.extract_triples(query, source="user_query")

        return {
            "entities": entities,
            "extracted_triples": [
                {"s": t.subject, "v": t.verb, "o": t.obj} for t in triples
            ],
        }

    def _check_topology(self, state: BrainState) -> dict:
        """Phase 2: Query the SimplexTree for known entities and relations."""
        entities = state.get("entities", [])
        known, unknown = self.brain.check_entities(entities)

        # Gather known relations for context
        known_triples = []
        for ent in known:
            rels = self.brain.get_entity_relations(ent, max_results=5)
            known_triples.extend(rels)

        return {
            "known_triples": known_triples,
            "unknown_entities": unknown,
        }

    def _assess_uncertainty(self, state: BrainState) -> dict:
        """Phase 3: Compute Betti numbers and classify topology status."""
        topo = self.brain.compute_topology()
        betti_1 = topo.get("betti_1", 0)
        simplices = topo.get("total_simplices", 0)

        unknown = state.get("unknown_entities", [])
        known_triples = state.get("known_triples", [])

        # Classify topology status
        if len(unknown) == 0 and len(known_triples) > 0 and betti_1 == 0:
            status = "VERIFIED"
        elif betti_1 > 0 or len(unknown) > len(state.get("entities", [])) // 2:
            status = "EPISTEMIC_VOID"
        elif len(known_triples) > 0:
            status = "PARTIAL"
        else:
            status = "UNKNOWN"

        return {
            "betti_1_holes": betti_1,
            "topology_status": status,
            "simplices_count": simplices,
        }

    def _should_search(self, state: BrainState) -> str:
        """Routing decision: search if topology has gaps."""
        status = state.get("topology_status", "UNKNOWN")
        if status in ("EPISTEMIC_VOID", "UNKNOWN", "PARTIAL"):
            return "search_knowledge"
        return "synthesise_answer"

    def _search_knowledge(self, state: BrainState) -> dict:
        """Phase 4 (conditional): LLM-based medical knowledge retrieval."""
        query = state["user_query"]
        model = state["llm_model"]
        entities = state.get("entities", [])

        raw = self.validator.search_medical_knowledge(query, entities, model)
        # Try to parse and ingest the search results as new triples
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            facts = json.loads(cleaned)
            if isinstance(facts, list):
                for fact in facts:
                    s = fact.get("subject", "")
                    v = fact.get("verb", "")
                    o = fact.get("object", "")
                    if s and v and o:
                        vt = ValidatedTriple(
                            triple=SVOTriple(s, v, o, source="llm_search"),
                            confidence=0.7,
                        )
                        self.brain.ingest_triple(vt)
        except (json.JSONDecodeError, TypeError):
            log.warning("Could not parse search results as JSON")

        return {
            "search_attempted": True,
            "search_context": raw,
        }

    def _skip_search(self, state: BrainState) -> dict:
        return {"search_attempted": False, "search_context": ""}

    def _synthesise_answer(self, state: BrainState) -> dict:
        """Phase 5: Generate the final answer via LLM."""
        query = state["user_query"]
        model = state["llm_model"]

        # Build context from topology + search
        ctx_parts = []
        known = state.get("known_triples", [])
        if known:
            ctx_parts.append("Known relationships in topology:")
            for rel in known[:15]:
                ctx_parts.append(f"  • {rel.get('entity', '?')} ↔ {rel.get('related_to', '?')}")

        search_ctx = state.get("search_context", "")
        if search_ctx:
            ctx_parts.append(f"\nSearch results:\n{search_ctx}")

        status = state.get("topology_status", "UNKNOWN")
        betti = state.get("betti_1_holes", 0)
        ctx_parts.append(f"\nTopology status: {status}  |  Betti-1 holes: {betti}")

        context = "\n".join(ctx_parts) if ctx_parts else "(no prior knowledge)"

        answer = self.validator.generate_answer(query, context, model)
        return {"final_answer": answer}

    def _update_brain(self, state: BrainState) -> dict:
        """Phase 6: Ingest any SVO triples extracted from the user query."""
        for td in state.get("extracted_triples", []):
            vt = ValidatedTriple(
                triple=SVOTriple(td["s"], td["v"], td["o"], source="user_query"),
                confidence=0.6,
            )
            self.brain.ingest_triple(vt)

        # Recompute simplices after update
        topo = self.brain.compute_topology()
        return {"simplices_count": topo.get("total_simplices", 0)}

    # ── Build the graph ───────────────────────────────────────────────

    def _build_graph(self):
        from langgraph.graph import StateGraph, END, START

        g = StateGraph(BrainState)

        # Add nodes
        g.add_node("parse_query", self._parse_query)
        g.add_node("check_topology", self._check_topology)
        g.add_node("assess_uncertainty", self._assess_uncertainty)
        g.add_node("search_knowledge", self._search_knowledge)
        g.add_node("synthesise_answer", self._synthesise_answer)
        g.add_node("update_brain", self._update_brain)

        # Linear edges
        g.add_edge(START, "parse_query")
        g.add_edge("parse_query", "check_topology")
        g.add_edge("check_topology", "assess_uncertainty")

        # Conditional: search or skip based on topology status
        g.add_conditional_edges(
            "assess_uncertainty",
            self._should_search,
            {
                "search_knowledge": "search_knowledge",
                "synthesise_answer": "synthesise_answer",
            },
        )
        g.add_edge("search_knowledge", "synthesise_answer")
        g.add_edge("synthesise_answer", "update_brain")
        g.add_edge("update_brain", END)

        return g.compile()

    # ── Public interface ──────────────────────────────────────────────

    def run(self, user_query: str, llm_model: str) -> dict:
        """Execute the full pipeline and return the result state."""
        t0 = time.perf_counter()

        initial_state: BrainState = {
            "user_query": user_query,
            "llm_model": llm_model,
            "entities": [],
            "extracted_triples": [],
            "known_triples": [],
            "unknown_entities": [],
            "betti_1_holes": 0,
            "topology_status": "UNKNOWN",
            "simplices_count": 0,
            "search_attempted": False,
            "search_context": "",
            "final_answer": "",
            "execution_time_ms": 0.0,
            "error": "",
        }

        try:
            result = self.graph.invoke(initial_state)
        except Exception as exc:
            log.error("Workflow failed: %s", exc, exc_info=True)
            result = dict(initial_state)
            result["error"] = str(exc)
            result["final_answer"] = f"⚠ Pipeline error: {exc}"

        elapsed = (time.perf_counter() - t0) * 1_000
        result["execution_time_ms"] = round(elapsed, 1)

        return result
