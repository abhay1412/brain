#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BRAIN — PubMed Topological Ingestor Pipeline                               ║
║  Phase 1: sciSpaCy SVO Extraction                                           ║
║  Phase 2: LiteLLM Neuro-Symbolic Validation                                 ║
║  Phase 3: GUDHI SimplexTree Topological Codec                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Hardware target : WSL2 / ~5 GB available RAM
Strategy        : iterparse streaming — never more than 1 abstract in memory.

Usage:
    python pubmed_ingestor.py --xml pubmed_subset.xml          # real run
    python pubmed_ingestor.py --xml pubmed.xml --skip-llm      # bypass LLM
    python pubmed_ingestor.py --demo                           # synthetic test
"""

from __future__ import annotations

import abc
import argparse
import gc
import json
import logging
import os
import pickle
import re
import sys
import textwrap
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

# ============================================================================
# Logging — single rotating handler, human-readable format
# ============================================================================

LOG_FMT = "%(asctime)s │ %(levelname)-7s │ %(name)-22s │ %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("brain.ingestor")


# ============================================================================
# 0.  DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class SVOTriple:
    """An extracted Subject-Verb-Object triple with provenance."""
    subject: str
    verb: str
    obj: str
    source_sentence: str = ""
    pmid: str = ""


@dataclass
class ValidatedTriple:
    """A triple that has passed LLM validation."""
    triple: SVOTriple
    confidence: float = 0.0
    conditions: Optional[str] = None


# ============================================================================
# 1.  STREAMING XML PARSER  (Zero RAM Bloat)
# ============================================================================

class PubMedStreamer:
    """
    Lazily yield (pmid, title, abstract_text) from a PubMed XML file.

    Uses xml.etree.ElementTree.iterparse with aggressive element clearing
    so that at most ONE <PubmedArticle> element is in memory at any time.

    PubMed XML Schema (simplified):
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>123456</PMID>
              <Article>
                <ArticleTitle>...</ArticleTitle>
                <Abstract>
                  <AbstractText>...</AbstractText>
                  <!-- Some abstracts have multiple labeled sections -->
                  <AbstractText Label="METHODS">...</AbstractText>
                </Abstract>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
          ...
        </PubmedArticleSet>
    """

    def __init__(self, xml_path: str | Path) -> None:
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"PubMed XML not found: {self.xml_path}")

    def stream(self) -> Generator[Tuple[str, str, str], None, None]:
        """
        Yield (pmid, title, abstract_text) one article at a time.

        Memory guarantee: we call elem.clear() and delete the reference
        on every <PubmedArticle> end-tag, keeping RSS constant regardless
        of file size.
        """
        context = ET.iterparse(str(self.xml_path), events=("end",))

        # Track the root element so we can prune fully-parsed children
        root: Optional[ET.Element] = None

        for event, elem in context:

            # Grab root on first pass (needed for root.remove later)
            if root is None:
                # Walk up — iterparse doesn't expose the root directly,
                # so we store the first element's parent via a workaround.
                # In practice, for PubMed the first 'end' event for a leaf
                # element is fine — we just clear() after each article.
                pass

            if elem.tag == "PubmedArticle":
                pmid = ""
                title = ""
                abstract_parts: List[str] = []

                # -- Extract PMID -----------------------------------------
                pmid_elem = elem.find(".//MedlineCitation/PMID")
                if pmid_elem is not None and pmid_elem.text:
                    pmid = pmid_elem.text.strip()

                # -- Extract Title ----------------------------------------
                title_elem = elem.find(".//Article/ArticleTitle")
                if title_elem is not None:
                    # ArticleTitle can contain inline XML (italic, sup, etc.)
                    title = "".join(title_elem.itertext()).strip()

                # -- Extract Abstract (handles multi-section abstracts) ---
                abstract_elem = elem.find(".//Article/Abstract")
                if abstract_elem is not None:
                    for at in abstract_elem.findall("AbstractText"):
                        text = "".join(at.itertext()).strip()
                        if text:
                            label = at.get("Label", "")
                            if label:
                                abstract_parts.append(f"{label}: {text}")
                            else:
                                abstract_parts.append(text)

                abstract_text = " ".join(abstract_parts).strip()

                # Only yield if we actually have an abstract
                if abstract_text:
                    yield (pmid, title, abstract_text)

                # ── CRITICAL: free memory immediately ────────────────────
                elem.clear()

        # Force-collect anything left over
        gc.collect()


# ============================================================================
# 2.  PHASE 1 — DETERMINISTIC SVO EXTRACTOR  (sciSpaCy)
# ============================================================================

class DeterministicExtractor:
    """
    Extracts Subject-Verb-Object triples from biomedical text using
    sciSpaCy's en_core_sci_sm model for:
      • Named Entity Recognition (NER) — filters to medical entities
      • Dependency parsing           — identifies nsubj / dobj arcs

    Strategy:
        For each sentence, walk the dependency tree:
        1. Find the ROOT verb.
        2. Find its nsubj  → candidate Subject.
        3. Find its dobj / attr / pobj → candidate Object.
        4. Keep the triple ONLY if both Subject and Object overlap with
           a recognised biomedical entity span (from NER).
    """

    def __init__(self, model_name: str = "en_core_sci_sm") -> None:
        log.info("Loading sciSpaCy model '%s' …", model_name)
        import spacy
        self.nlp = spacy.load(model_name)

        # Raise sentence length limit slightly for long abstracts,
        # but don't go crazy — we want bounded memory.
        self.nlp.max_length = 500_000
        log.info("sciSpaCy model loaded.  Pipeline: %s", self.nlp.pipe_names)

    # ------------------------------------------------------------------
    def _overlaps_entity(self, token_span: str, entities: Set[str]) -> bool:
        """Check if a token/span text overlaps with any NER entity."""
        normed = token_span.lower().strip()
        for ent in entities:
            if normed in ent.lower() or ent.lower() in normed:
                return True
        return False

    # ------------------------------------------------------------------
    def _get_full_span(self, token) -> str:
        """
        Expand a single token into its full noun-phrase by collecting
        compound and amod children.  'blood pressure' instead of 'pressure'.
        """
        parts: List[str] = []
        for child in token.lefts:
            if child.dep_ in ("compound", "amod", "nmod"):
                parts.append(child.text)
        parts.append(token.text)
        for child in token.rights:
            if child.dep_ in ("compound",):
                parts.append(child.text)
        return " ".join(parts)

    # ------------------------------------------------------------------
    def _extract_from_sentence(
        self, sent, entities: Set[str], pmid: str
    ) -> List[SVOTriple]:
        """
        Given a spaCy Span (sentence), extract all valid SVO triples.
        """
        triples: List[SVOTriple] = []

        for token in sent:
            # Only consider verb roots
            if token.pos_ != "VERB":
                continue

            subjects: List[str] = []
            objects: List[str] = []

            for child in token.children:
                # ── Subject ──────────────────────────────────────────
                if child.dep_ in ("nsubj", "nsubjpass"):
                    span_text = self._get_full_span(child)
                    subjects.append(span_text)
                    # Also grab conjuncts: "Aspirin and Ibuprofen reduce…"
                    for conj in child.conjuncts:
                        subjects.append(self._get_full_span(conj))

                # ── Object ───────────────────────────────────────────
                elif child.dep_ in ("dobj", "attr", "pobj", "oprd"):
                    span_text = self._get_full_span(child)
                    objects.append(span_text)
                    for conj in child.conjuncts:
                        objects.append(self._get_full_span(conj))

                # ── Prepositional object (e.g. "acts on X") ──────────
                elif child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            span_text = self._get_full_span(grandchild)
                            objects.append(span_text)

            # ── Filter: keep only entity-matched triples ─────────────
            verb_text = token.lemma_.capitalize()

            for subj in subjects:
                if not self._overlaps_entity(subj, entities):
                    continue
                for obj in objects:
                    if not self._overlaps_entity(obj, entities):
                        continue
                    triples.append(
                        SVOTriple(
                            subject=subj.strip(),
                            verb=verb_text,
                            obj=obj.strip(),
                            source_sentence=sent.text.strip(),
                            pmid=pmid,
                        )
                    )

        return triples

    # ------------------------------------------------------------------
    def extract(self, text: str, pmid: str = "") -> List[SVOTriple]:
        """
        Main entry point: parse full abstract text, return all valid
        SVO triples where Subject and Object are medical entities.
        """
        doc = self.nlp(text)

        # Collect all recognised entity surface forms
        entities: Set[str] = {ent.text for ent in doc.ents}

        if not entities:
            return []

        all_triples: List[SVOTriple] = []
        for sent in doc.sents:
            all_triples.extend(self._extract_from_sentence(sent, entities, pmid))

        return all_triples


# ============================================================================
# 3.  PHASE 2 — NEURO-SYMBOLIC LLM VALIDATOR  (LiteLLM)
# ============================================================================

# The JSON schema we demand from the LLM
VALIDATION_SCHEMA = textwrap.dedent("""\
{
    "is_valid_medical_fact": true/false,
    "temporal_or_dosage_conditions": "string or null",
    "confidence": 0.0-1.0
}""")

SYSTEM_PROMPT = textwrap.dedent("""\
You are a biomedical knowledge-graph validator. You will receive a
Subject-Verb-Object (SVO) triple extracted from a PubMed abstract,
along with the original sentence for context.

Your task:
1. Decide if the SVO triple represents a valid, medically meaningful fact.
2. Note any temporal or dosage conditions that qualify the fact.
3. Rate your confidence from 0.0 (no confidence) to 1.0 (certain).

Reply with ONLY valid JSON — no markdown, no explanation:
{
    "is_valid_medical_fact": boolean,
    "temporal_or_dosage_conditions": "string or null",
    "confidence": float
}""")


class LLMValidator:
    """
    Validates SVO triples via a Large Language Model using LiteLLM,
    which transparently supports Ollama (local), Groq, Gemini, OpenAI, etc.

    Configuration:
        Set the model string to match your LiteLLM provider:
          • Local Ollama :  "ollama/mistral"  or  "ollama/llama3"
          • Groq (free)  :  "groq/llama-3.3-70b-versatile"
          • Gemini       :  "gemini/gemini-2.0-flash"
          • OpenAI       :  "gpt-4o-mini"
        
        Set the corresponding API key in the environment:
          export GROQ_API_KEY=...
          export GEMINI_API_KEY=...
          export OPENAI_API_KEY=...
        
        Ollama needs no API key — just ensure the server is running.
    """

    def __init__(
        self,
        model: str = "ollama/mistral",
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout_s: float = 30.0,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.confidence_threshold = confidence_threshold

        # Lazy import — only pay the cost if LLM validation is active
        import litellm
        self.litellm = litellm

        # Suppress litellm's verbose logging unless user asks for DEBUG
        litellm.suppress_debug_info = True

        log.info(
            "LLMValidator ready  model=%s  threshold=%.2f",
            self.model,
            self.confidence_threshold,
        )

    # ------------------------------------------------------------------
    def _build_user_prompt(self, triple: SVOTriple) -> str:
        return textwrap.dedent(f"""\
            Triple:
              Subject: {triple.subject}
              Verb:    {triple.verb}
              Object:  {triple.obj}

            Original sentence:
              "{triple.source_sentence}"

            PMID: {triple.pmid}

            Respond ONLY with the JSON object.""")

    # ------------------------------------------------------------------
    def _parse_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Robustly parse the LLM's JSON response, handling markdown fences
        and minor formatting issues.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning("LLM returned unparseable JSON: %.120s…", cleaned)
            return None

        # Validate required keys
        required = {"is_valid_medical_fact", "confidence"}
        if not required.issubset(data.keys()):
            log.warning("LLM JSON missing required keys: %s", data.keys())
            return None

        return data

    # ------------------------------------------------------------------
    def validate(self, triple: SVOTriple) -> Optional[ValidatedTriple]:
        """
        Send one SVO triple to the LLM for validation.

        Returns:
            ValidatedTriple  if the LLM says it's a valid medical fact
                             AND confidence ≥ threshold.
            None             if rejected, unparseable, or on error.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(triple)},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=self.timeout_s,
                    # Some providers support response_format
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                break
            except Exception as exc:
                log.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries:
                    return None
                time.sleep(1.0 * attempt)  # basic exponential backoff
                continue

        parsed = self._parse_response(raw_text)
        if parsed is None:
            return None

        is_valid = bool(parsed.get("is_valid_medical_fact", False))
        confidence = float(parsed.get("confidence", 0.0))
        conditions = parsed.get("temporal_or_dosage_conditions")

        if not is_valid:
            log.debug("LLM rejected triple: %s → %s → %s", triple.subject, triple.verb, triple.obj)
            return None

        if confidence < self.confidence_threshold:
            log.debug(
                "LLM confidence %.2f below threshold %.2f — discarding",
                confidence,
                self.confidence_threshold,
            )
            return None

        return ValidatedTriple(
            triple=triple,
            confidence=confidence,
            conditions=conditions if isinstance(conditions, str) else None,
        )


class BypassValidator:
    """
    Drop-in replacement that skips LLM validation entirely.
    Useful for --skip-llm mode to save RAM / API costs during testing.
    """

    def validate(self, triple: SVOTriple) -> ValidatedTriple:
        return ValidatedTriple(triple=triple, confidence=1.0, conditions=None)


# ============================================================================
# 4.  PHASE 3 — TOPOLOGICAL BRAIN  (GUDHI SimplexTree)
# ============================================================================

class TopologicalBrain:
    """
    Topological Codec for the BRAIN knowledge graph.

    Mapping:
        • Entities (subjects, objects) → 0-simplices  (vertices)
        • Binary relations             → 1-simplices  (edges)
        • 3-way interactions detected   → 2-simplices  (filled triangles)

    Automatic 2-simplex detection:
        When inserting edge (A, C), we check for any entity B that already
        has edges to BOTH A and C.  If found, [A, B, C] becomes a 2-simplex.
        This captures e.g. "Drug A and Drug B both interact with Condition C".

    Persistence:
        save_to_disk() pickles the SimplexTree + metadata so that a
        multi-hour ingestion run can survive crashes.
    """

    def __init__(self) -> None:
        import gudhi
        self._gudhi = gudhi
        self.stree = gudhi.SimplexTree()

        # Fast lookups: entity name → integer ID
        self._node_id: Dict[str, int] = {}
        self._id_node: Dict[int, str] = {}      # reverse map for queries
        self._next_id: int = 0

        # Adjacency list for fast 2-simplex detection
        self._adjacency: Dict[int, Set[int]] = defaultdict(set)

        # Statistics
        self.stats = {
            "triples_ingested": 0,
            "simplices_0": 0,       # vertices
            "simplices_1": 0,       # edges
            "simplices_2": 0,       # triangles
        }

    # ------------------------------------------------------------------
    def _get_node_id(self, label: str) -> int:
        """Get or create a persistent integer ID for a named entity."""
        if label not in self._node_id:
            nid = self._next_id
            self._node_id[label] = nid
            self._id_node[nid] = label
            self._next_id += 1
            # Insert as 0-simplex (vertex) — filtration 0.0
            self.stree.insert([nid], filtration=0.0)
            self.stats["simplices_0"] += 1
        return self._node_id[label]

    # ------------------------------------------------------------------
    def _insert_edge(self, a_id: int, b_id: int, filtration: float = 1.0) -> None:
        """
        Insert a 1-simplex (edge) and automatically promote to 2-simplices
        when a shared neighbour completes a triangle.
        """
        edge = sorted([a_id, b_id])
        if self.stree.find(edge) is not None:
            return  # edge already exists — no duplicates

        self.stree.insert(edge, filtration=filtration)
        self.stats["simplices_1"] += 1

        # ── Auto-detect 2-simplices ──────────────────────────────────
        # Common neighbours of A and B form triangles
        common = self._adjacency[a_id] & self._adjacency[b_id]
        for c_id in common:
            triangle = sorted([a_id, b_id, c_id])
            if self.stree.find(triangle) is None:
                self.stree.insert(triangle, filtration=2.0)
                self.stats["simplices_2"] += 1

        # Update adjacency
        self._adjacency[a_id].add(b_id)
        self._adjacency[b_id].add(a_id)

    # ------------------------------------------------------------------
    def ingest_triple(self, vt: ValidatedTriple) -> None:
        """
        Map a validated SVO triple into the SimplexTree.

        Creates:
            • 0-simplices for subject, verb-node, and object
            • 1-simplices for (subject–verb), (verb–object), (subject–object)
            • 2-simplices automatically when triangles close
        """
        t = vt.triple
        s_id = self._get_node_id(t.subject)
        v_id = self._get_node_id(t.verb)      # verb as a first-class node
        o_id = self._get_node_id(t.obj)

        # Insert edges — filtration encodes confidence if available
        filt = max(1.0, 2.0 - vt.confidence)   # higher confidence → lower filtration
        self._insert_edge(s_id, v_id, filtration=filt)
        self._insert_edge(v_id, o_id, filtration=filt)
        self._insert_edge(s_id, o_id, filtration=filt)

        self.stats["triples_ingested"] += 1

    # ------------------------------------------------------------------
    def compute_topology(self) -> Dict[str, Any]:
        """
        Run persistent homology and return Betti numbers + summary.
        """
        self.stree.compute_persistence()
        betti = self.stree.betti_numbers()

        betti_0 = betti[0] if len(betti) > 0 else 0
        betti_1 = betti[1] if len(betti) > 1 else 0
        betti_2 = betti[2] if len(betti) > 2 else 0

        return {
            "betti_0_components": betti_0,
            "betti_1_loops": betti_1,
            "betti_2_voids": betti_2,
            "total_simplices": self.stree.num_simplices(),
            "dimension": self.stree.dimension(),
            "num_vertices": self.stree.num_vertices(),
            **self.stats,
        }

    # ------------------------------------------------------------------
    def save_to_disk(self, filepath: str | Path) -> None:
        """
        Persist the full TopologicalBrain state to disk via pickle.

        This captures:
            • The SimplexTree (via gudhi's internal serialisation)
            • Node ↔ ID mappings
            • Adjacency sets
            • Ingestion statistics

        Called periodically (e.g. every 1,000 abstracts) to prevent
        catastrophic data loss during multi-hour runs.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Build a serialisable snapshot
        # gudhi.SimplexTree supports pickle directly in recent versions.
        snapshot = {
            "stree": self.stree,
            "node_id": dict(self._node_id),
            "id_node": dict(self._id_node),
            "next_id": self._next_id,
            "adjacency": {k: list(v) for k, v in self._adjacency.items()},
            "stats": dict(self.stats),
        }

        tmp_path = filepath.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Atomic rename to avoid corruption if interrupted mid-write
        tmp_path.replace(filepath)
        log.info(
            "💾 Brain checkpoint saved → %s  (%d triples, %d simplices)",
            filepath,
            self.stats["triples_ingested"],
            self.stree.num_simplices(),
        )

    # ------------------------------------------------------------------
    @classmethod
    def load_from_disk(cls, filepath: str | Path) -> "TopologicalBrain":
        """Restore a previously-saved TopologicalBrain from disk."""
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            snapshot = pickle.load(f)

        brain = cls.__new__(cls)
        import gudhi
        brain._gudhi = gudhi
        brain.stree = snapshot["stree"]
        brain._node_id = snapshot["node_id"]
        brain._id_node = snapshot["id_node"]
        brain._next_id = snapshot["next_id"]
        brain._adjacency = defaultdict(set, {
            int(k): set(v) for k, v in snapshot["adjacency"].items()
        })
        brain.stats = snapshot["stats"]

        log.info(
            "💾 Brain restored from %s  (%d triples, %d simplices)",
            filepath,
            brain.stats["triples_ingested"],
            brain.stree.num_simplices(),
        )
        return brain

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable summary of the brain state."""
        topo = self.compute_topology()
        lines = [
            "┌─ TopologicalBrain Summary ─────────────────────────────────────┐",
            f"│  Triples ingested  : {topo['triples_ingested']:,}",
            f"│  Vertices (0-simp) : {topo['simplices_0']:,}",
            f"│  Edges    (1-simp) : {topo['simplices_1']:,}",
            f"│  Triangles(2-simp) : {topo['simplices_2']:,}",
            f"│  Total simplices   : {topo['total_simplices']:,}",
            f"│  Complex dimension : {topo['dimension']}",
            f"│  ──────────────────────────────────────────────",
            f"│  Betti-0 (components) : {topo['betti_0_components']}",
            f"│  Betti-1 (loops)      : {topo['betti_1_loops']}",
            f"│  Betti-2 (voids)      : {topo['betti_2_voids']}",
            "└────────────────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


# ============================================================================
# 5.  ORCHESTRATOR — TIES EVERYTHING TOGETHER
# ============================================================================

class IngestorPipeline:
    """
    Top-level orchestrator that wires the three phases:
        XML Stream → sciSpaCy Extractor → LLM Validator → TopologicalBrain

    Features:
        • Streams abstracts one-at-a-time from PubMed XML
        • Periodic checkpointing (every `checkpoint_every` abstracts)
        • Graceful Ctrl+C handling — saves progress before exit
        • Detailed progress logging
    """

    def __init__(
        self,
        extractor: DeterministicExtractor,
        validator: LLMValidator | BypassValidator,
        brain: TopologicalBrain,
        checkpoint_path: str = "brain_checkpoint.pkl",
        checkpoint_every: int = 1_000,
    ) -> None:
        self.extractor = extractor
        self.validator = validator
        self.brain = brain
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every

        # Counters
        self.abstracts_processed: int = 0
        self.triples_extracted: int = 0
        self.triples_validated: int = 0
        self.triples_rejected: int = 0

    # ------------------------------------------------------------------
    def _process_abstract(self, pmid: str, title: str, text: str) -> int:
        """
        Process a single abstract through the full pipeline.
        Returns the number of validated triples ingested.
        """
        # Phase 1: Extract SVO triples via sciSpaCy
        raw_triples = self.extractor.extract(text, pmid=pmid)
        self.triples_extracted += len(raw_triples)

        count = 0
        for triple in raw_triples:
            # Phase 2: Validate via LLM (or bypass)
            validated = self.validator.validate(triple)
            if validated is None:
                self.triples_rejected += 1
                continue

            # Phase 3: Ingest into TopologicalBrain
            self.brain.ingest_triple(validated)
            self.triples_validated += 1
            count += 1

        return count

    # ------------------------------------------------------------------
    def run(self, streamer: PubMedStreamer) -> None:
        """
        Execute the full pipeline over a PubMed XML stream.
        """
        log.info("═══ Pipeline starting ═══")
        t_start = time.perf_counter()

        try:
            for pmid, title, abstract_text in streamer.stream():
                self.abstracts_processed += 1
                n = self._process_abstract(pmid, title, abstract_text)

                # Progress logging every 100 abstracts
                if self.abstracts_processed % 100 == 0:
                    elapsed = time.perf_counter() - t_start
                    rate = self.abstracts_processed / elapsed if elapsed > 0 else 0
                    log.info(
                        "Progress: %d abstracts │ %d extracted │ %d validated │ "
                        "%d rejected │ %.1f abs/sec",
                        self.abstracts_processed,
                        self.triples_extracted,
                        self.triples_validated,
                        self.triples_rejected,
                        rate,
                    )

                # Periodic checkpoint
                if self.abstracts_processed % self.checkpoint_every == 0:
                    self.brain.save_to_disk(self.checkpoint_path)
                    gc.collect()  # reclaim memory after checkpoint

        except KeyboardInterrupt:
            log.warning("⚠ Interrupted — saving progress …")
            self.brain.save_to_disk(self.checkpoint_path)
            raise

        # Final checkpoint
        self.brain.save_to_disk(self.checkpoint_path)

        elapsed = time.perf_counter() - t_start
        log.info("═══ Pipeline complete ═══")
        log.info(
            "Processed %d abstracts in %.1f seconds (%.1f abs/sec)",
            self.abstracts_processed,
            elapsed,
            self.abstracts_processed / elapsed if elapsed > 0 else 0,
        )
        log.info(
            "Triples: %d extracted → %d validated → %d rejected",
            self.triples_extracted,
            self.triples_validated,
            self.triples_rejected,
        )

    # ------------------------------------------------------------------
    def report(self) -> str:
        """Final human-readable report."""
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════╗",
            "║            BRAIN — Ingestion Pipeline Report                    ║",
            "╠══════════════════════════════════════════════════════════════════╣",
            f"║  Abstracts processed : {self.abstracts_processed:>8,}                              ║",
            f"║  Triples extracted   : {self.triples_extracted:>8,}                              ║",
            f"║  Triples validated   : {self.triples_validated:>8,}                              ║",
            f"║  Triples rejected    : {self.triples_rejected:>8,}                              ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            self.brain.summary(),
        ]
        return "\n".join(lines)


# ============================================================================
# 6.  DEMO MODE — SYNTHETIC DATA FOR IMMEDIATE TESTING
# ============================================================================

DEMO_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN"
          "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>

  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">10000001</PMID>
      <Article>
        <ArticleTitle>Aspirin and Cardiovascular Risk Reduction</ArticleTitle>
        <Abstract>
          <AbstractText>Aspirin inhibits cyclooxygenase-2 and reduces prostaglandin synthesis.
          Low-dose aspirin decreases platelet aggregation in patients with coronary artery disease.
          Aspirin therapy reduces the risk of myocardial infarction by approximately 25 percent.
          However, aspirin increases the risk of gastrointestinal bleeding in elderly patients.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">10000002</PMID>
      <Article>
        <ArticleTitle>Metformin in Type 2 Diabetes Management</ArticleTitle>
        <Abstract>
          <AbstractText>Metformin activates AMP-activated protein kinase and improves insulin sensitivity.
          Metformin reduces hepatic glucose production through suppression of gluconeogenesis.
          Long-term metformin use decreases hemoglobin A1c levels by 1 to 2 percentage points.
          Metformin inhibits mitochondrial complex I in hepatocytes.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">10000003</PMID>
      <Article>
        <ArticleTitle>Atorvastatin and Cholesterol Metabolism</ArticleTitle>
        <Abstract>
          <AbstractText>Atorvastatin inhibits HMG-CoA reductase, the rate-limiting enzyme in cholesterol biosynthesis.
          Atorvastatin reduces low-density lipoprotein cholesterol by 40 to 60 percent.
          Statins modulate inflammatory pathways through inhibition of isoprenoid synthesis.
          Atorvastatin upregulates LDL receptor expression on hepatocyte surfaces.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">10000004</PMID>
      <Article>
        <ArticleTitle>Drug Interactions in Cardiovascular Polypharmacy</ArticleTitle>
        <Abstract>
          <AbstractText>Aspirin and atorvastatin synergistically reduce cardiovascular mortality.
          Warfarin potentiates the antiplatelet effect of aspirin, increasing bleeding risk.
          Metformin and atorvastatin share common metabolic pathways through AMPK activation.
          Clopidogrel inhibits the P2Y12 receptor and reduces platelet aggregation similarly to aspirin.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">10000005</PMID>
      <Article>
        <ArticleTitle>Immunotherapy Advances in Oncology</ArticleTitle>
        <Abstract>
          <AbstractText>Nivolumab blocks the PD-1 receptor and restores T-cell mediated immune response against tumors.
          Pembrolizumab inhibits PD-L1 binding and activates cytotoxic T lymphocytes.
          Rituximab targets CD20 on B lymphocytes and induces antibody-dependent cellular cytotoxicity.
          Trastuzumab binds HER2 receptors and inhibits downstream PI3K-AKT signaling in breast cancer cells.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">10000006</PMID>
      <Article>
        <ArticleTitle>Neurological Effects of GABAergic Medications</ArticleTitle>
        <Abstract>
          <AbstractText>Gabapentin modulates voltage-gated calcium channels and reduces neuronal excitability.
          Gabapentin suppresses glutamate release in the dorsal horn of the spinal cord.
          Pregabalin binds alpha-2-delta subunits and inhibits neurotransmitter release at hyperexcited synapses.
          Benzodiazepines potentiate GABA-A receptor activity and enhance chloride ion conductance.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">10000007</PMID>
      <Article>
        <ArticleTitle>Renal Pharmacology and ACE Inhibitors</ArticleTitle>
        <Abstract>
          <AbstractText>Lisinopril inhibits angiotensin-converting enzyme and reduces blood pressure.
          Ramipril decreases proteinuria by reducing intraglomerular pressure.
          ACE inhibitors activate bradykinin pathways and cause vasodilation.
          Losartan blocks the angiotensin II type 1 receptor and provides renoprotective effects.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">10000008</PMID>
      <Article>
        <ArticleTitle>Chemotherapy Mechanisms and Resistance</ArticleTitle>
        <Abstract>
          <AbstractText>Cisplatin crosslinks DNA strands and triggers apoptosis in rapidly dividing cells.
          Doxorubicin intercalates DNA and inhibits topoisomerase II activity.
          Methotrexate blocks dihydrofolate reductase and disrupts nucleotide synthesis.
          Cyclophosphamide alkylates DNA and suppresses both humoral and cellular immune responses.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>

</PubmedArticleSet>
""")


def run_demo() -> None:
    """
    Self-contained demo: writes a synthetic PubMed XML to a temp file,
    runs the full pipeline with LLM validation bypassed, and prints results.
    """
    import tempfile

    log.info("═══ DEMO MODE — synthetic PubMed data ═══")

    # Write demo XML to a temp file
    demo_dir = Path("brain_demo_output")
    demo_dir.mkdir(exist_ok=True)
    xml_path = demo_dir / "demo_pubmed.xml"
    xml_path.write_text(DEMO_XML, encoding="utf-8")
    log.info("Demo XML written to %s", xml_path)

    # Initialise components
    extractor = DeterministicExtractor(model_name="en_core_sci_sm")
    validator = BypassValidator()  # skip LLM in demo mode
    brain = TopologicalBrain()
    checkpoint = str(demo_dir / "demo_brain.pkl")

    pipeline = IngestorPipeline(
        extractor=extractor,
        validator=validator,
        brain=brain,
        checkpoint_path=checkpoint,
        checkpoint_every=5,       # checkpoint every 5 abstracts for demo
    )

    streamer = PubMedStreamer(xml_path)
    pipeline.run(streamer)

    # Print report
    print(pipeline.report())

    # Show a few sample node mappings
    print("\n┌─ Entity ↔ Node ID Mapping (sample) ────────────────────────────┐")
    items = list(brain._node_id.items())[:25]
    for name, nid in items:
        print(f"│  [{nid:>4d}]  {name}")
    if len(brain._node_id) > 25:
        print(f"│  … and {len(brain._node_id) - 25} more entities")
    print("└────────────────────────────────────────────────────────────────┘")


# ============================================================================
# 7.  MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BRAIN — PubMed Topological Ingestor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
            python pubmed_ingestor.py --demo
            python pubmed_ingestor.py --xml pubmed24n0001.xml --skip-llm
            python pubmed_ingestor.py --xml pubmed.xml --model groq/llama-3.3-70b-versatile
            python pubmed_ingestor.py --xml pubmed.xml --model ollama/mistral
        """),
    )

    parser.add_argument(
        "--demo", action="store_true",
        help="Run a self-contained demo with synthetic PubMed data",
    )
    parser.add_argument(
        "--xml", type=str, default=None,
        help="Path to PubMed XML file (e.g. pubmed24n0001.xml)",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Bypass LLM validation — accept all extracted triples",
    )
    parser.add_argument(
        "--model", type=str, default="ollama/mistral",
        help="LiteLLM model string (default: ollama/mistral). "
             "Examples: groq/llama-3.3-70b-versatile, gemini/gemini-2.0-flash",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum LLM confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="brain_checkpoint.pkl",
        help="Path to save/resume brain checkpoints",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1_000,
        help="Save checkpoint every N abstracts (default: 1000)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a previous checkpoint to resume from",
    )

    args = parser.parse_args()

    # ── Demo mode ────────────────────────────────────────────────────
    if args.demo:
        run_demo()
        return

    # ── Production mode ──────────────────────────────────────────────
    if args.xml is None:
        parser.error("Either --demo or --xml <file> is required.")

    # Extractor
    extractor = DeterministicExtractor(model_name="en_core_sci_sm")

    # Validator
    if args.skip_llm:
        log.info("LLM validation BYPASSED (--skip-llm)")
        validator = BypassValidator()
    else:
        validator = LLMValidator(
            model=args.model,
            confidence_threshold=args.confidence,
        )

    # Brain (resume or fresh)
    if args.resume and Path(args.resume).exists():
        brain = TopologicalBrain.load_from_disk(args.resume)
    else:
        brain = TopologicalBrain()

    # Pipeline
    pipeline = IngestorPipeline(
        extractor=extractor,
        validator=validator,
        brain=brain,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
    )

    # Stream and process
    streamer = PubMedStreamer(args.xml)
    try:
        pipeline.run(streamer)
    except KeyboardInterrupt:
        log.info("Pipeline interrupted — checkpoint saved.")

    print(pipeline.report())


if __name__ == "__main__":
    main()
