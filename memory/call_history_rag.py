"""
memory/call_history_rag.py – semantic retrieval of historical call summaries.

Stores one summary document per call in a ChromaDB collection.
Retrieval is embedding-based (all-MiniLM-L6-v2 via ChromaDB's default
embedding function), so queries match by meaning rather than exact keywords.

Intended use
------------
Before each outbound call the agent retrieves the k most contextually
relevant historical summaries.  This provides soft continuity: if a
previous call noted "son started university", the next call planner can
pick up that thread naturally without hard-coding topic tracking.

Collection schema
-----------------
Collection name : "call_summaries"
Document        : free-text call summary (what gets embedded)
Metadata        :
  patient_id  str   – used as the per-patient namespace filter
  call_id     str   – unique identifier for the call
  timestamp   str   – ISO-8601 datetime string (used for recency sort)
  topics      str   – comma-separated hint tags (e.g. "blood_glucose,diet")
                       stored as string because ChromaDB metadata must be
                       a primitive type; split back to list on retrieval

Document IDs    : "{patient_id}::{call_id}" (deterministic → upsert is idempotent)

Usage
-----
    rag = CallHistoryRAG()                       # persists to memory/.chroma/
    rag = CallHistoryRAG(persist_dir=None)        # ephemeral (tests / REPL)

    rag.add_call_summary(
        patient_id="PAT-001",
        call_id="CALL-2024-06-01",
        summary="Blood glucose averaging 6.8 mmol/L. HbA1c 7.1%. ...",
        timestamp="2024-06-01T10:30:00Z",
        topics=["blood_glucose", "medication"],
    )

    results = rag.retrieve("PAT-001", "blood sugar glucose levels", n_results=3)
    for r in results:
        print(r.call_id, r.distance, r.summary[:80])

    recent = rag.retrieve_recent("PAT-001", n=5)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLLECTION_NAME = "call_summaries"
# Default persistence directory: memory/.chroma/ (sibling of this file)
DEFAULT_PERSIST_DIR: str = str(Path(__file__).parent / ".chroma")

# ---------------------------------------------------------------------------
# Ephemeral client (shared per process)
# ---------------------------------------------------------------------------
# chromadb 1.5.9: constructing many EphemeralClient() instances in one process
# corrupts the query path of collections created by EARLIER clients. The
# collection keeps its documents and `.get()` still works, but `.query()`
# raises from the Rust bindings:
#
#   chromadb.errors.InternalError: Error executing plan:
#   Internal error: Error finding id
#
# Measured on this machine: 32/40 queries fail after 40 clients are created,
# versus 0/40 when a single client is reused for the same 40 collections. The
# failure is not transient - an immediate retry fails too (0/20 recovered) -
# so it has to be avoided rather than retried around.
#
# Reusing one client per process is also the semantically correct model here:
# ephemeral mode exists for tests and REPL work, and isolation between stores
# comes from `collection_name`, not from the client. Two instances sharing a
# collection name already shared a store before this change.
_EPHEMERAL_CLIENT = None


def _shared_ephemeral_client():
    """Return the process-wide in-memory chromadb client, creating it once."""
    global _EPHEMERAL_CLIENT
    if _EPHEMERAL_CLIENT is None:
        _EPHEMERAL_CLIENT = chromadb.EphemeralClient()
    return _EPHEMERAL_CLIENT


# The default embedding function loads an ONNX MiniLM model. Building one per
# instance both wastes time (~0.3s each) and contributes to the client/segment
# resource churn behind the InternalError above, so it is cached per process
# too. The model is stateless for our purposes - it only encodes text - so a
# single instance is safe to share.
_DEFAULT_EMBEDDING_FN = None


def _shared_default_embedding_fn():
    """Return the process-wide default embedding function, creating it once."""
    global _DEFAULT_EMBEDDING_FN
    if _DEFAULT_EMBEDDING_FN is None:
        _DEFAULT_EMBEDDING_FN = DefaultEmbeddingFunction()
    return _DEFAULT_EMBEDDING_FN


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class CallSummaryResult:
    """One retrieved call summary with its metadata and similarity score."""
    patient_id: str
    call_id: str
    summary: str
    timestamp: str
    topics: list[str]
    # Cosine distance in [0.0, 2.0]; lower = more similar to the query.
    # Set to 0.0 for recency-based results (no query distance applies).
    distance: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CallHistoryRAG:
    """
    Semantic store for per-patient call summaries.

    One ChromaDB collection ("call_summaries") is shared across all patients;
    per-patient queries are isolated via metadata filter on `patient_id`.

    Parameters
    ----------
    persist_dir : str | Path | None
        Directory for ChromaDB's on-disk storage.
        Pass ``None`` for an ephemeral in-memory store (useful in tests and
        interactive sessions where persistence is not needed).
    embedding_function : optional
        Custom embedding function (must follow chromadb EmbeddingFunction
        protocol).  Defaults to ChromaDB's built-in all-MiniLM-L6-v2.
        Inject a stub in fast unit tests if model download is undesirable.
    """

    def __init__(
        self,
        persist_dir: Optional[str | Path] = DEFAULT_PERSIST_DIR,
        embedding_function=None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        if persist_dir is None:
            # Shared per process - see _shared_ephemeral_client for why a
            # fresh EphemeralClient per instance breaks earlier collections.
            self._client = _shared_ephemeral_client()
        else:
            self._client = chromadb.PersistentClient(path=str(persist_dir))

        ef = embedding_function or _shared_default_embedding_fn()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_call_summary(
        self,
        patient_id: str,
        call_id: str,
        summary: str,
        timestamp: str,
        topics: Optional[list[str]] = None,
    ) -> None:
        """
        Store (or update) a call summary.

        The document ID is deterministic: calling this twice with the same
        patient_id + call_id updates the existing record (upsert semantics).

        Parameters
        ----------
        patient_id  : Patient identifier (used as namespace filter).
        call_id     : Unique identifier for this call (e.g. "CALL-2024-06-01").
        summary     : Free-text summary of the call.  This is what the
                      embedding model encodes — richer text → better retrieval.
        timestamp   : ISO-8601 string (e.g. "2024-06-01T10:30:00Z").
                      Used for recency-based retrieval.
        topics      : Optional hint tags.  Not used for embedding; useful for
                      downstream filtering or display.
        """
        doc_id = f"{patient_id}::{call_id}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[summary],
            metadatas=[{
                "patient_id": patient_id,
                "call_id": call_id,
                "timestamp": timestamp,
                "topics": ",".join(topics or []),
            }],
        )

    def delete_patient(self, patient_id: str) -> int:
        """
        Delete all call summaries for a patient.

        Returns the number of records deleted.
        Intended for GDPR / data-retention workflows.
        """
        response = self._collection.get(
            where={"patient_id": {"$eq": patient_id}},
            include=[],  # IDs only; skip documents and metadata
        )
        ids = response["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    # ------------------------------------------------------------------
    # Read – semantic retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        patient_id: str,
        query: str,
        n_results: int = 3,
    ) -> list[CallSummaryResult]:
        """
        Return the ``n_results`` most semantically similar call summaries
        for this patient.

        Results are ordered ascending by cosine distance (most relevant first).
        Returns an empty list if the patient has no stored summaries.
        """
        patient_count = self._count_patient_docs(patient_id)
        if patient_count == 0:
            return []
        effective_n = min(n_results, patient_count)

        response = self._collection.query(
            query_texts=[query],
            n_results=effective_n,
            where={"patient_id": {"$eq": patient_id}},
            include=["documents", "metadatas", "distances"],
        )

        results: list[CallSummaryResult] = []
        for doc, meta, dist in zip(
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            results.append(
                CallSummaryResult(
                    patient_id=meta["patient_id"],
                    call_id=meta["call_id"],
                    summary=doc,
                    timestamp=meta["timestamp"],
                    topics=_split_topics(meta.get("topics", "")),
                    distance=float(dist),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Read – recency retrieval
    # ------------------------------------------------------------------

    def retrieve_recent(
        self,
        patient_id: str,
        n: int = 5,
    ) -> list[CallSummaryResult]:
        """
        Return the ``n`` most recent call summaries for this patient.

        Sorted descending by ``timestamp`` (ISO-8601 lexicographic order).
        ``distance`` is set to 0.0 because recency queries are not
        similarity-ranked.
        """
        response = self._collection.get(
            where={"patient_id": {"$eq": patient_id}},
            include=["documents", "metadatas"],
        )

        pairs = list(zip(response["documents"], response["metadatas"]))
        pairs.sort(key=lambda x: x[1].get("timestamp", ""), reverse=True)

        results: list[CallSummaryResult] = []
        for doc, meta in pairs[:n]:
            results.append(
                CallSummaryResult(
                    patient_id=meta["patient_id"],
                    call_id=meta["call_id"],
                    summary=doc,
                    timestamp=meta["timestamp"],
                    topics=_split_topics(meta.get("topics", "")),
                    distance=0.0,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self, patient_id: Optional[str] = None) -> int:
        """Return total number of stored summaries (optionally filtered by patient)."""
        if patient_id is None:
            return self._collection.count()
        return self._count_patient_docs(patient_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_patient_docs(self, patient_id: str) -> int:
        """Return the number of stored summaries for a specific patient."""
        response = self._collection.get(
            where={"patient_id": {"$eq": patient_id}},
            include=[],  # IDs only
        )
        return len(response["ids"])


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _split_topics(raw: str) -> list[str]:
    """Split the comma-separated topics string back into a list."""
    return [t.strip() for t in raw.split(",") if t.strip()]
