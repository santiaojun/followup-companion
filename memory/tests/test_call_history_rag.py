"""
Tests for CallHistoryRAG.

Core assertions
---------------
1. Given 5 summaries on distinct clinical/social topics, a semantic query on
   "blood glucose" returns the blood glucose summary at the top — and does NOT
   return the purely social/family summary in the top 2 results.
2. Symmetric check: a "family and social" query returns the family summary, not
   the blood glucose one.
3. A sleep-disorder query correctly surfaces the sleep summary.
4. retrieve_recent returns summaries sorted by timestamp (newest first).
5. Results for PAT-001 are isolated from PAT-002 (no cross-patient leakage).
6. Querying a patient with no history returns an empty list.
7. Upsert is idempotent (adding the same call twice keeps count at 1).
8. delete_patient removes all records and subsequent retrieve returns [].

Isolation note
--------------
Every fixture gets its own temporary directory via `tmp_path_factory`, so
stores are isolated by filesystem path and pytest cleans them up.

This deliberately uses PersistentClient rather than in-memory mode, for two
reasons. It is the client production runs, so these tests exercise the real
path. And chromadb 1.5.9 has a bug in ephemeral mode: once enough
EphemeralClient instances exist in one process, `.query()` against a
collection created earlier fails from the Rust bindings with

    InternalError: Error executing plan: Internal error: Error finding id

Measured 23/60 queries failing after 60 ephemeral instances, versus 0/40
for the identical sequence against persistent stores. It is not transient -
an immediate retry fails too - so it has to be avoided rather than retried
around. It is also why only the two `.query()`-based TestMetadataIntegrity
tests used to flake: `retrieve_recent` goes through `.get()`, which is
unaffected, and the unknown-patient tests short-circuit on a zero count
before reaching the query.
"""
from __future__ import annotations

import uuid

import pytest

from memory.call_history_rag import CallHistoryRAG

# ---------------------------------------------------------------------------
# Synthetic call summaries – five clearly distinct topics
# ---------------------------------------------------------------------------

PATIENT = "PAT-001"

BLOOD_GLUCOSE_SUMMARY = (
    "Patient reported stable blood glucose readings this week, averaging 6.8 mmol/L. "
    "HbA1c was reviewed at the last clinic visit and came in at 7.1 percent. "
    "Metformin dose remains unchanged. Patient managing dietary carbohydrate intake "
    "well and monitoring blood sugar levels before meals. No hypoglycaemic episodes "
    "reported. Endocrinologist satisfied with glycaemic control."
)

FAMILY_SOCIAL_SUMMARY = (
    "Patient mentioned that his son recently started university in another city. "
    "Feeling a mix of pride and loneliness since the son moved away. "
    "Daughter visits every weekend which provides good social support. "
    "Patient's wife is also in good health. Overall social support network appears "
    "adequate. No clinical concerns raised during this part of the conversation."
)

SLEEP_SUMMARY = (
    "Patient reporting persistent insomnia for the past two weeks. "
    "Wakes frequently around 3 AM and cannot return to sleep. "
    "Daytime fatigue is affecting ability to concentrate and complete daily tasks. "
    "Patient has tried relaxation techniques without success. "
    "Agreed to raise sleep difficulties with GP at next appointment to discuss "
    "possible sleep hygiene strategies or referral to a sleep clinic."
)

MEDICATION_SIDE_EFFECTS_SUMMARY = (
    "Patient experiencing nausea since starting the new antihypertensive medication "
    "three weeks ago. Nausea typically occurs approximately 30 minutes after taking "
    "the morning tablet. Suggested patient try taking the tablet with food rather than "
    "on an empty stomach. Doctor may consider a dose adjustment or alternative "
    "formulation at the next review appointment. Medication adherence maintained "
    "despite the side effects."
)

EXERCISE_PHYSIO_SUMMARY = (
    "Patient completed the first full week of post-operative physiotherapy. "
    "Walking 20 to 30 minutes daily as advised by the physiotherapist. "
    "Mild muscle discomfort during exercises but this is expected and manageable. "
    "Physiotherapist pleased with the range of motion progress in the operated joint. "
    "Patient encouraged to continue the home exercise programme. "
    "Next physiotherapy session scheduled for the following week."
)

# (call_id, summary, ISO timestamp, topic tags)
# CALL-A is oldest, CALL-E is newest.
_CALL_RECORDS = [
    ("CALL-A", BLOOD_GLUCOSE_SUMMARY,         "2024-06-01T10:00:00Z", ["blood_glucose", "medication"]),
    ("CALL-B", FAMILY_SOCIAL_SUMMARY,          "2024-06-08T10:00:00Z", ["family", "social"]),
    ("CALL-C", SLEEP_SUMMARY,                  "2024-06-15T10:00:00Z", ["sleep", "fatigue"]),
    ("CALL-D", MEDICATION_SIDE_EFFECTS_SUMMARY,"2024-06-22T10:00:00Z", ["medication", "side_effects"]),
    ("CALL-E", EXERCISE_PHYSIO_SUMMARY,         "2024-06-29T10:00:00Z", ["exercise", "physiotherapy"]),
]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _unique_name() -> str:
    """Return a unique collection name to prevent cross-test contamination."""
    return f"test_{uuid.uuid4().hex[:12]}"


@pytest.fixture(scope="module")
def populated_rag(tmp_path_factory):
    """
    Module-scoped RAG with all 5 diverse summaries loaded for PAT-001.
    Embedding inference runs once per module; subsequent tests reuse the store.
    """
    rag = CallHistoryRAG(persist_dir=tmp_path_factory.mktemp("populated_rag"))
    for call_id, summary, ts, topics in _CALL_RECORDS:
        rag.add_call_summary(PATIENT, call_id, summary, ts, topics)

    # Verify the store populated before any test runs. Without this, a setup
    # problem surfaces downstream as an IndexError inside an unrelated
    # assertion, which is a much harder thing to diagnose than a fixture
    # that fails loudly with a count.
    stored = rag.count(PATIENT)
    assert stored == len(_CALL_RECORDS), (
        f"fixture setup failed: expected {len(_CALL_RECORDS)} summaries for "
        f"{PATIENT}, collection holds {stored}"
    )
    return rag


@pytest.fixture
def fresh_rag(tmp_path_factory):
    """
    Function-scoped empty RAG in its own temporary directory.

    Use this in tests that write data and need a clean starting state.
    """
    return CallHistoryRAG(persist_dir=tmp_path_factory.mktemp("fresh_rag"))


# ---------------------------------------------------------------------------
# 1. Semantic precision – blood glucose query
# ---------------------------------------------------------------------------

class TestBloodGlucoseRetrieval:

    def test_top_result_is_blood_glucose_summary(self, populated_rag):
        """Top result for 'blood sugar glucose mmol HbA1c' must be CALL-A."""
        results = populated_rag.retrieve(
            PATIENT, "blood sugar glucose levels HbA1c mmol", n_results=1
        )
        assert len(results) == 1, "Expected exactly 1 result"
        assert results[0].call_id == "CALL-A", (
            f"Expected CALL-A (blood glucose) at position 0, got {results[0].call_id}. "
            f"Summary snippet: {results[0].summary[:60]!r}"
        )

    def test_family_summary_not_in_top_two_for_glucose_query(self, populated_rag):
        """Family/social summary must NOT be in top 2 for a blood glucose query."""
        results = populated_rag.retrieve(
            PATIENT, "blood sugar glucose levels HbA1c mmol", n_results=2
        )
        returned_ids = {r.call_id for r in results}
        assert "CALL-B" not in returned_ids, (
            f"Family summary (CALL-B) must not appear in top 2 for glucose query. "
            f"Got: {returned_ids}"
        )

    def test_blood_glucose_has_lower_distance_than_family(self, populated_rag):
        """Blood glucose summary must be semantically closer to a glucose query than family."""
        results = populated_rag.retrieve(
            PATIENT, "blood sugar glucose HbA1c", n_results=5
        )
        dist_map = {r.call_id: r.distance for r in results}
        assert "CALL-A" in dist_map and "CALL-B" in dist_map, (
            "Both CALL-A and CALL-B must be returned when n_results=5"
        )
        assert dist_map["CALL-A"] < dist_map["CALL-B"], (
            f"Blood glucose distance {dist_map['CALL-A']:.4f} must be < "
            f"family distance {dist_map['CALL-B']:.4f} for glucose query"
        )


# ---------------------------------------------------------------------------
# 2. Semantic precision – family/social query
# ---------------------------------------------------------------------------

class TestFamilySocialRetrieval:

    def test_top_result_is_family_summary(self, populated_rag):
        """Top result for 'son university family support' must be CALL-B."""
        results = populated_rag.retrieve(
            PATIENT, "son university family emotional support", n_results=1
        )
        assert results[0].call_id == "CALL-B", (
            f"Expected CALL-B (family) at position 0, got {results[0].call_id}"
        )

    def test_glucose_summary_not_in_top_two_for_family_query(self, populated_rag):
        results = populated_rag.retrieve(
            PATIENT, "son university family emotional support", n_results=2
        )
        returned_ids = {r.call_id for r in results}
        assert "CALL-A" not in returned_ids, (
            f"Blood glucose summary (CALL-A) must not appear in top 2 for family query. "
            f"Got: {returned_ids}"
        )


# ---------------------------------------------------------------------------
# 3. Semantic precision – sleep query
# ---------------------------------------------------------------------------

class TestSleepRetrieval:

    def test_top_result_is_sleep_summary(self, populated_rag):
        """Top result for 'insomnia poor sleep waking at night' must be CALL-C."""
        results = populated_rag.retrieve(
            PATIENT, "insomnia poor sleep waking at night fatigue", n_results=1
        )
        assert results[0].call_id == "CALL-C", (
            f"Expected CALL-C (sleep) at position 0, got {results[0].call_id}"
        )

    def test_sleep_has_lower_distance_than_glucose_for_sleep_query(self, populated_rag):
        results = populated_rag.retrieve(
            PATIENT, "insomnia poor sleep waking at night", n_results=5
        )
        dist_map = {r.call_id: r.distance for r in results}
        assert dist_map["CALL-C"] < dist_map["CALL-A"], (
            f"Sleep distance {dist_map['CALL-C']:.4f} must be < "
            f"glucose distance {dist_map['CALL-A']:.4f} for sleep query"
        )


# ---------------------------------------------------------------------------
# 4. Recency retrieval
# ---------------------------------------------------------------------------

class TestRetrieveRecent:

    def test_most_recent_first(self, populated_rag):
        """retrieve_recent must return CALL-E (newest) as the first result."""
        results = populated_rag.retrieve_recent(PATIENT, n=5)
        assert len(results) == 5
        assert results[0].call_id == "CALL-E", (
            f"Expected CALL-E (newest) first; got {results[0].call_id}"
        )

    def test_oldest_is_last(self, populated_rag):
        results = populated_rag.retrieve_recent(PATIENT, n=5)
        assert results[-1].call_id == "CALL-A", (
            f"Expected CALL-A (oldest) last; got {results[-1].call_id}"
        )

    def test_recency_n_truncates_correctly(self, populated_rag):
        results = populated_rag.retrieve_recent(PATIENT, n=3)
        assert len(results) == 3
        call_ids = [r.call_id for r in results]
        assert call_ids == ["CALL-E", "CALL-D", "CALL-C"]

    def test_timestamps_are_preserved(self, populated_rag):
        results = populated_rag.retrieve_recent(PATIENT, n=1)
        assert results[0].timestamp == "2024-06-29T10:00:00Z"


# ---------------------------------------------------------------------------
# 5. Patient isolation – no cross-patient leakage
# ---------------------------------------------------------------------------

class TestPatientIsolation:

    def test_different_patient_does_not_see_pat001_summaries(self, fresh_rag):
        """PAT-002 retrieval must return empty – its data lives in its own namespace."""
        for call_id, summary, ts, topics in _CALL_RECORDS:
            fresh_rag.add_call_summary("PAT-001", call_id, summary, ts, topics)
        results = fresh_rag.retrieve("PAT-002", "blood glucose levels", n_results=3)
        assert results == [], (
            f"PAT-002 should see no results, but got {[r.call_id for r in results]}"
        )

    def test_pat002_summary_not_returned_for_pat001_query(self, fresh_rag):
        """A PAT-002 summary must not appear in PAT-001 query results."""
        fresh_rag.add_call_summary("PAT-001", "CALL-A", BLOOD_GLUCOSE_SUMMARY, "2024-06-01T10:00:00Z")
        fresh_rag.add_call_summary("PAT-002", "CALL-X", BLOOD_GLUCOSE_SUMMARY, "2024-06-01T10:00:00Z")
        results = fresh_rag.retrieve("PAT-001", "blood glucose", n_results=5)
        returned_ids = {r.call_id for r in results}
        assert "CALL-X" not in returned_ids, "PAT-002's CALL-X must not appear in PAT-001 results"
        assert all(r.patient_id == "PAT-001" for r in results), "All results must belong to PAT-001"


# ---------------------------------------------------------------------------
# 6. Empty patient
# ---------------------------------------------------------------------------

class TestEmptyPatient:

    def test_retrieve_unknown_patient_returns_empty(self, populated_rag):
        results = populated_rag.retrieve("PAT-UNKNOWN", "blood glucose", n_results=3)
        assert results == []

    def test_retrieve_recent_unknown_patient_returns_empty(self, populated_rag):
        results = populated_rag.retrieve_recent("PAT-UNKNOWN", n=5)
        assert results == []

    def test_count_unknown_patient_is_zero(self, populated_rag):
        assert populated_rag.count("PAT-UNKNOWN") == 0


# ---------------------------------------------------------------------------
# 7. Upsert idempotency
# ---------------------------------------------------------------------------

class TestUpsertIdempotency:

    def test_adding_same_call_twice_keeps_count_at_one(self, fresh_rag):
        fresh_rag.add_call_summary("PAT-001", "CALL-A", BLOOD_GLUCOSE_SUMMARY, "2024-06-01T10:00:00Z")
        fresh_rag.add_call_summary("PAT-001", "CALL-A", BLOOD_GLUCOSE_SUMMARY, "2024-06-01T10:00:00Z")
        assert fresh_rag.count("PAT-001") == 1

    def test_upsert_updates_summary_text(self, fresh_rag):
        fresh_rag.add_call_summary("PAT-001", "CALL-A", "original summary text", "2024-06-01T10:00:00Z")
        fresh_rag.add_call_summary("PAT-001", "CALL-A", "updated summary text after correction", "2024-06-01T10:00:00Z")
        results = fresh_rag.retrieve_recent("PAT-001", n=1)
        assert "updated" in results[0].summary


# ---------------------------------------------------------------------------
# 8. delete_patient
# ---------------------------------------------------------------------------

class TestDeletePatient:

    def test_delete_removes_all_records(self, fresh_rag):
        for call_id, summary, ts, topics in _CALL_RECORDS:
            fresh_rag.add_call_summary(PATIENT, call_id, summary, ts, topics)
        assert fresh_rag.count(PATIENT) == 5

        deleted = fresh_rag.delete_patient(PATIENT)

        assert deleted == 5
        assert fresh_rag.count(PATIENT) == 0

    def test_retrieve_after_delete_returns_empty(self, fresh_rag):
        fresh_rag.add_call_summary(PATIENT, "CALL-A", BLOOD_GLUCOSE_SUMMARY, "2024-06-01T10:00:00Z")
        fresh_rag.delete_patient(PATIENT)
        results = fresh_rag.retrieve(PATIENT, "blood glucose", n_results=3)
        assert results == []

    def test_delete_does_not_affect_other_patients(self, fresh_rag):
        fresh_rag.add_call_summary("PAT-001", "CALL-A", BLOOD_GLUCOSE_SUMMARY, "2024-06-01T10:00:00Z")
        fresh_rag.add_call_summary("PAT-002", "CALL-B", FAMILY_SOCIAL_SUMMARY,  "2024-06-01T10:00:00Z")
        fresh_rag.delete_patient("PAT-001")
        assert fresh_rag.count("PAT-001") == 0
        assert fresh_rag.count("PAT-002") == 1


# ---------------------------------------------------------------------------
# 9. Metadata integrity
# ---------------------------------------------------------------------------

class TestMetadataIntegrity:

    def test_topics_are_round_tripped_correctly(self, populated_rag):
        results = populated_rag.retrieve_recent(PATIENT, n=5)
        call_a = next(r for r in results if r.call_id == "CALL-A")
        assert "blood_glucose" in call_a.topics
        assert "medication" in call_a.topics

    # Both tests below previously used n_results=1 and indexed results[0],
    # which made a metadata assertion depend on retrieval RANK: the bare
    # query "blood glucose" had to put CALL-A first, ahead of CALL-D, whose
    # summary also mentions medication and tablets. That is a near-tie
    # between two float distances, and it made these two tests flake
    # intermittently under full-suite runs while their sibling above -- which
    # locates CALL-A by call_id -- never did.
    #
    # Ranking is already covered deliberately in TestBloodGlucoseRetrieval
    # with a query chosen to be discriminating ("blood sugar glucose mmol
    # HbA1c"). These tests are about metadata surviving the round trip, so
    # they now locate the record instead of assuming its position.

    def test_patient_id_is_preserved(self, populated_rag):
        """Every result must carry the queried patient's id."""
        results = populated_rag.retrieve(PATIENT, "blood glucose", n_results=5)
        assert results, "retrieve returned no results for a populated patient"
        assert all(r.patient_id == PATIENT for r in results), (
            f"foreign patient_id in results: "
            f"{[(r.patient_id, r.call_id) for r in results]}"
        )

    def test_timestamp_is_preserved_on_retrieve(self, populated_rag):
        """CALL-A's timestamp must survive the write/embed/read round trip."""
        results = populated_rag.retrieve(PATIENT, "blood glucose", n_results=5)
        assert results, "retrieve returned no results for a populated patient"
        call_a = next((r for r in results if r.call_id == "CALL-A"), None)
        assert call_a is not None, (
            f"CALL-A not among retrieved results: {[r.call_id for r in results]}"
        )
        assert call_a.timestamp == "2024-06-01T10:00:00Z"


# ---------------------------------------------------------------------------
# 10. In-memory mode – shared client per process
# ---------------------------------------------------------------------------

class TestEphemeralMode:
    """
    Ephemeral mode reuses one chromadb client per process.

    The fixtures above use PersistentClient, so this class is what covers
    the ephemeral path. Sharing the client is required, not an
    optimisation: a fresh EphemeralClient per instance corrupts the query
    path of collections created earlier in the same process
    (chromadb 1.5.9). Isolation still comes from `collection_name`.
    """

    def test_instances_share_one_client(self):
        a = CallHistoryRAG(persist_dir=None, collection_name=_unique_name())
        b = CallHistoryRAG(persist_dir=None, collection_name=_unique_name())
        assert a._client is b._client

    def test_persistent_instances_do_not_share_the_ephemeral_client(self, tmp_path):
        ephemeral = CallHistoryRAG(persist_dir=None, collection_name=_unique_name())
        persistent = CallHistoryRAG(persist_dir=tmp_path / "store")
        assert persistent._client is not ephemeral._client

    def test_distinct_collection_names_stay_isolated(self):
        a = CallHistoryRAG(persist_dir=None, collection_name=_unique_name())
        b = CallHistoryRAG(persist_dir=None, collection_name=_unique_name())
        a.add_call_summary("PAT-A", "CALL-A", BLOOD_GLUCOSE_SUMMARY,
                           "2024-06-01T10:00:00Z", ["blood_glucose"])
        b.add_call_summary("PAT-B", "CALL-B", BLOOD_GLUCOSE_SUMMARY,
                           "2024-07-01T10:00:00Z", ["blood_glucose"])

        assert a.count() == 1
        assert b.count() == 1
        results = a.retrieve("PAT-A", "blood glucose", n_results=5)
        assert [r.call_id for r in results] == ["CALL-A"]
        assert a.retrieve("PAT-B", "blood glucose", n_results=5) == []

    def test_same_collection_name_shares_the_store(self):
        name = _unique_name()
        writer = CallHistoryRAG(persist_dir=None, collection_name=name)
        reader = CallHistoryRAG(persist_dir=None, collection_name=name)
        writer.add_call_summary("PAT-S", "CALL-S", SLEEP_SUMMARY,
                                "2024-06-15T10:00:00Z", ["sleep"])
        assert reader.count("PAT-S") == 1

    def test_default_embedding_function_is_reused(self):
        # One ONNX model load per process rather than one per instance.
        from memory.call_history_rag import _shared_default_embedding_fn
        assert _shared_default_embedding_fn() is _shared_default_embedding_fn()
