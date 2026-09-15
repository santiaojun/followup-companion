"""
agents/ - the pre-call and in-call decision layer for FollowUp Companion.

The rest of the system reacts to a finished call (eval/ extracts,
arbitration/ grades, triage/ routes, memory/ remembers). This package acts
*during* one: what to ask, and what to do when an answer is not usable.

Submodules
----------
planner         Pre-call script: opening, questions, pacing. The only
                module here that calls an LLM.
confirmation    Tiered re-confirmation for values we may have misheard.
symptom_probe   Detail questions for values heard clearly but described
                too vaguely to act on.
models          Dataclasses, thresholds and enums shared by the above.

Two kinds of ambiguity, two mechanisms
--------------------------------------
    "Was it stomach pain or chest pain?"   -> confirmation.py
    "My arm hurts" (certain, but useless)  -> symptom_probe.py

They are independent on purpose: reading a vague sentence back to the
patient clarifies nothing, and asking where it hurts is pointless if the
sentence was never heard. symptom_probe.classify_ambiguity() is the fork.

Public surface:
    from agents import CallPlanner, ConfirmationTracker, build_probe_plan
"""
from agents.confirmation import (
    ConfirmationTracker,
    decide_confirmation,
    is_critical_field,
)
from agents.models import (
    STANDARD_PROTOCOL_ITEMS,
    AmbiguityKind,
    CallPlan,
    ConfirmationAction,
    ConfirmationDecision,
    ConfirmationLevel,
    PlannedQuestion,
    ProbeDimension,
    ProtocolItem,
    SymptomCategory,
    SymptomProbePlan,
)
from agents.symptom_probe import (
    PROBE_REGISTRY,
    build_probe_plan,
    classify_ambiguity,
    classify_symptom,
    is_symptom_incomplete,
    plan_for_claim,
)

__all__ = [
    # Tier 1 - confirmation ("didn't hear it")
    "ConfirmationTracker",
    "decide_confirmation",
    "is_critical_field",
    "ConfirmationAction",
    "ConfirmationDecision",
    "ConfirmationLevel",
    # Tier 2 - symptom probing ("heard it, too vague")
    "AmbiguityKind",
    "PROBE_REGISTRY",
    "ProbeDimension",
    "SymptomCategory",
    "SymptomProbePlan",
    "build_probe_plan",
    "classify_ambiguity",
    "classify_symptom",
    "is_symptom_incomplete",
    "plan_for_claim",
    # Tier 3 - call planning
    "CallPlan",
    "PlannedQuestion",
    "ProtocolItem",
    "STANDARD_PROTOCOL_ITEMS",
]


def __getattr__(name: str):
    """
    Expose CallPlanner lazily.

    agents.planner imports openai at construction time; keeping it out of
    the eager import list means `import agents` works (and the confirmation
    and probe logic stays testable) without the openai package installed.
    """
    if name == "CallPlanner":
        from agents.planner import CallPlanner
        return CallPlanner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
