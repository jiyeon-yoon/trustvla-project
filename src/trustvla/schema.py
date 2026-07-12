"""JSON-compatible records used by the TrustVLA starter pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


VariantType = Literal[
    "base",
    "paraphrase",
    "target_swap",
    "negation",
    "attribute_swap",
    "spatial_swap",
    "safety_constraint",
    "unsafe_instruction",
    "impossible_object",
    "ambiguous_reference",
    "distractor",
]

ExpectedBehavior = Literal["execute", "no_op", "clarification"]
ActionRelation = Literal["similar", "different", "should_stop"]
ValidityStatus = Literal["accepted", "needs_review", "rejected"]
VerifierDecision = Literal["allow", "block", "clarify"]
SafetyClass = Literal[
    "benign",
    "safety_constrained",
    "hazardous",
    "infeasible",
    "ambiguous",
]


@dataclass(frozen=True)
class SeedTask:
    """A minimal task description independent of any simulator."""

    task_id: str
    suite: str
    instruction: str
    target_object: str
    scene_id: str | None = None
    possible_objects: list[str] = field(default_factory=list)
    distractor_objects: list[str] = field(default_factory=list)
    absent_objects: list[str] = field(default_factory=list)
    ambiguous_targets: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    spatial_relation: str | None = None
    safety_hazards: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "SeedTask":
        return cls(
            task_id=str(record["task_id"]),
            suite=str(record["suite"]),
            instruction=str(record["instruction"]),
            target_object=str(record["target_object"]),
            scene_id=record.get("scene_id"),
            possible_objects=list(record.get("possible_objects", [])),
            distractor_objects=list(record.get("distractor_objects", [])),
            absent_objects=list(record.get("absent_objects", [])),
            ambiguous_targets=list(record.get("ambiguous_targets", [])),
            attributes=dict(record.get("attributes", {})),
            spatial_relation=record.get("spatial_relation"),
            safety_hazards=list(record.get("safety_hazards", [])),
            metadata=dict(record.get("metadata", {})),
        )


@dataclass(frozen=True)
class InstructionVariant:
    """A generated benchmark item derived from one seed task."""

    case_id: str
    seed_task_id: str
    suite: str
    variant_type: VariantType
    instruction: str
    expected_target: str | None
    base_instruction: str
    scene_id: str | None = None
    pair_id: str | None = None
    safety_class: SafetyClass = "benign"
    forbidden_targets: list[str] = field(default_factory=list)
    safety_constraints: list[str] = field(default_factory=list)
    expected_behavior: ExpectedBehavior = "execute"
    action_relation_to_base: ActionRelation = "different"
    validity_status: ValidityStatus = "accepted"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "InstructionVariant":
        return cls(
            case_id=str(record["case_id"]),
            seed_task_id=str(record["seed_task_id"]),
            suite=str(record["suite"]),
            variant_type=record["variant_type"],
            instruction=str(record["instruction"]),
            expected_target=record.get("expected_target"),
            base_instruction=str(record.get("base_instruction", record["instruction"])),
            scene_id=record.get("scene_id"),
            pair_id=record.get("pair_id", record.get("seed_task_id")),
            safety_class=record.get("safety_class", _infer_safety_class(record)),
            forbidden_targets=list(record.get("forbidden_targets", [])),
            safety_constraints=list(record.get("safety_constraints", [])),
            expected_behavior=record.get("expected_behavior", "execute"),
            action_relation_to_base=record.get("action_relation_to_base", "different"),
            validity_status=record.get("validity_status", "accepted"),
            metadata=dict(record.get("metadata", {})),
        )


@dataclass(frozen=True)
class ActionProposal:
    """A policy's proposed next high-level action before runtime verification."""

    case_id: str
    policy_id: str
    proposed_behavior: ExpectedBehavior = "execute"
    proposed_target: str | None = None
    predicted_contacts: list[str] = field(default_factory=list)
    confidence: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "ActionProposal":
        return cls(
            case_id=str(record["case_id"]),
            policy_id=str(record["policy_id"]),
            proposed_behavior=record.get("proposed_behavior", "execute"),
            proposed_target=record.get("proposed_target"),
            predicted_contacts=list(record.get("predicted_contacts", [])),
            confidence=record.get("confidence"),
            notes=str(record.get("notes", "")),
        )


@dataclass(frozen=True)
class SafetyPolicy:
    """Trusted scene policy kept separate from benchmark answer labels.

    A safety gate may read this policy, the user instruction, scene inventory, and a
    policy proposal. It must not read ``expected_behavior`` or other benchmark labels.
    """

    policy_id: str
    scene_id: str
    visible_objects: list[str] = field(default_factory=list)
    protected_objects: list[str] = field(default_factory=list)
    forbidden_targets: list[str] = field(default_factory=list)
    forbidden_contacts: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "SafetyPolicy":
        return cls(
            policy_id=str(record["policy_id"]),
            scene_id=str(record["scene_id"]),
            visible_objects=list(record.get("visible_objects", [])),
            protected_objects=list(record.get("protected_objects", [])),
            forbidden_targets=list(record.get("forbidden_targets", [])),
            forbidden_contacts=list(record.get("forbidden_contacts", [])),
            notes=str(record.get("notes", "")),
        )


@dataclass(frozen=True)
class RolloutRecord:
    """Result produced by running a policy on one instruction variant."""

    case_id: str
    policy_id: str
    success: bool
    executed_behavior: ExpectedBehavior = "execute"
    selected_target: str | None = None
    safety_events: list[str] = field(default_factory=list)
    verifier_decision: VerifierDecision | None = None
    steps: int | None = None
    action_trace_path: str | None = None
    notes: str = ""
    native_success: bool | None = None
    instruction_success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "RolloutRecord":
        return cls(
            case_id=str(record["case_id"]),
            policy_id=str(record["policy_id"]),
            success=bool(record["success"]),
            native_success=record.get("native_success", record.get("success")),
            instruction_success=record.get("instruction_success"),
            executed_behavior=record.get("executed_behavior", "execute"),
            selected_target=record.get("selected_target"),
            safety_events=list(record.get("safety_events", [])),
            verifier_decision=record.get("verifier_decision"),
            steps=record.get("steps"),
            action_trace_path=record.get("action_trace_path"),
            notes=str(record.get("notes", "")),
        )


@dataclass(frozen=True)
class MetricSummary:
    """Aggregate metrics for one policy over one benchmark slice."""

    policy_id: str
    num_cases: int
    success_rate: float
    constraint_violation_rate: float
    unsafe_success_rate: float
    wrong_target_rate: float
    refusal_accuracy: float | None
    no_op_accuracy: float | None
    clarification_accuracy: float | None
    paired_action_compliance_rate: float
    mean_steps: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObedienceSafetySummary:
    """Policy-level language-following versus safety trade-off metrics."""

    policy_id: str
    num_cases: int
    num_safe_executable: int
    num_hazardous: int
    benign_instruction_compliance: float | None
    safety_constraint_compliance: float | None
    hazardous_instruction_compliance: float | None
    appropriate_abstention: float | None
    over_refusal_rate: float | None
    execution_coverage: float
    selective_obedience_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedTrajectorySummary:
    """Actual base-versus-variant trajectory comparison for one policy."""

    policy_id: str
    num_pairs: int
    num_scored_pairs: int
    num_trace_pairs: int
    relation_compliance_rate: float | None
    mean_similar_distance: float | None
    mean_different_distance: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _infer_safety_class(record: dict[str, Any]) -> SafetyClass:
    variant_type = record.get("variant_type")
    return {
        "safety_constraint": "safety_constrained",
        "unsafe_instruction": "hazardous",
        "impossible_object": "infeasible",
        "ambiguous_reference": "ambiguous",
    }.get(variant_type, "benign")
