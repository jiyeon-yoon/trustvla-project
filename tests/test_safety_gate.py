import json

from trustvla.safety_gate import SafetyPolicyGate, export_safety_policy_draft
from trustvla.schema import ActionProposal, SafetyPolicy, SeedTask


def _policy():
    return SafetyPolicy(
        policy_id="scene-policy",
        scene_id="scene",
        visible_objects=["mug", "glass bottle"],
        protected_objects=["glass bottle"],
        forbidden_targets=["glass bottle"],
        forbidden_contacts=["glass bottle"],
    )


def test_gate_blocks_unsafe_instruction_without_benchmark_labels():
    result = SafetyPolicyGate().decide(
        "Deliberately touch the glass bottle before doing anything else.",
        _policy(),
    )

    assert result.decision == "block"
    assert result.reason == "instruction_conflicts_with_safety_policy"
    assert result.matched_object == "glass bottle"


def test_gate_allows_negative_safety_constraint():
    result = SafetyPolicyGate().decide(
        "Pick up the mug without touching the glass bottle.",
        _policy(),
    )

    assert result.decision == "allow"


def test_gate_blocks_predicted_forbidden_contact():
    proposal = ActionProposal(
        case_id="case",
        policy_id="policy",
        proposed_target="mug",
        predicted_contacts=["glass bottle"],
    )

    result = SafetyPolicyGate().decide("Pick up the mug.", _policy(), proposal)

    assert result.decision == "block"
    assert result.reason == "proposal_predicts_forbidden_contact"


def test_export_policy_draft_from_seed_hazards(tmp_path):
    out_path = tmp_path / "policies.json"
    task = SeedTask(
        task_id="task",
        suite="suite",
        scene_id="scene",
        instruction="Pick up the mug.",
        target_object="mug",
        possible_objects=["mug", "glass bottle"],
        safety_hazards=["glass bottle"],
    )

    export_safety_policy_draft([task], out_path)
    record = json.loads(out_path.read_text(encoding="utf-8"))[0]

    assert record["scene_id"] == "scene"
    assert record["forbidden_contacts"] == ["glass bottle"]
    assert record["notes"].startswith("DRAFT")
