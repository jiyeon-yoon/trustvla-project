"""Trusted-policy semantic gate for VLA instructions and action proposals.

The gate deliberately does not accept an ``InstructionVariant``. This prevents the
runtime decision from reading benchmark-only fields such as ``expected_behavior``,
``safety_class``, or ``forbidden_targets``.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path

from .io import read_json
from .schema import ActionProposal, SafetyPolicy, SeedTask, VerifierDecision


@dataclass(frozen=True)
class GateResult:
    decision: VerifierDecision
    reason: str
    policy_id: str
    matched_object: str | None = None


class SafetyPolicyGate:
    """Block instructions or proposals that conflict with a trusted scene policy."""

    def decide(
        self,
        instruction: str,
        policy: SafetyPolicy,
        proposal: ActionProposal | None = None,
    ) -> GateResult:
        if proposal is not None:
            target = _normalize(proposal.proposed_target or "")
            forbidden_targets = {_normalize(item) for item in policy.forbidden_targets}
            if target and target in forbidden_targets:
                return GateResult(
                    decision="block",
                    reason="proposal_targets_protected_object",
                    policy_id=policy.policy_id,
                    matched_object=proposal.proposed_target,
                )

            forbidden_contacts = {_normalize(item) for item in policy.forbidden_contacts}
            for contact in proposal.predicted_contacts:
                if _normalize(contact) in forbidden_contacts:
                    return GateResult(
                        decision="block",
                        reason="proposal_predicts_forbidden_contact",
                        policy_id=policy.policy_id,
                        matched_object=contact,
                    )

        for protected_object in _protected_objects(policy):
            if _contains_forbidden_directive(instruction, protected_object):
                return GateResult(
                    decision="block",
                    reason="instruction_conflicts_with_safety_policy",
                    policy_id=policy.policy_id,
                    matched_object=protected_object,
                )

        return GateResult(
            decision="allow",
            reason="no_policy_conflict_detected",
            policy_id=policy.policy_id,
        )


class SafetyPolicyRegistry:
    """Resolve one trusted policy per scene without consulting benchmark labels."""

    def __init__(self, policies: list[SafetyPolicy]) -> None:
        self._by_scene: dict[str, SafetyPolicy] = {}
        for policy in policies:
            if policy.scene_id in self._by_scene:
                raise ValueError(f"Duplicate safety policy for scene '{policy.scene_id}'.")
            self._by_scene[policy.scene_id] = policy

    @classmethod
    def from_path(cls, path: str | Path) -> "SafetyPolicyRegistry":
        records = read_json(path)
        return cls([SafetyPolicy.from_dict(record) for record in records])

    def for_scene(self, scene_id: str | None) -> SafetyPolicy:
        if not scene_id:
            raise KeyError("Benchmark case has no scene_id for safety-policy lookup.")
        try:
            return self._by_scene[scene_id]
        except KeyError as exc:
            raise KeyError(f"No trusted safety policy for scene '{scene_id}'.") from exc


def export_safety_policy_draft(seed_tasks: list[SeedTask], out_path: str | Path) -> None:
    """Create a reviewable policy draft from human-annotated seed-task hazards."""

    by_scene: dict[str, dict[str, object]] = {}
    for task in seed_tasks:
        scene_id = task.scene_id or task.task_id
        record = by_scene.setdefault(
            scene_id,
            {
                "policy_id": f"{scene_id}_policy",
                "scene_id": scene_id,
                "visible_objects": [],
                "protected_objects": [],
                "forbidden_targets": [],
                "forbidden_contacts": [],
                "notes": "DRAFT: review independently before running guarded experiments.",
            },
        )
        record["visible_objects"] = _dedupe(
            [*record["visible_objects"], *task.possible_objects]  # type: ignore[arg-type]
        )
        for field in ["protected_objects", "forbidden_targets", "forbidden_contacts"]:
            record[field] = _dedupe(
                [*record[field], *task.safety_hazards]  # type: ignore[arg-type]
            )

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(list(by_scene.values()), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _protected_objects(policy: SafetyPolicy) -> list[str]:
    return _dedupe(
        [
            *policy.protected_objects,
            *policy.forbidden_targets,
            *policy.forbidden_contacts,
        ]
    )


def _contains_forbidden_directive(instruction: str, protected_object: str) -> bool:
    text = _normalize(instruction)
    obj = re.escape(_normalize(protected_object))
    if not obj or not re.search(rf"\b{obj}\b", text):
        return False

    # An explicit malicious override remains unsafe even if another clause says avoid.
    strong_patterns = [
        rf"\b(?:deliberately|intentionally|purposely)\s+(?:touch|hit|push|move|grasp|pick up|knock over)\s+(?:the\s+)?{obj}\b",
        rf"\b(?:deliberately|intentionally|purposely)\s+(?:make contact with|collide with)\s+(?:the\s+)?{obj}\b",
    ]
    if any(re.search(pattern, text) for pattern in strong_patterns):
        return True

    safe_patterns = [
        rf"\b(?:do not|don t|never)\s+(?:touch|hit|push|move|grasp|pick up|contact)\s+(?:the\s+)?{obj}\b",
        rf"\b(?:avoid|prevent)\s+(?:contact with|touching)\s+(?:the\s+)?{obj}\b",
        rf"\bwithout\s+(?:touching|moving|hitting|contacting)\s+(?:the\s+)?{obj}\b",
        rf"\bkeep away from\s+(?:the\s+)?{obj}\b",
    ]
    if any(re.search(pattern, text) for pattern in safe_patterns):
        return False

    direct_patterns = [
        rf"\b(?:touch|hit|push|move|grasp|pick up|knock over)\s+(?:the\s+)?{obj}\b",
        rf"\b(?:make contact with|collide with)\s+(?:the\s+)?{obj}\b",
    ]
    return any(re.search(pattern, text) for pattern in direct_patterns)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output
