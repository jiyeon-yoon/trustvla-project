"""Policy adapter interfaces.

Real adapters should translate an InstructionVariant into the inputs required by a
specific simulator and VLA checkpoint, then return a RolloutRecord.
"""

from __future__ import annotations

from typing import Protocol

from .schema import ActionProposal, InstructionVariant, RolloutRecord


class PolicyAdapter(Protocol):
    policy_id: str

    def run_case(self, variant: InstructionVariant) -> RolloutRecord:
        """Run one benchmark case and return a rollout record."""


class DummyPolicyAdapter:
    """A deterministic adapter for smoke tests.

    It intentionally fails negation cases by choosing the forbidden target. This gives
    the metric code something realistic to detect before a real VLA is integrated.
    """

    policy_id = "dummy_visual_prior_policy"

    def propose_case(self, variant: InstructionVariant) -> ActionProposal:
        base_target = variant.metadata.get("base_target")
        proposed_target = variant.expected_target
        predicted_contacts: list[str] = []

        if variant.variant_type in {
            "target_swap",
            "negation",
            "attribute_swap",
            "impossible_object",
            "ambiguous_reference",
        }:
            proposed_target = str(base_target) if base_target else variant.expected_target

        if variant.variant_type == "safety_constraint":
            predicted_contacts = list(variant.metadata.get("hazards", []))

        return ActionProposal(
            case_id=variant.case_id,
            policy_id=self.policy_id,
            proposed_behavior="execute",
            proposed_target=proposed_target,
            predicted_contacts=predicted_contacts,
            confidence=0.8,
            notes="Synthetic proposal that follows visual priors over edited language.",
        )

    def run_case(self, variant: InstructionVariant) -> RolloutRecord:
        proposal = self.propose_case(variant)
        safety_events = [
            f"contact_with_{contact.replace(' ', '_')}" for contact in proposal.predicted_contacts
        ]
        success = _task_success(
            variant=variant,
            executed_behavior=proposal.proposed_behavior,
            selected_target=proposal.proposed_target,
        )
        return RolloutRecord(
            case_id=variant.case_id,
            policy_id=self.policy_id,
            success=success,
            executed_behavior=proposal.proposed_behavior,
            selected_target=proposal.proposed_target,
            safety_events=safety_events,
            steps=25,
            notes="Synthetic rollout for pipeline smoke tests.",
        )


def _task_success(
    variant: InstructionVariant,
    executed_behavior: str,
    selected_target: str | None,
) -> bool:
    if variant.expected_behavior == "execute":
        return executed_behavior == "execute" and selected_target == variant.expected_target
    return executed_behavior == variant.expected_behavior
