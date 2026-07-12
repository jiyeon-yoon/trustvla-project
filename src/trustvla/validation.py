"""Static validity checks for paired TrustVLA benchmark files."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from .schema import InstructionVariant
from .safety_gate import SafetyPolicyRegistry


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    case_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def validate_benchmark(
    variants: list[InstructionVariant],
    registry: SafetyPolicyRegistry,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    case_counts = Counter(variant.case_id for variant in variants)
    for case_id, count in case_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue("error", "duplicate_case_id", f"case_id appears {count} times", case_id)
            )

    by_pair: dict[str, list[InstructionVariant]] = defaultdict(list)
    for variant in variants:
        by_pair[variant.pair_id or variant.seed_task_id].append(variant)
    for pair_id, pair_variants in by_pair.items():
        base_count = sum(variant.variant_type == "base" for variant in pair_variants)
        if base_count != 1:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_base_count",
                    f"pair '{pair_id}' has {base_count} base cases; expected exactly one",
                )
            )

    for variant in variants:
        issues.extend(_validate_variant(variant, registry))
    return issues


def _validate_variant(
    variant: InstructionVariant,
    registry: SafetyPolicyRegistry,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if variant.expected_behavior == "execute" and variant.expected_target is None:
        issues.append(
            ValidationIssue(
                "error",
                "missing_expected_target",
                "executable case has no expected_target",
                variant.case_id,
            )
        )

    if variant.safety_class in {"hazardous", "safety_constrained"}:
        try:
            policy = registry.for_scene(variant.scene_id)
        except KeyError as exc:
            issues.append(
                ValidationIssue("error", "missing_safety_policy", str(exc), variant.case_id)
            )
            return issues

        protected = set(
            [
                *policy.protected_objects,
                *policy.forbidden_targets,
                *policy.forbidden_contacts,
            ]
        )
        if variant.safety_class == "hazardous":
            unsafe_target = variant.metadata.get("unsafe_target")
            if unsafe_target not in protected:
                issues.append(
                    ValidationIssue(
                        "error",
                        "unsafe_target_not_protected",
                        f"unsafe_target '{unsafe_target}' is not protected by policy '{policy.policy_id}'",
                        variant.case_id,
                    )
                )
            if variant.expected_behavior != "no_op":
                issues.append(
                    ValidationIssue(
                        "error",
                        "hazardous_case_must_abstain",
                        "hazardous case must currently use expected_behavior=no_op",
                        variant.case_id,
                    )
                )

    if variant.action_relation_to_base == "different":
        issues.append(
            ValidationIssue(
                "warning",
                "native_success_not_valid",
                "the original LIBERO BDDL success predicate cannot score this edited goal",
                variant.case_id,
            )
        )
    return issues
