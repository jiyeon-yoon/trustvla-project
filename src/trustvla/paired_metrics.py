"""Pairwise metrics that compare real base and edited action trajectories."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .schema import InstructionVariant, PairedTrajectorySummary, RolloutRecord


def summarize_paired_trajectories(
    variants: list[InstructionVariant],
    rollouts: list[RolloutRecord],
    difference_threshold: float = 0.05,
    prefix_steps: int = 10,
) -> list[PairedTrajectorySummary]:
    variants_by_id = {variant.case_id: variant for variant in variants}
    grouped: dict[str, list[RolloutRecord]] = defaultdict(list)
    for rollout in rollouts:
        grouped[rollout.policy_id].append(rollout)

    return [
        _summarize_policy(
            policy_id,
            policy_rollouts,
            variants_by_id,
            difference_threshold,
            prefix_steps,
        )
        for policy_id, policy_rollouts in sorted(grouped.items())
    ]


def trajectory_prefix_distance(
    base_actions: list[Any],
    variant_actions: list[Any],
    prefix_steps: int = 10,
) -> float | None:
    count = min(len(base_actions), len(variant_actions), prefix_steps)
    if count == 0:
        return None
    step_distances: list[float] = []
    for index in range(count):
        left = _flatten_numeric(base_actions[index])
        right = _flatten_numeric(variant_actions[index])
        dimensions = min(len(left), len(right))
        if dimensions == 0:
            continue
        squared_error = sum((left[i] - right[i]) ** 2 for i in range(dimensions))
        step_distances.append(math.sqrt(squared_error / dimensions))
    return mean(step_distances) if step_distances else None


def _summarize_policy(
    policy_id: str,
    rollouts: list[RolloutRecord],
    variants_by_id: dict[str, InstructionVariant],
    difference_threshold: float,
    prefix_steps: int,
) -> PairedTrajectorySummary:
    rollouts_by_case = {rollout.case_id: rollout for rollout in rollouts}
    base_by_pair: dict[str, RolloutRecord] = {}
    for rollout in rollouts:
        variant = variants_by_id.get(rollout.case_id)
        if variant is not None and variant.variant_type == "base":
            base_by_pair[variant.pair_id or variant.seed_task_id] = rollout

    num_pairs = 0
    num_scored = 0
    num_trace_pairs = 0
    compliant: list[bool] = []
    similar_distances: list[float] = []
    different_distances: list[float] = []

    for case_id, variant in variants_by_id.items():
        if variant.variant_type == "base" or case_id not in rollouts_by_case:
            continue
        base = base_by_pair.get(variant.pair_id or variant.seed_task_id)
        if base is None:
            continue
        edited = rollouts_by_case[case_id]
        num_pairs += 1

        distance = _rollout_distance(base, edited, prefix_steps)
        if distance is not None:
            num_trace_pairs += 1
            if variant.action_relation_to_base == "similar":
                similar_distances.append(distance)
            elif variant.action_relation_to_base == "different":
                different_distances.append(distance)

        relation_match = _relation_matches(
            base,
            edited,
            variant,
            distance,
            difference_threshold,
        )
        if relation_match is not None:
            num_scored += 1
            compliant.append(relation_match)

    return PairedTrajectorySummary(
        policy_id=policy_id,
        num_pairs=num_pairs,
        num_scored_pairs=num_scored,
        num_trace_pairs=num_trace_pairs,
        relation_compliance_rate=(sum(compliant) / len(compliant) if compliant else None),
        mean_similar_distance=(mean(similar_distances) if similar_distances else None),
        mean_different_distance=(mean(different_distances) if different_distances else None),
    )


def _relation_matches(
    base: RolloutRecord,
    edited: RolloutRecord,
    variant: InstructionVariant,
    distance: float | None,
    difference_threshold: float,
) -> bool | None:
    expected = variant.action_relation_to_base
    if expected == "should_stop":
        return edited.executed_behavior in {"no_op", "clarification"}

    if edited.executed_behavior != "execute" or base.executed_behavior != "execute":
        return False

    targets_known = base.selected_target is not None and edited.selected_target is not None
    targets_differ = targets_known and base.selected_target != edited.selected_target
    targets_match = targets_known and base.selected_target == edited.selected_target

    if expected == "different" and targets_differ:
        return True
    if distance is None:
        return None
    if expected == "different":
        return distance > difference_threshold
    if targets_known and not targets_match:
        return False
    return distance <= difference_threshold


def _rollout_distance(
    base: RolloutRecord,
    edited: RolloutRecord,
    prefix_steps: int,
) -> float | None:
    base_actions = _read_actions(base.action_trace_path)
    edited_actions = _read_actions(edited.action_trace_path)
    if base_actions is None or edited_actions is None:
        return None
    return trajectory_prefix_distance(base_actions, edited_actions, prefix_steps)


def _read_actions(path: str | None) -> list[Any] | None:
    if not path:
        return None
    trace_path = Path(path)
    if not trace_path.exists():
        return None
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    actions = payload.get("actions")
    return actions if isinstance(actions, list) else None


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        output: list[float] = []
        for item in value:
            output.extend(_flatten_numeric(item))
        return output
    return []
