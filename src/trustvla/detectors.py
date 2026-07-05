"""Object and contact event detection from rollout traces.

Real robotics simulators expose object/contact information through different info keys.
This module starts with a conservative trace-level heuristic that can be refined once
actual LIBERO traces are available.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl
from .schema import InstructionVariant, RolloutRecord


OBJECT_KEYS = {
    "object",
    "objects",
    "target",
    "target_object",
    "grasped_object",
    "held_object",
    "picked_object",
    "touched_object",
    "contact_object",
    "contact_objects",
    "collision_object",
    "collision_objects",
}

CONTACT_TOKENS = {"contact", "collision", "touch", "touched", "grasp", "grasped", "held"}


@dataclass(frozen=True)
class DetectionResult:
    selected_target: str | None = None
    safety_events: list[str] = field(default_factory=list)
    matched_objects: list[str] = field(default_factory=list)


class TraceObjectContactDetector:
    """Infer selected objects and contact violations from a rollout trace."""

    def detect(
        self,
        variant: InstructionVariant,
        trace: dict[str, Any] | None,
    ) -> DetectionResult:
        info_records = _extract_info_records(trace)
        candidates = _candidate_objects(variant)
        matched_objects: list[str] = []
        contact_events: list[str] = []

        for info in info_records:
            matched_objects.extend(_objects_in_info(info, candidates))
            contact_events.extend(_contact_events_in_info(info, variant))

        selected_target = _choose_selected_target(matched_objects, variant)
        safety_events = sorted(set(contact_events))
        return DetectionResult(
            selected_target=selected_target,
            safety_events=safety_events,
            matched_objects=sorted(set(matched_objects)),
        )


def enrich_rollouts_with_detections(
    benchmark_path: str,
    rollouts_path: str,
    out_path: str,
) -> None:
    variants = {
        variant.case_id: variant
        for variant in (
            InstructionVariant.from_dict(record) for record in read_jsonl(benchmark_path)
        )
    }
    detector = TraceObjectContactDetector()
    enriched: list[dict[str, Any]] = []

    for record in read_jsonl(rollouts_path):
        rollout = RolloutRecord.from_dict(record)
        variant = variants.get(rollout.case_id)
        if variant is None:
            enriched.append(rollout.to_dict())
            continue

        trace = _read_trace(rollout.action_trace_path)
        detection = detector.detect(variant, trace)
        selected_target = rollout.selected_target or detection.selected_target
        safety_events = sorted(set([*rollout.safety_events, *detection.safety_events]))

        enriched.append(
            RolloutRecord(
                case_id=rollout.case_id,
                policy_id=rollout.policy_id,
                success=rollout.success,
                executed_behavior=rollout.executed_behavior,
                selected_target=selected_target,
                safety_events=safety_events,
                verifier_decision=rollout.verifier_decision,
                steps=rollout.steps,
                action_trace_path=rollout.action_trace_path,
                notes=_append_note(
                    rollout.notes,
                    f"detector_matched={','.join(detection.matched_objects) or 'none'}",
                ),
            ).to_dict()
        )

    write_jsonl(out_path, enriched)


def _extract_info_records(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    if trace is None:
        return []
    infos = trace.get("infos", [])
    return [info for info in infos if isinstance(info, dict)]


def _candidate_objects(variant: InstructionVariant) -> list[str]:
    objects: list[str] = []
    for value in [
        variant.expected_target,
        variant.metadata.get("base_target"),
        *variant.forbidden_targets,
        *variant.metadata.get("possible_objects", []),
        *variant.metadata.get("hazards", []),
    ]:
        if isinstance(value, str) and value:
            objects.append(value)
    return _dedupe_preserve_order(objects)


def _objects_in_info(info: dict[str, Any], candidates: list[str]) -> list[str]:
    text_fragments = _contact_relevant_fragments(info)
    matched: list[str] = []
    for candidate in candidates:
        normalized_candidate = _normalize(candidate)
        if any(normalized_candidate in _normalize(fragment) for fragment in text_fragments):
            matched.append(candidate)
    return matched


def _contact_relevant_fragments(info: dict[str, Any]) -> list[str]:
    fragments: list[str] = []
    for key, value in _flatten_info(info):
        normalized_key = _normalize(key)
        if key in OBJECT_KEYS or any(token in normalized_key for token in CONTACT_TOKENS):
            fragments.append(key)
            fragments.extend(_value_fragments(value))
    return fragments


def _contact_events_in_info(info: dict[str, Any], variant: InstructionVariant) -> list[str]:
    events: list[str] = []
    forbidden = set(variant.forbidden_targets)
    hazards = set(variant.metadata.get("hazards", []))
    for constraint in variant.safety_constraints:
        if constraint.startswith("avoid contact with "):
            hazards.add(constraint.removeprefix("avoid contact with "))

    contact_fragments = _contact_relevant_fragments(info)
    for obj in sorted(forbidden | hazards):
        normalized_obj = _normalize(obj)
        if any(normalized_obj in _normalize(fragment) for fragment in contact_fragments):
            if obj in hazards:
                events.append(f"contact_with_hazard:{obj}")
            if obj in forbidden:
                events.append(f"contact_with_forbidden:{obj}")
    return events


def _choose_selected_target(
    matched_objects: list[str],
    variant: InstructionVariant,
) -> str | None:
    if not matched_objects:
        return None

    priority = [
        variant.expected_target,
        *variant.forbidden_targets,
        variant.metadata.get("base_target"),
    ]
    for obj in priority:
        if isinstance(obj, str) and obj in matched_objects:
            return obj
    return matched_objects[-1]


def _flatten_info(info: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    for key, value in info.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        pairs.append((full_key, value))
        if isinstance(value, dict):
            pairs.extend(_flatten_info(value, full_key))
    return pairs


def _value_fragments(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)] if bool(value) else []
    if isinstance(value, (list, tuple, set)):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_value_fragments(item))
        return fragments
    if isinstance(value, dict):
        fragments: list[str] = []
        for key, item in value.items():
            fragments.append(str(key))
            fragments.extend(_value_fragments(item))
        return fragments
    return [str(value)]


def _read_trace(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    trace_path = Path(path)
    if not trace_path.exists():
        return None
    return json.loads(trace_path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _append_note(note: str, addition: str) -> str:
    if not note:
        return addition
    return f"{note} | {addition}"

