"""Markdown reporting helpers for TrustVLA experiments."""

from __future__ import annotations

from pathlib import Path

from .io import read_jsonl
from .metrics import summarize
from .schema import InstructionVariant, RolloutRecord


def write_comparison_report(
    benchmark_path: str,
    rollout_specs: list[tuple[str, str]],
    out_path: str,
) -> None:
    variants = [InstructionVariant.from_dict(record) for record in read_jsonl(benchmark_path)]
    rows: list[dict[str, object]] = []
    for label, rollout_path in rollout_specs:
        rollouts = [RolloutRecord.from_dict(record) for record in read_jsonl(rollout_path)]
        for summary in summarize(variants, rollouts):
            row = summary.to_dict()
            row["label"] = label
            rows.append(row)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_to_markdown(rows), encoding="utf-8")


def _to_markdown(rows: list[dict[str, object]]) -> str:
    headers = [
        "label",
        "policy_id",
        "num_cases",
        "success_rate",
        "wrong_target_rate",
        "constraint_violation_rate",
        "unsafe_success_rate",
        "refusal_accuracy",
        "paired_action_compliance_rate",
    ]
    lines = [
        "# TrustVLA Experiment Report",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [_format_cell(row.get(header)) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "-"
    return str(value)

