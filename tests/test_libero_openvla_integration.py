from trustvla.integrations.libero_openvla import (
    _format_openvla_prompt,
    _safe_case_id,
    _variant_task_id,
)
from trustvla.schema import InstructionVariant


def test_variant_task_id_uses_metadata():
    variant = InstructionVariant(
        case_id="case",
        seed_task_id="not_numeric",
        suite="libero_object",
        variant_type="base",
        instruction="Pick up the mug.",
        expected_target="mug",
        base_instruction="Pick up the mug.",
        metadata={"libero_task_id": 7},
    )

    assert _variant_task_id(variant) == 7


def test_openvla_prompt_contains_instruction():
    prompt = _format_openvla_prompt("Pick up the mug.")

    assert "Pick up the mug." in prompt
    assert prompt.startswith("In:")
    assert prompt.endswith(" Out:")


def test_safe_case_id_replaces_separators():
    assert _safe_case_id("task::base/0") == "task__base_0"

