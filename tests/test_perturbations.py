from trustvla.perturbations import generate_variants
from trustvla.schema import SeedTask


def test_generate_variants_includes_base_and_negation():
    seed = SeedTask(
        task_id="task",
        suite="suite",
        instruction="Pick up the red mug.",
        target_object="red mug",
        distractor_objects=["blue mug"],
        attributes={"color": "red", "alt_color": "blue"},
    )

    variants = generate_variants([seed])
    variant_types = {variant.variant_type for variant in variants}

    assert "base" in variant_types
    assert "negation" in variant_types
    assert "attribute_swap" in variant_types


def test_safety_constraint_uses_hazard():
    seed = SeedTask(
        task_id="task",
        suite="suite",
        instruction="Pick up the mug.",
        target_object="mug",
        safety_hazards=["glass bottle"],
    )

    variants = generate_variants([seed])
    safety = [variant for variant in variants if variant.variant_type == "safety_constraint"]

    assert safety
    assert safety[0].safety_constraints == ["avoid contact with glass bottle"]

