"""Command line utilities for the TrustVLA starter scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from importlib.util import find_spec

from .adapters import (
    DummyGroundedPolicyAdapter,
    DummyPolicyAdapter,
    run_guarded_proposal,
)
from .detectors import enrich_rollouts_with_detections
from .hf_datasets import download_libero_suite_from_hf
from .io import read_json, read_jsonl, write_jsonl
from .metrics import summarize
from .paired_metrics import summarize_paired_trajectories
from .perturbations import generate_variants
from .report import write_comparison_report
from .schema import InstructionVariant, RolloutRecord, SeedTask
from .safety_gate import SafetyPolicyRegistry, export_safety_policy_draft
from .tradeoff_metrics import summarize_obedience_safety
from .validation import validate_benchmark
from .integrations.libero_openvla import (
    LiberoOpenVLAConfig,
    export_libero_seed_tasks,
    run_openvla_rollouts,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="trustvla")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.set_defaults(func=_doctor)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--seed-tasks", required=True)
    generate_parser.add_argument("--out", required=True)
    generate_parser.add_argument("--init-states", type=int, default=1)
    generate_parser.set_defaults(func=_generate)

    dummy_parser = subparsers.add_parser("dummy-rollouts")
    dummy_parser.add_argument("--benchmark", required=True)
    dummy_parser.add_argument("--out", required=True)
    dummy_parser.set_defaults(func=_dummy_rollouts)

    guard_dummy_parser = subparsers.add_parser("guard-dummy-rollouts")
    guard_dummy_parser.add_argument("--benchmark", required=True)
    guard_dummy_parser.add_argument("--safety-policies", required=True)
    guard_dummy_parser.add_argument("--out", required=True)
    guard_dummy_parser.set_defaults(func=_guard_dummy_rollouts)

    tradeoff_dummy_parser = subparsers.add_parser("tradeoff-dummy-rollouts")
    tradeoff_dummy_parser.add_argument("--benchmark", required=True)
    tradeoff_dummy_parser.add_argument("--safety-policies", required=True)
    tradeoff_dummy_parser.add_argument("--out-dir", required=True)
    tradeoff_dummy_parser.set_defaults(func=_tradeoff_dummy_rollouts)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--benchmark", required=False)
    score_parser.add_argument("--rollouts", required=True)
    score_parser.set_defaults(func=_score)

    tradeoff_score_parser = subparsers.add_parser("tradeoff-score")
    tradeoff_score_parser.add_argument("--benchmark", required=True)
    tradeoff_score_parser.add_argument("--rollouts", required=True)
    tradeoff_score_parser.set_defaults(func=_tradeoff_score)

    pair_score_parser = subparsers.add_parser("pair-score")
    pair_score_parser.add_argument("--benchmark", required=True)
    pair_score_parser.add_argument("--rollouts", required=True)
    pair_score_parser.add_argument("--difference-threshold", type=float, default=0.05)
    pair_score_parser.add_argument("--prefix-steps", type=int, default=10)
    pair_score_parser.set_defaults(func=_pair_score)

    validate_parser = subparsers.add_parser("validate-benchmark")
    validate_parser.add_argument("--benchmark", required=True)
    validate_parser.add_argument("--safety-policies", required=True)
    validate_parser.set_defaults(func=_validate_benchmark)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--benchmark", required=True)
    compare_parser.add_argument("--rollout", action="append", required=True)
    compare_parser.add_argument("--out", required=True)
    compare_parser.set_defaults(func=_compare)

    detect_parser = subparsers.add_parser("detect-rollout-events")
    detect_parser.add_argument("--benchmark", required=True)
    detect_parser.add_argument("--rollouts", required=True)
    detect_parser.add_argument("--out", required=True)
    detect_parser.set_defaults(func=_detect_rollout_events)

    hf_parser = subparsers.add_parser("download-libero-hf")
    hf_parser.add_argument("--suite", default="libero_object")
    hf_parser.add_argument("--local-dir", required=True)
    hf_parser.add_argument("--repo-id", default="yifengzhu-hf/LIBERO-datasets")
    hf_parser.add_argument("--revision")
    hf_parser.add_argument("--dry-run", action="store_true")
    hf_parser.set_defaults(func=_download_libero_hf)

    export_libero_parser = subparsers.add_parser("export-libero-seeds")
    export_libero_parser.add_argument("--suite", default="libero_object")
    export_libero_parser.add_argument("--out", required=True)
    export_libero_parser.add_argument("--limit", type=int)
    export_libero_parser.set_defaults(func=_export_libero_seeds)

    export_policy_parser = subparsers.add_parser("export-safety-policies")
    export_policy_parser.add_argument("--seed-tasks", required=True)
    export_policy_parser.add_argument("--out", required=True)
    export_policy_parser.set_defaults(func=_export_safety_policies)

    run_openvla_parser = subparsers.add_parser("run-openvla-libero")
    run_openvla_parser.add_argument("--benchmark", required=True)
    run_openvla_parser.add_argument("--out", required=True)
    run_openvla_parser.add_argument("--model-path", default="openvla/openvla-7b")
    run_openvla_parser.add_argument("--suite", default="libero_object")
    run_openvla_parser.add_argument("--device", default="cuda:0")
    run_openvla_parser.add_argument("--torch-dtype", default="bfloat16")
    run_openvla_parser.add_argument("--image-key", default="agentview_image")
    run_openvla_parser.add_argument("--unnorm-key")
    run_openvla_parser.add_argument("--max-steps", type=int, default=300)
    run_openvla_parser.add_argument("--camera-height", type=int, default=256)
    run_openvla_parser.add_argument("--camera-width", type=int, default=256)
    run_openvla_parser.add_argument("--seed", type=int, default=0)
    run_openvla_parser.add_argument("--trace-dir", default="runs/openvla/traces")
    run_openvla_parser.add_argument("--policy-id", default="openvla")
    run_openvla_parser.add_argument(
        "--grounding-mode",
        choices=["none", "language_emphasis"],
        default="none",
    )
    run_openvla_parser.add_argument("--guarded", action="store_true")
    run_openvla_parser.add_argument("--safety-policies")
    run_openvla_parser.add_argument("--resume", action="store_true")
    run_openvla_parser.set_defaults(func=_run_openvla_libero)

    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _doctor(args: argparse.Namespace) -> None:
    del args
    checks = {
        "libero": find_spec("libero") is not None,
        "torch": find_spec("torch") is not None,
        "transformers": find_spec("transformers") is not None,
        "PIL": find_spec("PIL") is not None,
        "numpy": find_spec("numpy") is not None,
    }
    for name, installed in checks.items():
        status = "ok" if installed else "missing"
        print(f"{name}: {status}")

    if not checks["libero"]:
        print("")
        print("LIBERO is missing. Local smoke tests can run, but real LIBERO/OpenVLA")
        print("commands such as export-libero-seeds and run-openvla-libero require")
        print("a GPU/simulation environment with LIBERO, robosuite, and MuJoCo installed.")


def _generate(args: argparse.Namespace) -> None:
    seed_records = read_json(args.seed_tasks)
    seed_tasks = [SeedTask.from_dict(record) for record in seed_records]
    variants = generate_variants(seed_tasks, init_states=args.init_states)
    write_jsonl(args.out, [variant.to_dict() for variant in variants])
    print(f"wrote {len(variants)} variants to {args.out}")


def _dummy_rollouts(args: argparse.Namespace) -> None:
    variants = [InstructionVariant.from_dict(record) for record in read_jsonl(args.benchmark)]
    adapter = DummyPolicyAdapter()
    rollouts = [adapter.run_case(variant).to_dict() for variant in variants]
    write_jsonl(args.out, rollouts)
    print(f"wrote {len(rollouts)} dummy rollouts to {args.out}")


def _guard_dummy_rollouts(args: argparse.Namespace) -> None:
    variants = [InstructionVariant.from_dict(record) for record in read_jsonl(args.benchmark)]
    adapter = DummyPolicyAdapter()
    registry = SafetyPolicyRegistry.from_path(args.safety_policies)
    rollouts = [
        run_guarded_proposal(
            variant,
            adapter.propose_case(variant),
            registry.for_scene(variant.scene_id),
        ).to_dict()
        for variant in variants
    ]
    write_jsonl(args.out, rollouts)
    print(f"wrote {len(rollouts)} guarded dummy rollouts to {args.out}")


def _tradeoff_dummy_rollouts(args: argparse.Namespace) -> None:
    variants = [InstructionVariant.from_dict(record) for record in read_jsonl(args.benchmark)]
    registry = SafetyPolicyRegistry.from_path(args.safety_policies)
    visual_prior = DummyPolicyAdapter()
    grounded = DummyGroundedPolicyAdapter()
    out_dir = Path(args.out_dir)

    visual_rollouts = [visual_prior.run_case(variant).to_dict() for variant in variants]
    grounded_rollouts = [grounded.run_case(variant).to_dict() for variant in variants]
    guarded_rollouts = [
        run_guarded_proposal(
            variant,
            grounded.propose_case(variant),
            registry.for_scene(variant.scene_id),
        ).to_dict()
        for variant in variants
    ]
    write_jsonl(out_dir / "dummy_visual_prior.jsonl", visual_rollouts)
    write_jsonl(out_dir / "dummy_grounded.jsonl", grounded_rollouts)
    write_jsonl(out_dir / "dummy_grounded_guarded.jsonl", guarded_rollouts)
    print(f"wrote synthetic trade-off pilot rollouts to {out_dir}")


def _score(args: argparse.Namespace) -> None:
    rollouts = [RolloutRecord.from_dict(record) for record in read_jsonl(args.rollouts)]
    if args.benchmark:
        variants = [InstructionVariant.from_dict(record) for record in read_jsonl(args.benchmark)]
    else:
        variants = _variants_from_rollout_directory(args.rollouts)
    summaries = summarize(variants, rollouts)
    print(json.dumps([summary.to_dict() for summary in summaries], indent=2, sort_keys=True))


def _tradeoff_score(args: argparse.Namespace) -> None:
    variants = [InstructionVariant.from_dict(record) for record in read_jsonl(args.benchmark)]
    rollouts = [RolloutRecord.from_dict(record) for record in read_jsonl(args.rollouts)]
    summaries = summarize_obedience_safety(variants, rollouts)
    print(json.dumps([summary.to_dict() for summary in summaries], indent=2, sort_keys=True))


def _pair_score(args: argparse.Namespace) -> None:
    variants = [InstructionVariant.from_dict(record) for record in read_jsonl(args.benchmark)]
    rollouts = [RolloutRecord.from_dict(record) for record in read_jsonl(args.rollouts)]
    summaries = summarize_paired_trajectories(
        variants,
        rollouts,
        difference_threshold=args.difference_threshold,
        prefix_steps=args.prefix_steps,
    )
    print(json.dumps([summary.to_dict() for summary in summaries], indent=2, sort_keys=True))


def _validate_benchmark(args: argparse.Namespace) -> None:
    variants = [InstructionVariant.from_dict(record) for record in read_jsonl(args.benchmark)]
    registry = SafetyPolicyRegistry.from_path(args.safety_policies)
    issues = validate_benchmark(variants, registry)
    errors = sum(issue.severity == "error" for issue in issues)
    warnings = sum(issue.severity == "warning" for issue in issues)
    print(
        json.dumps(
            {
                "num_cases": len(variants),
                "errors": errors,
                "warnings": warnings,
                "issues": [issue.to_dict() for issue in issues],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if errors:
        raise SystemExit(1)


def _compare(args: argparse.Namespace) -> None:
    specs = [_parse_rollout_spec(spec) for spec in args.rollout]
    write_comparison_report(args.benchmark, specs, args.out)
    print(f"wrote comparison report to {args.out}")


def _detect_rollout_events(args: argparse.Namespace) -> None:
    enrich_rollouts_with_detections(args.benchmark, args.rollouts, args.out)
    print(f"wrote detector-enriched rollouts to {args.out}")


def _download_libero_hf(args: argparse.Namespace) -> None:
    plan = download_libero_suite_from_hf(
        suite=args.suite,
        local_dir=args.local_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        dry_run=args.dry_run,
    )
    print(json.dumps(plan.__dict__, indent=2, sort_keys=True))
    if args.dry_run:
        print("dry run only; no files downloaded")
    else:
        print(f"downloaded {args.suite} to {args.local_dir}")


def _parse_rollout_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError("--rollout must use LABEL=PATH format")
    label, path = spec.split("=", 1)
    return label, path


def _export_libero_seeds(args: argparse.Namespace) -> None:
    export_libero_seed_tasks(args.suite, args.out, args.limit)
    print(f"wrote LIBERO seed-task draft to {args.out}")


def _export_safety_policies(args: argparse.Namespace) -> None:
    records = read_json(args.seed_tasks)
    seed_tasks = [SeedTask.from_dict(record) for record in records]
    export_safety_policy_draft(seed_tasks, args.out)
    print(f"wrote reviewable safety-policy draft to {args.out}")


def _run_openvla_libero(args: argparse.Namespace) -> None:
    policy_id = args.policy_id
    if args.grounding_mode != "none" and policy_id == "openvla":
        policy_id = f"openvla+{args.grounding_mode}"
    config = LiberoOpenVLAConfig(
        model_path=args.model_path,
        suite_name=args.suite,
        device=args.device,
        torch_dtype=args.torch_dtype,
        image_key=args.image_key,
        unnorm_key=args.unnorm_key,
        max_steps=args.max_steps,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        seed=args.seed,
        trace_dir=args.trace_dir,
        policy_id=policy_id,
        grounding_mode=args.grounding_mode,
    )
    run_openvla_rollouts(
        args.benchmark,
        args.out,
        config,
        guarded=args.guarded,
        safety_policies_path=args.safety_policies,
        resume=args.resume,
    )
    print(f"wrote OpenVLA/LIBERO rollouts to {args.out}")


def _variants_from_rollout_directory(rollout_path: str) -> list[InstructionVariant]:
    candidate = Path(rollout_path).with_name("generated_benchmark.jsonl")
    if not candidate.exists():
        return []
    return [InstructionVariant.from_dict(record) for record in read_jsonl(candidate)]


if __name__ == "__main__":
    main()
