"""Command line utilities for the TrustVLA starter scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from importlib.util import find_spec

from .adapters import DummyPolicyAdapter
from .detectors import enrich_rollouts_with_detections
from .hf_datasets import download_libero_suite_from_hf
from .io import read_json, read_jsonl, write_jsonl
from .metrics import summarize
from .perturbations import generate_variants
from .report import write_comparison_report
from .schema import InstructionVariant, RolloutRecord, SeedTask
from .verifier import RuntimeVerifier
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
    generate_parser.set_defaults(func=_generate)

    dummy_parser = subparsers.add_parser("dummy-rollouts")
    dummy_parser.add_argument("--benchmark", required=True)
    dummy_parser.add_argument("--out", required=True)
    dummy_parser.set_defaults(func=_dummy_rollouts)

    guard_dummy_parser = subparsers.add_parser("guard-dummy-rollouts")
    guard_dummy_parser.add_argument("--benchmark", required=True)
    guard_dummy_parser.add_argument("--out", required=True)
    guard_dummy_parser.set_defaults(func=_guard_dummy_rollouts)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--benchmark", required=False)
    score_parser.add_argument("--rollouts", required=True)
    score_parser.set_defaults(func=_score)

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
    run_openvla_parser.add_argument("--guarded", action="store_true")
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
    variants = generate_variants(seed_tasks)
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
    verifier = RuntimeVerifier()
    rollouts = [
        verifier.apply(variant, adapter.propose_case(variant)).to_dict()
        for variant in variants
    ]
    write_jsonl(args.out, rollouts)
    print(f"wrote {len(rollouts)} guarded dummy rollouts to {args.out}")


def _score(args: argparse.Namespace) -> None:
    rollouts = [RolloutRecord.from_dict(record) for record in read_jsonl(args.rollouts)]
    if args.benchmark:
        variants = [InstructionVariant.from_dict(record) for record in read_jsonl(args.benchmark)]
    else:
        variants = _variants_from_rollout_directory(args.rollouts)
    summaries = summarize(variants, rollouts)
    print(json.dumps([summary.to_dict() for summary in summaries], indent=2, sort_keys=True))


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


def _run_openvla_libero(args: argparse.Namespace) -> None:
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
        policy_id=args.policy_id,
    )
    run_openvla_rollouts(args.benchmark, args.out, config, guarded=args.guarded)
    print(f"wrote OpenVLA/LIBERO rollouts to {args.out}")


def _variants_from_rollout_directory(rollout_path: str) -> list[InstructionVariant]:
    candidate = Path(rollout_path).with_name("generated_benchmark.jsonl")
    if not candidate.exists():
        return []
    return [InstructionVariant.from_dict(record) for record in read_jsonl(candidate)]


if __name__ == "__main__":
    main()
