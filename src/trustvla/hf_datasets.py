"""Hugging Face helpers for selective LIBERO dataset downloads."""

from __future__ import annotations

from dataclasses import dataclass


LIBERO_HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
LIBERO_SUITES = {
    "libero_object",
    "libero_spatial",
    "libero_goal",
    "libero_90",
    "libero_10",
}


@dataclass(frozen=True)
class HFDatasetDownloadPlan:
    repo_id: str
    repo_type: str
    suite: str
    local_dir: str
    allow_patterns: list[str]
    revision: str | None = None


def build_libero_hf_download_plan(
    suite: str,
    local_dir: str,
    repo_id: str = LIBERO_HF_REPO_ID,
    revision: str | None = None,
) -> HFDatasetDownloadPlan:
    """Return a selective download plan for one LIBERO suite."""

    if suite not in LIBERO_SUITES:
        allowed = ", ".join(sorted(LIBERO_SUITES))
        raise ValueError(f"Unknown LIBERO suite '{suite}'. Choose one of: {allowed}")

    return HFDatasetDownloadPlan(
        repo_id=repo_id,
        repo_type="dataset",
        suite=suite,
        local_dir=local_dir,
        allow_patterns=[f"{suite}/*"],
        revision=revision,
    )


def download_libero_suite_from_hf(
    suite: str,
    local_dir: str,
    repo_id: str = LIBERO_HF_REPO_ID,
    revision: str | None = None,
    dry_run: bool = False,
) -> HFDatasetDownloadPlan:
    """Download one LIBERO suite from Hugging Face, or return the plan in dry-run mode."""

    plan = build_libero_hf_download_plan(
        suite=suite,
        local_dir=local_dir,
        repo_id=repo_id,
        revision=revision,
    )
    if dry_run:
        return plan

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for HF dataset download. Install it with "
            "`pip install huggingface_hub` in the RunPod environment."
        ) from exc

    snapshot_download(
        repo_id=plan.repo_id,
        repo_type=plan.repo_type,
        revision=plan.revision,
        local_dir=plan.local_dir,
        allow_patterns=plan.allow_patterns,
    )
    return plan

