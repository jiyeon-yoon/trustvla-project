from trustvla.hf_datasets import build_libero_hf_download_plan


def test_build_libero_hf_download_plan_uses_suite_pattern():
    plan = build_libero_hf_download_plan(
        suite="libero_object",
        local_dir="/workspace/LIBERO-datasets",
    )

    assert plan.repo_id == "yifengzhu-hf/LIBERO-datasets"
    assert plan.repo_type == "dataset"
    assert plan.allow_patterns == ["libero_object/*"]

