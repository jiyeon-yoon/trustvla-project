"""LIBERO + OpenVLA rollout integration.

This module is intentionally dependency-light at import time. LIBERO, PyTorch, PIL, and
Transformers are imported only inside runtime methods so the rest of TrustVLA remains
usable on laptops without robot-learning dependencies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trustvla.detectors import TraceObjectContactDetector
from trustvla.io import read_jsonl, write_jsonl
from trustvla.schema import InstructionVariant, RolloutRecord, SeedTask
from trustvla.verifier import RuntimeVerifier


@dataclass(frozen=True)
class LiberoOpenVLAConfig:
    model_path: str
    suite_name: str = "libero_object"
    device: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    image_key: str = "agentview_image"
    unnorm_key: str | None = None
    max_steps: int = 300
    camera_height: int = 256
    camera_width: int = 256
    seed: int = 0
    save_traces: bool = True
    trace_dir: str = "runs/openvla/traces"
    policy_id: str = "openvla"


class LiberoOpenVLAAdapter:
    """Run OpenVLA in LIBERO for TrustVLA benchmark variants."""

    def __init__(self, config: LiberoOpenVLAConfig) -> None:
        self.config = config
        self.policy_id = config.policy_id
        self._model = None
        self._processor = None

    def run_case(self, variant: InstructionVariant) -> RolloutRecord:
        """Run one LIBERO episode using the variant instruction."""

        task_id = _variant_task_id(variant)
        init_state_id = int(variant.metadata.get("libero_init_state_id", 0))
        env, task_description = self._make_env(task_id, init_state_id)
        model, processor = self._load_openvla()

        rewards: list[float] = []
        info_records: list[dict[str, Any]] = []
        action_records: list[list[float]] = []
        safety_events: list[str] = []
        success = False
        steps = 0

        try:
            obs = env.reset()
            init_obs = self._set_init_state_if_needed(env, task_id, init_state_id)
            if init_obs is not None:
                obs = init_obs
            prompt = _format_openvla_prompt(variant.instruction)

            for step in range(self.config.max_steps):
                image = _obs_to_pil(obs, self.config.image_key)
                action = _predict_openvla_action(
                    model=model,
                    processor=processor,
                    prompt=prompt,
                    image=image,
                    device=self.config.device,
                    torch_dtype=self.config.torch_dtype,
                    unnorm_key=self.config.unnorm_key,
                )
                obs, reward, done, info = env.step(action)
                rewards.append(float(reward))
                action_records.append(_action_to_list(action))
                info_records.append(_jsonable_info(info))
                safety_events.extend(_extract_safety_events(info))
                success = success or _is_success(reward, info)
                steps = step + 1
                if done or success:
                    break
        finally:
            env.close()

        trace_path = None
        detection = TraceObjectContactDetector().detect(variant, {"infos": info_records})
        safety_events = sorted(set([*safety_events, *detection.safety_events]))
        if self.config.save_traces:
            trace_path = self._write_trace(
                variant=variant,
                task_id=task_id,
                task_description=task_description,
                rewards=rewards,
                actions=action_records,
                infos=info_records,
            )

        return RolloutRecord(
            case_id=variant.case_id,
            policy_id=self.policy_id,
            success=success,
            executed_behavior="execute",
            selected_target=detection.selected_target,
            safety_events=sorted(set(safety_events)),
            steps=steps,
            action_trace_path=trace_path,
            notes=(
                "Real LIBERO/OpenVLA rollout. "
                f"detector_matched={','.join(detection.matched_objects) or 'none'}"
            ),
        )

    def _load_openvla(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "OpenVLA rollout requires torch and transformers. Install the OpenVLA "
                "runtime dependencies in your GPU environment."
            ) from exc

        dtype = _torch_dtype(torch, self.config.torch_dtype)
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.config.device)
        self._model.eval()
        return self._model, self._processor

    def _make_env(self, task_id: int, init_state_id: int):
        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
            from libero.libero.utils import get_libero_path
        except ImportError as exc:
            raise RuntimeError(
                "LIBERO is not installed. Install LIBERO and its robosuite/MuJoCo "
                "dependencies in the rollout environment."
            ) from exc

        benchmark_dict = benchmark.get_benchmark_dict()
        if self.config.suite_name not in benchmark_dict:
            available = ", ".join(sorted(benchmark_dict))
            raise ValueError(
                f"Unknown LIBERO suite '{self.config.suite_name}'. Available: {available}"
            )
        task_suite = benchmark_dict[self.config.suite_name]()
        task = task_suite.get_task(task_id)
        bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=self.config.camera_height,
            camera_widths=self.config.camera_width,
        )
        env.seed(self.config.seed + task_id * 1000 + init_state_id)
        return env, task.language

    def _set_init_state_if_needed(self, env, task_id: int, init_state_id: int):
        try:
            from libero.libero import benchmark
        except ImportError:
            return

        task_suite = benchmark.get_benchmark_dict()[self.config.suite_name]()
        init_states = task_suite.get_task_init_states(task_id)
        if init_states is not None and len(init_states) > 0:
            return env.set_init_state(init_states[init_state_id % len(init_states)])
        return None

    def _write_trace(
        self,
        variant: InstructionVariant,
        task_id: int,
        task_description: str,
        rewards: list[float],
        actions: list[list[float]],
        infos: list[dict[str, Any]],
    ) -> str:
        trace_dir = Path(self.config.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"{_safe_case_id(variant.case_id)}.json"
        payload = {
            "case_id": variant.case_id,
            "task_id": task_id,
            "task_description": task_description,
            "instruction": variant.instruction,
            "rewards": rewards,
            "actions": actions,
            "infos": infos,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return str(path)


def export_libero_seed_tasks(
    suite_name: str,
    out_path: str,
    limit: int | None = None,
) -> None:
    """Export LIBERO tasks into a seed-task draft for human annotation."""

    try:
        from libero.libero import benchmark
    except ImportError as exc:
        raise RuntimeError("LIBERO is not installed in this environment.") from exc

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    num_tasks = task_suite.n_tasks if hasattr(task_suite, "n_tasks") else len(task_suite.tasks)
    if limit is not None:
        num_tasks = min(num_tasks, limit)

    seeds: list[dict[str, Any]] = []
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        seeds.append(
            {
                "task_id": f"{suite_name}_{task_id}",
                "suite": suite_name,
                "scene_id": f"{suite_name}:{task_id}",
                "instruction": task.language,
                "target_object": "TODO_ANNOTATE_TARGET",
                "possible_objects": [],
                "distractor_objects": [],
                "absent_objects": [],
                "ambiguous_targets": [],
                "safety_hazards": [],
                "metadata": {"libero_task_id": task_id},
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(seeds, indent=2, ensure_ascii=False), encoding="utf-8")


def run_openvla_rollouts(
    benchmark_path: str,
    out_path: str,
    config: LiberoOpenVLAConfig,
    guarded: bool = False,
) -> None:
    variants = [InstructionVariant.from_dict(record) for record in read_jsonl(benchmark_path)]
    adapter = LiberoOpenVLAAdapter(config)
    verifier = RuntimeVerifier()
    rollouts: list[dict[str, Any]] = []

    for variant in variants:
        if guarded and variant.expected_behavior in {"no_op", "clarification"}:
            proposal = _stop_case_proposal(config.policy_id, variant)
            rollout = verifier.apply(variant, proposal)
        else:
            rollout = adapter.run_case(variant)
        rollouts.append(rollout.to_dict())

    write_jsonl(out_path, rollouts)


def _stop_case_proposal(policy_id: str, variant: InstructionVariant):
    from trustvla.schema import ActionProposal

    return ActionProposal(
        case_id=variant.case_id,
        policy_id=policy_id,
        proposed_behavior="execute",
        proposed_target=variant.metadata.get("base_target"),
        notes="Coarse guard proposal for stop-required instruction.",
    )


def _variant_task_id(variant: InstructionVariant) -> int:
    if "libero_task_id" in variant.metadata:
        return int(variant.metadata["libero_task_id"])
    if variant.seed_task_id.rsplit("_", 1)[-1].isdigit():
        return int(variant.seed_task_id.rsplit("_", 1)[-1])
    raise ValueError(
        f"Variant {variant.case_id} lacks metadata.libero_task_id and cannot be mapped to LIBERO."
    )


def _format_openvla_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction}?\n Out:"


def _predict_openvla_action(
    model,
    processor,
    prompt: str,
    image,
    device: str,
    torch_dtype: str,
    unnorm_key: str | None,
):
    import torch

    dtype = _torch_dtype(torch, torch_dtype)
    inputs = processor(prompt, image).to(device, dtype=dtype)
    kwargs = {"do_sample": False}
    if unnorm_key:
        kwargs["unnorm_key"] = unnorm_key
    return model.predict_action(**inputs, **kwargs)


def _obs_to_pil(obs: dict[str, Any], image_key: str):
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL and numpy are required for image observations.") from exc

    if image_key not in obs:
        available = ", ".join(sorted(obs.keys()))
        raise KeyError(f"Observation key '{image_key}' not found. Available keys: {available}")
    image = obs[image_key]
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = image.transpose(1, 2, 0)
    return Image.fromarray(image)


def _torch_dtype(torch_module, dtype_name: str):
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _action_to_list(action) -> list[float]:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    if hasattr(action, "tolist"):
        return action.tolist()
    return list(action)


def _jsonable_info(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {"raw_info": str(info)}
    output: dict[str, Any] = {}
    for key, value in info.items():
        try:
            json.dumps(value)
            output[key] = value
        except TypeError:
            output[key] = str(value)
    return output


def _extract_safety_events(info: Any) -> list[str]:
    if not isinstance(info, dict):
        return []
    events: list[str] = []
    for key, value in info.items():
        lowered = key.lower()
        if any(token in lowered for token in ["collision", "contact", "unsafe", "violation"]):
            if bool(value):
                events.append(key)
    return events


def _is_success(reward: float, info: Any) -> bool:
    if float(reward) > 0:
        return True
    if isinstance(info, dict):
        for key in ["success", "is_success", "task_success"]:
            if key in info and bool(info[key]):
                return True
    return False


def _safe_case_id(case_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in case_id)
