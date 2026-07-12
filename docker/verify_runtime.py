"""Verify the Docker image has the core runtime packages TrustVLA needs.

LIBERO is checked as a warning here because importing the full simulator stack
can depend on runtime graphics/MuJoCo details that are only available once the
image is running on RunPod. The RunPod smoke test remains the final check for
LIBERO rollout readiness.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys


REQUIRED_IMPORTS = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("PIL", "PIL"),
    ("numpy", "numpy"),
]


def main() -> int:
    failures: list[str] = []
    for label, module_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - build-time diagnostic.
            print(f"{label}: import failed: {exc!r}", flush=True)
            failures.append(label)
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"{label}: import ok ({version})", flush=True)

    try:
        libero_version = importlib.metadata.version("libero")
    except importlib.metadata.PackageNotFoundError:
        print("libero: distribution not found; RunPod doctor must confirm this.", flush=True)
    else:
        print(f"libero: distribution installed ({libero_version})", flush=True)

    try:
        importlib.import_module("libero")
    except Exception as exc:  # noqa: BLE001 - build-time diagnostic.
        print(f"libero: import warning: {exc!r}", flush=True)
    else:
        print("libero: basic import ok", flush=True)

    if failures:
        print("missing runtime packages: " + ", ".join(failures), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
