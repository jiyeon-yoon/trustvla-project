"""Verify the Docker image has the runtime packages TrustVLA needs."""

from __future__ import annotations

import importlib
import sys


CHECKS = [
    ("libero", "libero"),
    ("libero.benchmark", "libero.libero.benchmark"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("PIL", "PIL"),
    ("numpy", "numpy"),
]


def main() -> int:
    failures: list[str] = []
    for label, module_name in CHECKS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - build-time diagnostic.
            print(f"{label}: import failed: {exc!r}", flush=True)
            failures.append(label)
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"{label}: import ok ({version})", flush=True)

    if failures:
        print("missing runtime packages: " + ", ".join(failures), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
