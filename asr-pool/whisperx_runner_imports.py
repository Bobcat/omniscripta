from __future__ import annotations

import gc
from typing import Any


def _cleanup_torch(torch_mod: Any) -> None:
  gc.collect()
  try:
    if torch_mod.cuda.is_available():
      torch_mod.cuda.empty_cache()
  except Exception:
    pass


def _as_positive_int(value: Any) -> int | None:
  try:
    v = int(value)
  except Exception:
    return None
  if v <= 0:
    return None
  return v


def _apply_torch_thread_tuning(
  torch_mod: Any,
  *,
  torch_num_threads: int | None,
  torch_num_interop_threads: int | None,
) -> dict[str, Any]:
  errors: list[str] = []
  if torch_num_threads is not None:
    try:
      torch_mod.set_num_threads(int(torch_num_threads))
    except Exception as e:
      errors.append(f"set_num_threads_failed:{e!r}")
  if torch_num_interop_threads is not None:
    try:
      torch_mod.set_num_interop_threads(int(torch_num_interop_threads))
    except Exception as e:
      errors.append(f"set_num_interop_threads_failed:{e!r}")

  effective_threads: int | None = None
  effective_interop: int | None = None
  try:
    effective_threads = int(torch_mod.get_num_threads())
  except Exception as e:
    errors.append(f"get_num_threads_failed:{e!r}")
  try:
    effective_interop = int(torch_mod.get_num_interop_threads())
  except Exception as e:
    errors.append(f"get_num_interop_threads_failed:{e!r}")

  return {
    "requested": {
      "torch_num_threads": torch_num_threads,
      "torch_num_interop_threads": torch_num_interop_threads,
    },
    "effective": {
      "torch_num_threads": effective_threads,
      "torch_num_interop_threads": effective_interop,
    },
    "errors": errors,
  }
