from __future__ import annotations

from typing import Any, Callable


def run_whisperx_phase_remote(
  *,
  request_payload: dict[str, Any],
  on_lifecycle_update: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
  """Worker pipeline phase wrapper for WhisperX ASR.

  The worker no longer executes WhisperX locally; it delegates this phase to
  the ASR pool service.
  """
  from asr_client_remote import transcribe_with_remote_pool

  return transcribe_with_remote_pool(
    request_payload=request_payload,
    on_lifecycle_update=on_lifecycle_update,
  )
