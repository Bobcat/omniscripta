# ASR Service Contract `asr_v1` (Draft)

Date: 2026-02-24  
Status: Draft (implementation target)

## Purpose

Define a transport-agnostic request/response contract for a shared ASR service that can serve both:

- semi-live chunk transcription (`live_chunk`)
- batch/upload transcription (`upload_audio`)

This contract is intentionally separate from use-case orchestration.

## Design Principles

1. Single contract, multiple profiles
2. Profile defaults + whitelisted overrides
3. Transport-agnostic (IPC first, HTTP later optional)
4. Normalized outputs (text / segments / artifacts / timings)

## Request (`asr_v1`)

### Top-level fields

- `schema_version` (string): must be `asr_v1`
- `request_id` (string): caller-generated id for correlation
- `profile_id` (string): execution profile (e.g. `live_fast`, `upload_full`)
- `priority` (string): `interactive` | `normal` | `background`
- `audio` (object): input reference and metadata
- `options` (object): requested ASR options (subject to profile policy)
- `context` (object, optional): use-case metadata for correlation/debug
- `outputs` (object, optional): which outputs are desired

### `audio`

Exactly one audio source should be provided. Initial implementation can support `local_path` only.

- `local_path` (string, optional): absolute path on local filesystem
- `inline_base64` (string, optional, reserved): base64-encoded audio payload for future HTTP/gRPC transport
- `blob_ref` (string, optional, reserved): opaque storage reference for future use
- `sample_rate_hz` (integer, optional)
- `channels` (integer, optional)
- `format` (string, optional): e.g. `wav`, `mp3`
- `duration_ms` (integer, optional)

Initial implementation rule:
- require `audio.local_path`
- reject `inline_base64` / `blob_ref` as not implemented yet

### `options`

Common ASR options. The service resolves these against `profile_id`.

- `language` (string, optional): e.g. `en`, `nl`
- `align_enabled` (boolean, optional)
- `diarize_enabled` (boolean, optional)
- `speaker_mode` (string, optional): `none` | `auto` | `fixed`
- `min_speakers` (integer, optional)
- `max_speakers` (integer, optional)
- `initial_prompt` (string, optional)
- `timestamps_mode` (string, optional): `segment` | `word` | `none`

### `context` (examples)

- `source_kind`: `live_chunk` | `upload_audio`
- `live_session_id`
- `live_chunk_index`
- `t0_offset_ms`
- `job_id`
- `user_id` (future, if needed)

### `outputs`

- `text` (boolean, default `true`)
- `segments` (boolean, default `true`)
- `srt` (boolean, default `true`)
- `srt_inline` (boolean, default `false`): request inline `result.srt_text` in addition to/instead of artifact path
- `word_timestamps` (boolean, default `false`)

## Response (`asr_v1`)

### Top-level fields

- `schema_version` (string): `asr_v1`
- `request_id` (string)
- `ok` (boolean)
- `profile_id` (string)
- `resolved_options` (object): actual options used after profile policy/defaults
- `result` (object, present when `ok=true`)
- `error` (object, present when `ok=false`)
- `warnings` (array of strings, optional)
- `timings` (object, optional)
- `runtime` (object, optional): backend/runner metadata

### `result`

- `text` (string, optional)
- `segments` (array, optional)
- `audio_processed_ms` (integer, optional): actual processed audio duration as observed by the service
- `srt_text` (string, optional): typically returned when `outputs.srt_inline=true`
- `artifacts` (object, optional)

### `result.segments[]`

Normalized segment shape.

Timestamp semantics:
- `t0_ms` / `t1_ms` are absolute if the caller provides an offset (e.g. semilive orchestrator applies `t0_offset_ms`)
- otherwise they are relative to the provided audio input

- `segment_id` (string, optional)
- `text` (string)
- `t0_ms` (integer)
- `t1_ms` (integer)
- `speaker` (string, optional)
- `confidence` (number, optional): may be omitted/null for engines that do not emit segment confidence (initial WhisperX path)

### `result.artifacts`

- `srt_path` (string, optional)
- `json_path` (string, optional)
- `speaker_lines_path` (string, optional)

### `timings`

Timings should be best-effort and stable across transports.

- `total_s` (number)
- `queue_wait_s` (number, optional)
- `prepare_s` (number, optional)
- `transcribe_s` (number, optional)
- `align_s` (number, optional)
- `diarize_s` (number, optional)
- `finalize_s` (number, optional)

### `runtime`

Useful for debugging/observability:

- `backend` (string): e.g. `whisperx`
- `runner_kind` (string): `oneshot_subprocess` | `persistent_local`
- `runner_reused` (boolean, optional)
- `hardware_key` (string, optional)
- `device` (string, optional)
- `model` (string, optional)

### `error`

- `code` (string): stable error code
- `message` (string): human-readable summary
- `retryable` (boolean, optional)
- `details` (object, optional)

## Profiles (initial)

Profiles are service-defined policies, not client-owned.

### `live_fast`

Intent:
- low latency for semilive chunk transcription

Defaults (initial target):
- `align_enabled=false`
- `diarize_enabled=false`
- `speaker_mode=none`
- `timestamps_mode=segment`

Allowed overrides (initial target):
- `language`
- `align_enabled` (for A/B testing)
- `initial_prompt`

### `upload_full`

Intent:
- richer, higher-quality outputs for full uploads

Defaults (initial target):
- `align_enabled=true`
- `diarize_enabled` depends on upload options
- `timestamps_mode=segment` (or `word` later)

Allowed overrides (initial target):
- `language`
- `speaker_mode`
- `min_speakers`
- `max_speakers`
- `diarize_enabled`

## Profile Policy Rules

The service must:

1. Resolve profile defaults first.
2. Apply only allowed overrides for that profile.
3. Reject unknown `profile_id` with a clear error (no implicit fallback profile).
4. Reject unknown/forbidden overrides with a clear error.
5. Return `resolved_options` in the response.

This keeps client behavior debuggable and prevents parameter drift.

## Example Request (live chunk)

```json
{
  "schema_version": "asr_v1",
  "request_id": "req_live_000123",
  "profile_id": "live_fast",
  "priority": "interactive",
  "audio": {
    "local_path": "/home/gunnar/projects/transcribe-dev/data/demo_jobs/running/job_x/upload/chunk_0012.wav",
    "format": "wav",
    "sample_rate_hz": 16000,
    "channels": 1
  },
  "options": {
    "language": "en",
    "align_enabled": false
  },
  "context": {
    "source_kind": "live_chunk",
    "live_session_id": "live_20260224T013451Z_dfa994ab",
    "live_chunk_index": 12,
    "t0_offset_ms": 84220
  },
  "outputs": {
    "text": true,
    "segments": true,
    "srt": true,
    "srt_inline": true
  }
}
```

## Example Response (success)

```json
{
  "schema_version": "asr_v1",
  "request_id": "req_live_000123",
  "ok": true,
  "profile_id": "live_fast",
  "resolved_options": {
    "language": "en",
    "align_enabled": false,
    "diarize_enabled": false,
    "speaker_mode": "none",
    "timestamps_mode": "segment"
  },
  "result": {
    "text": "I think the next step is to reduce the startup overhead.",
    "segments": [
      {
        "segment_id": "s0001",
        "text": "I think the next step is to reduce the startup overhead.",
        "t0_ms": 84220,
        "t1_ms": 87920
      }
    ],
    "audio_processed_ms": 3700,
    "srt_text": "1\n00:01:24,220 --> 00:01:27,920\nI think the next step is to reduce the startup overhead.\n",
    "artifacts": {
      "srt_path": "/.../chunk_0012.srt"
    }
  },
  "timings": {
    "total_s": 1.42,
    "transcribe_s": 1.12,
    "finalize_s": 0.01
  },
  "runtime": {
    "backend": "whisperx",
    "runner_kind": "persistent_local",
    "runner_reused": true,
    "device": "cuda",
    "model": "large-v3"
  },
  "warnings": []
}
```

## Example Response (error)

```json
{
  "schema_version": "asr_v1",
  "request_id": "req_live_000123",
  "ok": false,
  "profile_id": "live_fast",
  "resolved_options": {
    "language": "en",
    "align_enabled": false,
    "diarize_enabled": false,
    "speaker_mode": "none",
    "timestamps_mode": "segment"
  },
  "error": {
    "code": "ASR_RUNTIME_FAILURE",
    "message": "WhisperX runner exited unexpectedly",
    "retryable": true
  },
  "warnings": []
}
```

## Notes for Current Codebase Integration

Near-term mapping in this repository:

- `portal-api/live_chunk_transcribe.py` can become an adapter from semilive chunk jobs to `asr_v1` requests.
- `worker/pipeline_live_chunk.py` is the first target for consuming a persistent ASR backend under this contract.
- `upload_audio` pipeline can migrate later without changing live orchestration semantics.
