# Timestamped PC CSV Migration Plan

## Decision

Replace the old `.pc` format completely.

There will be one `.pc` export button in Omniscripta, and it will export the new CSV-like
format. The old `p,text` / `c,text` format is allowed to break. Existing sample `.pc` files
in consuming apps are allowed to become invalid and must be replaced or migrated as part of
the same change.

This is a coordinated breaking change: Omniscripta export and known `.pc` consumers move to
the new format together. No compatibility parser or fallback path should remain.

## New Format

Use one CSV-style row per replay event:

```csv
kind,speech_start_ms,speech_end_ms,text
```

Rows with `kind == p` represent preview state. Rows with `kind == c` represent committed
source text. Consumers that only care about translation/TTS should process only `kind == c`.

## Semantics

`speech_start_ms` and `speech_end_ms` are the best available source-audio speech interval for
the event.

Timing fields are required. If Omniscripta cannot produce timing for a non-empty `p` or `c`
row, that is a bug and the export should fail loudly.

For normal ASR segment commits:

- `speech_start_ms` comes from the committed segment or commit `t0_ms`.
- `speech_end_ms` comes from the committed segment or commit `t1_ms`.

For preview rows:

- `speech_start_ms` comes from the preview source start tracked by the live runner.
- `speech_end_ms` comes from the preview `audio_end_ms`.
- Empty preview-clear rows use a zero-length interval at the current source position, for
  example `p,1820,1820,`. This keeps the old preview-clear behavior without nullable timing.

For VAD/speech-gate forced preview-tail commits:

- The commit currently includes trailing silence.
- Export should estimate speech end by subtracting the effective force-commit silence
  threshold from the raw commit end.
- The estimate must be clamped so it never becomes earlier than `speech_start_ms`.

Effective force threshold:

```text
max(silence_enter_ms, force_commit_silence_ms)
```

In the current config this is:

```text
max(900, 2500) = 2500ms
```

## Commit Reason

The forced VAD/speech-gate tail commit must get its own unique reason:

```text
rolling_context_speech_gate_tail_commit
```

Do not infer this from a separate export-only flag. The commit reason should be the durable
semantic marker in Omniscripta commit rows and in realtime-asr-engine debug counters.

Existing tail commits that are not speech-gate forced keep:

```text
rolling_context_tail_preview_commit
```

Normal rolling commits keep:

```text
rolling_context_commit
```

## Required Code Changes

### Omniscripta

1. Change `.pc` export from old `kind,text` rows to the new CSV columns.
2. Keep the same export endpoint and button.
3. Store/export commit timing from the existing live commit rows.
4. Store/export preview timing by passing preview start and preview `audio_end_ms` into
   `update_live_preview()`.
5. When `_commit_preview_tail_if_needed(..., speech_gate_forced=True)` records a row, use
   `rolling_context_speech_gate_tail_commit`.
6. For that reason only, subtract the effective force threshold when writing
   `speech_end_ms`.
7. Update tests to expect the new `.pc` format.

### realtime-asr-engine

1. Let `commit_preview_tail(..., speech_gate_forced=True)` increment the commit-reason
   counter for `rolling_context_speech_gate_tail_commit`.
2. Keep non-forced tail commits counted as `rolling_context_tail_preview_commit`.
3. Update tests that assert commit reason counters.

### llm-workbench

1. Replace the old `kind,text` `.pc` parser.
2. Parse the new CSV columns only.
3. Remove old-format compatibility.
4. Update sample `.pc` files to the new format.
5. Pass commit timing through replay state so TTS can compare rendered duration with source
   speech duration.

## Minimal Example

Input `.pc`:

```csv
kind,speech_start_ms,speech_end_ms,text
c,0,1320,Good morning everyone.
c,1740,3860,Today we test the replay export.
c,4250,5480,That is all for now.
```

TTS target durations:

```text
Good morning everyone. -> 1320ms
Today we test the replay export. -> 2120ms
That is all for now. -> 1230ms
```

## Implementation Order

1. Update realtime-asr-engine commit reason handling.
2. Update Omniscripta commit row reason and `.pc` export.
3. Update Omniscripta tests.
4. Update llm-workbench parser and sample files.
5. Update llm-workbench tests.
6. Only after that, wire TTS metrics/speed logic to `speech_start_ms` and `speech_end_ms`.
