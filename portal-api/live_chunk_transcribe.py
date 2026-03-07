from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from queue_fs import BASE as JOBS_BASE, init_job_in_inbox


DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH_BYTES = 2


def _repo_root() -> Path:
    # portal-api/live_chunk_transcribe.py -> portal-api -> repo root
    return Path(__file__).resolve().parents[1]


def _safe_session_id(session_id: str) -> str:
    text = str(session_id or "").strip()
    if not text:
        return "unknown"
    return "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in text)


def _normalize_optional_language(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _find_job_dir(job_id: str, *, jobs_base: Path) -> Path | None:
    for state in ("inbox", "running", "done", "error"):
        d = jobs_base / state / str(job_id)
        if d.exists():
            return d
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_speaker_lines_parse_error(status: dict[str, Any], err: str) -> bool:
    text = str(err or status.get("error") or "").strip().lower()
    if not text:
        return False
    if "speaker_lines" not in text or "parseable" not in text:
        return False
    speaker_mode = str(status.get("speaker_mode") or "").strip().lower()
    subphase = str(status.get("subphase") or "").strip().lower()
    return speaker_mode in {"none", ""} or subphase == "chunk_speaker_lines"


def _pick_srt_result_path(job_dir: Path, status: dict[str, Any]) -> Path | None:
    srt_name = str(status.get("srt_filename") or "").strip()
    whisperx_dir = (job_dir / "whisperx").resolve()
    if srt_name:
        p = (whisperx_dir / srt_name).resolve()
        if p.exists():
            return p
    if not whisperx_dir.exists():
        return None
    candidates = sorted(p for p in whisperx_dir.glob("*.srt") if p.is_file())
    if not candidates:
        return None
    return candidates[-1]


def _srt_ts_to_ms(token: str) -> int:
    t = str(token or "").strip()
    if not t:
        return 0
    # "HH:MM:SS,mmm"
    parts = t.split(":")
    if len(parts) != 3:
        return 0
    h = int(parts[0] or 0)
    m = int(parts[1] or 0)
    sec_ms = parts[2].split(",")
    s = int(sec_ms[0] or 0)
    ms = int(sec_ms[1] or 0) if len(sec_ms) > 1 else 0
    return max(0, ((h * 3600 + m * 60 + s) * 1000) + ms)


def _parse_srt_segments(srt_text: str, *, t0_offset_ms: int = 0) -> list[dict[str, Any]]:
    text = str(srt_text or "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = [blk.strip() for blk in text.split("\n\n") if blk.strip()]
    out: list[dict[str, Any]] = []
    seg_index = 0
    for blk in blocks:
        lines = [ln for ln in blk.split("\n") if ln.strip()]
        if len(lines) < 2:
            continue
        time_line_idx = 1 if "-->" in lines[1] else 0
        if "-->" not in lines[time_line_idx]:
            continue
        a, b = [p.strip() for p in lines[time_line_idx].split("-->", 1)]
        t0_ms = _srt_ts_to_ms(a) + int(max(0, t0_offset_ms))
        t1_ms = _srt_ts_to_ms(b) + int(max(0, t0_offset_ms))
        txt_lines = lines[time_line_idx + 1 :]
        seg_text = " ".join(s.strip() for s in txt_lines if s.strip()).strip()
        if not seg_text:
            continue
        seg_index += 1
        out.append(
            {
                "segment_id": f"s{seg_index:04d}",
                "text": seg_text,
                "t0_ms": int(max(0, t0_ms)),
                "t1_ms": int(max(t0_ms, t1_ms)),
            }
        )
    return out


def _srt_to_plain_text(srt_text: str) -> str:
    segs = _parse_srt_segments(srt_text)
    return "\n".join(seg["text"] for seg in segs if str(seg.get("text") or "").strip())


@dataclass(frozen=True)
class EnqueuedChunkJob:
    session_id: str
    chunk_index: int
    job_id: str
    job_dir: str
    chunk_wav_path: str
    language: str | None
    initial_prompt_chars: int = 0
    initial_prompt_words: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": str(self.session_id),
            "chunk_index": int(self.chunk_index),
            "job_id": str(self.job_id),
            "job_dir": str(self.job_dir),
            "chunk_wav_path": str(self.chunk_wav_path),
            "language": self.language,
            "initial_prompt_chars": int(max(0, self.initial_prompt_chars)),
            "initial_prompt_words": int(max(0, self.initial_prompt_words)),
        }


@dataclass(frozen=True)
class ChunkJobPollResult:
    job_id: str
    state: str
    status: dict[str, Any]
    done: bool
    ok: bool
    error: str
    text: str
    srt_text: str
    segments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": str(self.job_id),
            "state": str(self.state),
            "done": bool(self.done),
            "ok": bool(self.ok),
            "error": str(self.error),
            "text": str(self.text),
            "segments": [dict(seg) for seg in self.segments],
            "status": dict(self.status),
        }


class LiveChunkBatchBridge:
    def __init__(
        self,
        *,
        jobs_base: Path | None = None,
        chunks_root: Path | None = None,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        channels: int = DEFAULT_CHANNELS,
        language: str | None = None,
    ) -> None:
        self.jobs_base = (jobs_base if jobs_base is not None else JOBS_BASE).resolve()
        self.chunks_root = (
            chunks_root if chunks_root is not None else (_repo_root() / "data" / "live_chunk_jobs")
        ).resolve()
        self.sample_rate_hz = int(max(1, sample_rate_hz))
        self.channels = int(max(1, channels))
        self.language = _normalize_optional_language(language)

    def enqueue_chunk_pcm16(
        self,
        *,
        session_id: str,
        chunk_index: int,
        t0_ms: int,
        t1_ms: int,
        pcm16le: bytes,
        language: str | None = None,
        asr_beam_size: int | None = None,
        initial_prompt: str | None = None,
        live_lane: str | None = None,
        preview_seq: int | None = None,
        preview_audio_end_ms: int | None = None,
    ) -> EnqueuedChunkJob:
        safe_id = _safe_session_id(session_id)
        idx = int(max(0, chunk_index))
        sess_dir = (self.chunks_root / safe_id).resolve()
        sess_dir.mkdir(parents=True, exist_ok=True)
        chunk_wav = (sess_dir / f"chunk_{idx:04d}_{int(max(0, t0_ms))}_{int(max(0, t1_ms))}.wav").resolve()

        raw = bytes(pcm16le or b"")
        if (len(raw) % DEFAULT_SAMPLE_WIDTH_BYTES) != 0:
            raw = raw[: len(raw) - 1]
        with wave.open(str(chunk_wav), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(DEFAULT_SAMPLE_WIDTH_BYTES)
            wf.setframerate(self.sample_rate_hz)
            wf.writeframes(raw)

        prompt_text = str(initial_prompt or "").strip()
        prompt_words = len([tok for tok in prompt_text.split() if tok])
        resolved_language = _normalize_optional_language(language if language is not None else self.language)
        job = init_job_in_inbox(
            orig_filename=chunk_wav.name,
            options={
                "language": resolved_language,
                "beam_size": (int(max(1, asr_beam_size)) if asr_beam_size is not None else None),
                "speaker_mode": "none",
                "expected_speakers": None,
                "min_speakers": None,
                "max_speakers": None,
                # Keep chunk jobs focused/fast; worker may ignore unknown keys.
                "live_chunk_mode": True,
                "live_session_id": str(session_id),
                "live_chunk_index": int(idx),
                "live_chunk_t0_ms": int(max(0, t0_ms)),
                "live_chunk_t1_ms": int(max(max(0, t0_ms), t1_ms)),
                "initial_prompt": prompt_text if prompt_text else None,
                "live_lane": str(live_lane or "final"),
                "preview_seq": (
                    int(max(0, preview_seq)) if preview_seq is not None else None
                ),
                "preview_audio_end_ms": (
                    int(max(0, preview_audio_end_ms)) if preview_audio_end_ms is not None else None
                ),
            },
            job_kind="live_chunk",
            upload_src_path=chunk_wav,
            move_upload_src=True,
        )
        chunk_upload_path = (job.upload_dir / chunk_wav.name).resolve()
        return EnqueuedChunkJob(
            session_id=str(session_id),
            chunk_index=idx,
            job_id=str(job.job_id),
            job_dir=str(job.dir),
            chunk_wav_path=str(chunk_upload_path),
            language=resolved_language,
            initial_prompt_chars=len(prompt_text),
            initial_prompt_words=prompt_words,
        )

    def poll_job(self, job_id: str, *, t0_offset_ms: int = 0) -> ChunkJobPollResult:
        job_dir = _find_job_dir(str(job_id), jobs_base=self.jobs_base)
        if job_dir is None:
            raise FileNotFoundError(f"Job not found: {job_id}")

        status_path = (job_dir / "status.json").resolve()
        if not status_path.exists():
            raise FileNotFoundError(f"status.json missing for job: {job_id}")
        status = _read_json(status_path)
        state = str(status.get("state") or "")
        done = state in {"done", "error"}
        ok = state == "done"
        err = str(status.get("error") or "")
        srt_text = ""
        plain = ""
        segments: list[dict[str, Any]] = []

        # Primary result path: status == done and points to SRT.
        # Fallback path: some chunk jobs fail late on speaker_lines parsing while an SRT already exists.
        srt_path: Path | None = None
        if ok or (done and _is_speaker_lines_parse_error(status, err)):
            srt_path = _pick_srt_result_path(job_dir, status)
        if srt_path is not None and srt_path.exists():
            srt_text = srt_path.read_text(encoding="utf-8")
            segments = _parse_srt_segments(srt_text, t0_offset_ms=int(max(0, t0_offset_ms)))
            plain = "\n".join(
                str(seg.get("text") or "").strip() for seg in segments if str(seg.get("text") or "").strip()
            )
            if not plain:
                plain = _srt_to_plain_text(srt_text)
            if (not ok) and done and _is_speaker_lines_parse_error(status, err) and (plain.strip() or segments):
                ok = True
                state = "done_fallback_srt"
                err = ""

        # Some short/empty chunk jobs fail late on speaker_lines generation while transcript output is empty.
        # For live chunk mode this should not poison the whole session result.
        if (not ok) and done and _is_speaker_lines_parse_error(status, err):
            transcript_end = str(status.get("transcript_end") or "").strip()
            if transcript_end in {"", "00:00:00", "0:00:00"} and not plain.strip() and not segments:
                ok = True
                state = "done_empty_chunk"
                err = ""

        return ChunkJobPollResult(
            job_id=str(job_id),
            state=state,
            status=status,
            done=bool(done),
            ok=bool(ok),
            error=err,
            text=plain,
            srt_text=srt_text,
            segments=segments,
        )
