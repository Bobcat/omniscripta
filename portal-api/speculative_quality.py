from __future__ import annotations

from statistics import mean, median
from typing import Any

from live_quality import normalize_transcript_text, _word_levenshtein


def _tokenize_words(text: str) -> list[str]:
    normalized = normalize_transcript_text(text)
    return normalized.split(" ") if normalized else []


def _word_similarity_metrics(pred_text: str, target_text: str) -> dict[str, Any]:
    pred_raw = str(pred_text or "")
    target_raw = str(target_text or "")
    pred_words = _tokenize_words(pred_raw)
    target_words = _tokenize_words(target_raw)

    out: dict[str, Any] = {
        "pred_text_chars": len(pred_raw),
        "target_text_chars": len(target_raw),
        "pred_word_count": int(len(pred_words)),
        "target_word_count": int(len(target_words)),
        "word_edit_distance": None,
        "word_similarity": None,
        "exact_match": None,
    }
    if not target_words:
        return out

    dist = int(_word_levenshtein(pred_words, target_words))
    denom = max(1, len(target_words))
    similarity = max(0.0, 1.0 - (float(dist) / float(denom)))
    out["word_edit_distance"] = int(dist)
    out["word_similarity"] = round(similarity, 4)
    out["exact_match"] = bool(dist == 0)
    return out


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return round(xs[0], 4)
    p = max(0.0, min(100.0, float(p)))
    pos = (p / 100.0) * float(len(xs) - 1)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    if lo == hi:
        return round(xs[lo], 4)
    frac = pos - float(lo)
    val = xs[lo] + (xs[hi] - xs[lo]) * frac
    return round(float(val), 4)


def _summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    xs = [float(v) for v in values]
    return {
        "count": int(len(xs)),
        "mean": round(float(mean(xs)), 4),
        "median": round(float(median(xs)), 4),
        "p10": _percentile(xs, 10),
        "p50": _percentile(xs, 50),
        "p90": _percentile(xs, 90),
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
    }


def _item_variant_texts(item: dict[str, Any]) -> dict[str, str]:
    return {
        "raw": str(item.get("raw_text") or ""),
        "merged": str(item.get("merged_text_after_seam_dedup") or ""),
        "suffix": str(item.get("suffix_text_after_final_dedup") or ""),
    }


def _score_window_items_against_target(
    items: list[dict[str, Any]],
    *,
    target_text: str,
    verbose: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored_items: list[dict[str, Any]] = []
    target = str(target_text or "")
    for item in items:
        if not isinstance(item, dict):
            continue
        variants = _item_variant_texts(item)
        row = {
            "speculative_seq": int(max(-1, int(item.get("speculative_seq") or -1))),
            "audio_end_ms": int(max(0, int(item.get("audio_end_ms") or 0))),
            "variants": {
                "raw": _word_similarity_metrics(variants["raw"], target),
                "merged": _word_similarity_metrics(variants["merged"], target),
                "suffix": _word_similarity_metrics(variants["suffix"], target),
            },
        }
        if verbose:
            row["texts"] = variants
        scored_items.append(row)

    def _selector(variant_name: str) -> dict[str, Any]:
        vals: list[tuple[int, float, int]] = []
        for idx, row in enumerate(scored_items):
            sim = ((row.get("variants") or {}).get(variant_name) or {}).get("word_similarity")
            if sim is None:
                continue
            vals.append((idx, float(sim), int(row.get("speculative_seq") or -1)))
        if not vals:
            return {
                "last": None,
                "best": None,
                "best_item_index": None,
                "best_speculative_seq": None,
                "last_item_index": None,
                "last_speculative_seq": None,
            }
        last_idx = vals[-1][0]
        best_idx, best_val, best_seq = max(vals, key=lambda t: (t[1], t[0]))
        last_row = scored_items[last_idx]
        last_val = ((last_row.get("variants") or {}).get(variant_name) or {}).get("word_similarity")
        return {
            "last": round(float(last_val), 4) if last_val is not None else None,
            "best": round(float(best_val), 4),
            "best_item_index": int(best_idx),
            "best_speculative_seq": int(best_seq),
            "last_item_index": int(last_idx),
            "last_speculative_seq": int(last_row.get("speculative_seq") or -1),
            "best_equals_last": bool(best_idx == last_idx),
        }

    summary = {
        "raw": _selector("raw"),
        "merged": _selector("merged"),
        "suffix": _selector("suffix"),
    }
    return scored_items, summary


def _words_to_text(words: list[str]) -> str:
    return " ".join([str(w).strip() for w in words if str(w).strip()]).strip()


def _find_best_reference_span_for_chunk(
    *,
    reference_words: list[str],
    chunk_words: list[str],
    cursor_word_index: int,
    search_back_words: int = 8,
    search_ahead_words: int = 80,
    length_pad_words: int = 6,
    min_span_words: int = 3,
) -> dict[str, Any] | None:
    if not reference_words or not chunk_words:
        return None

    ref_len = len(reference_words)
    chunk_len = len(chunk_words)
    if ref_len <= 0 or chunk_len <= 0:
        return None

    start_min = max(0, int(cursor_word_index) - int(max(0, search_back_words)))
    start_max = min(ref_len - 1, int(cursor_word_index) + int(max(0, search_ahead_words)))
    if start_max < start_min:
        start_min, start_max = 0, ref_len - 1

    len_min = max(int(max(1, min_span_words)), chunk_len - int(max(0, length_pad_words)))
    len_max = chunk_len + int(max(0, length_pad_words))

    best: dict[str, Any] | None = None
    for start in range(start_min, start_max + 1):
        max_len_here = min(len_max, ref_len - start)
        if max_len_here < len_min:
            continue
        for span_len in range(len_min, max_len_here + 1):
            span_words = reference_words[start : start + span_len]
            if not span_words:
                continue
            dist = int(_word_levenshtein(chunk_words, span_words))
            denom = max(1, len(span_words))
            sim = max(0.0, 1.0 - (float(dist) / float(denom)))
            row = {
                "start_word_index": int(start),
                "end_word_index": int(start + span_len),
                "span_word_count": int(span_len),
                "alignment_word_similarity": round(float(sim), 4),
                "alignment_word_edit_distance": int(max(0, dist)),
                "text": _words_to_text(span_words),
                "cursor_distance_words": int(abs(start - int(cursor_word_index))),
            }
            if best is None:
                best = row
                continue
            best_sim = float(best.get("alignment_word_similarity") or 0.0)
            if sim > best_sim:
                best = row
                continue
            if sim == best_sim:
                if int(row["cursor_distance_words"]) < int(best.get("cursor_distance_words") or 10**9):
                    best = row
                    continue
                if int(row["cursor_distance_words"]) == int(best.get("cursor_distance_words") or 10**9):
                    if int(row["span_word_count"]) == int(chunk_len):
                        best = row
    return best


def score_speculative_history_against_final(
    *,
    speculative_history: dict[str, Any],
    verbose: bool = False,
) -> dict[str, Any]:
    history = speculative_history if isinstance(speculative_history, dict) else {}
    windows_src = history.get("speculative_windows")
    windows = [dict(w) for w in windows_src] if isinstance(windows_src, list) else []
    open_window_src = history.get("speculative_open_window")
    open_window = [dict(x) for x in open_window_src] if isinstance(open_window_src, list) else []

    window_rows: list[dict[str, Any]] = []
    suffix_last_scores: list[float] = []
    suffix_best_scores: list[float] = []
    merged_last_scores: list[float] = []
    merged_best_scores: list[float] = []
    raw_last_scores: list[float] = []
    raw_best_scores: list[float] = []
    windows_scored = 0
    windows_missing_target = 0

    for w in windows:
        items_src = w.get("items")
        items = [dict(item) for item in items_src] if isinstance(items_src, list) else []
        target_raw = str(w.get("target_final_chunk_text_raw") or "")
        target_effective = str(w.get("target_final_chunk_text_effective") or "")
        target_primary = target_effective or target_raw

        base_row: dict[str, Any] = {
            "window_index": int(max(0, int(w.get("window_index") or 0))),
            "close_reason": str(w.get("close_reason") or ""),
            "items_count": int(len(items)),
            "started_at_revision": int(max(0, int(w.get("started_at_revision") or 0))),
            "ended_by_final_revision": (
                int(max(0, int(w.get("ended_by_final_revision") or 0)))
                if w.get("ended_by_final_revision") is not None
                else None
            ),
            "last_speculative_seq": int(max(-1, int(w.get("last_speculative_seq") or -1))),
            "last_audio_end_ms": int(max(0, int(w.get("last_audio_end_ms") or 0))),
            "target": {
                "primary_text": target_primary,
                "raw_text": target_raw,
                "effective_text": target_effective,
                "primary_word_count": len(_tokenize_words(target_primary)),
            },
        }

        if not target_primary.strip():
            windows_missing_target += 1
            base_row["scored"] = False
            base_row["reason"] = "missing_target_final_chunk_text"
            if verbose:
                base_row["items"] = items
            window_rows.append(base_row)
            continue

        scored_items, score_summary = _score_window_items_against_target(items, target_text=target_primary, verbose=verbose)
        base_row["scored"] = True
        base_row["scores"] = score_summary
        if verbose:
            base_row["items"] = scored_items
        window_rows.append(base_row)
        windows_scored += 1

        def _push(selector: dict[str, Any], dst_last: list[float], dst_best: list[float]) -> None:
            if not isinstance(selector, dict):
                return
            if selector.get("last") is not None:
                dst_last.append(float(selector["last"]))
            if selector.get("best") is not None:
                dst_best.append(float(selector["best"]))

        _push(score_summary.get("suffix") or {}, suffix_last_scores, suffix_best_scores)
        _push(score_summary.get("merged") or {}, merged_last_scores, merged_best_scores)
        _push(score_summary.get("raw") or {}, raw_last_scores, raw_best_scores)

    open_window_preview = {
        "items_count": int(len(open_window)),
        "window_index": int(max(0, int(history.get("speculative_open_window_index") or 0))),
        "started_at_revision": int(max(0, int(history.get("speculative_open_window_started_revision") or 0))),
        "last_speculative_seq": (
            int(max(-1, int(open_window[-1].get("speculative_seq") or -1))) if open_window else None
        ),
        "last_audio_end_ms": (
            int(max(0, int(open_window[-1].get("audio_end_ms") or 0))) if open_window else None
        ),
    }
    if verbose and open_window:
        open_window_preview["items"] = [dict(item) for item in open_window]

    return {
        "metric_version": "speculative_quality_v1",
        "history_source": str(history.get("source") or ""),
        "session_id": str(history.get("session_id") or ""),
        "finalization_state": str(history.get("finalization_state") or ""),
        "transcript_revision": int(max(0, int(history.get("transcript_revision") or 0))),
        "summary": {
            "windows_total": int(len(windows)),
            "windows_scored": int(windows_scored),
            "windows_missing_target": int(windows_missing_target),
            "open_window_items_count": int(len(open_window)),
            "suffix_last_word_similarity": _summary_stats(suffix_last_scores),
            "suffix_best_word_similarity": _summary_stats(suffix_best_scores),
            "merged_last_word_similarity": _summary_stats(merged_last_scores),
            "merged_best_word_similarity": _summary_stats(merged_best_scores),
            "raw_last_word_similarity": _summary_stats(raw_last_scores),
            "raw_best_word_similarity": _summary_stats(raw_best_scores),
        },
        "windows": window_rows,
        "open_window": open_window_preview,
    }


def score_speculative_history_against_reference(
    *,
    speculative_history: dict[str, Any],
    reference_text: str,
    verbose: bool = False,
) -> dict[str, Any]:
    history = speculative_history if isinstance(speculative_history, dict) else {}
    windows_src = history.get("speculative_windows")
    windows = [dict(w) for w in windows_src] if isinstance(windows_src, list) else []

    ref_words = _tokenize_words(str(reference_text or ""))
    cursor = 0
    windows_aligned = 0
    windows_unaligned = 0
    alignment_sims: list[float] = []

    suffix_last_scores: list[float] = []
    suffix_best_scores: list[float] = []
    merged_last_scores: list[float] = []
    merged_best_scores: list[float] = []
    raw_last_scores: list[float] = []
    raw_best_scores: list[float] = []
    window_rows: list[dict[str, Any]] = []

    for w in windows:
        items_src = w.get("items")
        items = [dict(item) for item in items_src] if isinstance(items_src, list) else []
        target_raw = str(w.get("target_final_chunk_text_raw") or "")
        target_effective = str(w.get("target_final_chunk_text_effective") or "")
        chunk_text = target_effective or target_raw
        chunk_words = _tokenize_words(chunk_text)

        base_row: dict[str, Any] = {
            "window_index": int(max(0, int(w.get("window_index") or 0))),
            "items_count": int(len(items)),
            "chunk_target_word_count": int(len(chunk_words)),
            "aligned": False,
        }
        if not chunk_words or not ref_words:
            windows_unaligned += 1
            base_row["reason"] = "missing_chunk_or_reference_words"
            if verbose:
                base_row["items"] = items
            window_rows.append(base_row)
            continue

        span = _find_best_reference_span_for_chunk(
            reference_words=ref_words,
            chunk_words=chunk_words,
            cursor_word_index=int(max(0, cursor)),
        )
        if not isinstance(span, dict) or not str(span.get("text") or "").strip():
            windows_unaligned += 1
            base_row["reason"] = "alignment_not_found"
            if verbose:
                base_row["items"] = items
            window_rows.append(base_row)
            continue

        windows_aligned += 1
        base_row["aligned"] = True
        base_row["reference_target"] = {
            "start_word_index": int(max(0, int(span.get("start_word_index") or 0))),
            "end_word_index": int(max(0, int(span.get("end_word_index") or 0))),
            "span_word_count": int(max(0, int(span.get("span_word_count") or 0))),
            "alignment_word_similarity": float(span.get("alignment_word_similarity") or 0.0),
            "alignment_word_edit_distance": int(max(0, int(span.get("alignment_word_edit_distance") or 0))),
            "text": str(span.get("text") or ""),
        }
        cursor = int(max(cursor, int(span.get("end_word_index") or cursor)))
        alignment_sims.append(float(span.get("alignment_word_similarity") or 0.0))

        scored_items, score_summary = _score_window_items_against_target(
            items,
            target_text=str(span.get("text") or ""),
            verbose=verbose,
        )
        base_row["scores"] = score_summary
        if verbose:
            base_row["items"] = scored_items
        window_rows.append(base_row)

        def _push(selector: dict[str, Any], dst_last: list[float], dst_best: list[float]) -> None:
            if not isinstance(selector, dict):
                return
            if selector.get("last") is not None:
                dst_last.append(float(selector["last"]))
            if selector.get("best") is not None:
                dst_best.append(float(selector["best"]))

        _push(score_summary.get("suffix") or {}, suffix_last_scores, suffix_best_scores)
        _push(score_summary.get("merged") or {}, merged_last_scores, merged_best_scores)
        _push(score_summary.get("raw") or {}, raw_last_scores, raw_best_scores)

    return {
        "metric_version": "speculative_quality_vs_reference_v1",
        "history_source": str(history.get("source") or ""),
        "session_id": str(history.get("session_id") or ""),
        "finalization_state": str(history.get("finalization_state") or ""),
        "transcript_revision": int(max(0, int(history.get("transcript_revision") or 0))),
        "summary": {
            "windows_total": int(len(windows)),
            "windows_aligned": int(windows_aligned),
            "windows_unaligned": int(windows_unaligned),
            "alignment_word_similarity": _summary_stats(alignment_sims),
            "suffix_last_word_similarity": _summary_stats(suffix_last_scores),
            "suffix_best_word_similarity": _summary_stats(suffix_best_scores),
            "merged_last_word_similarity": _summary_stats(merged_last_scores),
            "merged_best_word_similarity": _summary_stats(merged_best_scores),
            "raw_last_word_similarity": _summary_stats(raw_last_scores),
            "raw_best_word_similarity": _summary_stats(raw_best_scores),
        },
        "windows": window_rows,
    }
