#!/usr/bin/env python3
"""Create monotonic, quality-filtered ASR chunk manifests.

Why this version is safer than the previous script:
- alignment can never move backward into words assigned to an earlier chunk;
- leaked prompt tags are removed before matching;
- numeric literals in the reference are replaced by the number words actually
  heard by the base ASR, so audio saying "twenty forty three" is not trained
  against the incompatible label "2043";
- low-score, empty and abnormal word-rate chunks are excluded from training;
- all decisions are written to an audit CSV for review.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

import torch
import nemo.collections.asr as nemo_asr

from app.asr_number_normalizer import NUMBER_WORDS, parse_number_phrase
from app.transcript_postprocessor import DomainEntityCorrector

LANG_TAG_RE = re.compile(r"<[a-z]{2}-[A-Z]{2}>", re.IGNORECASE)


def norm_name(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def clean_text(text: str) -> str:
    text = LANG_TAG_RE.sub(" ", text or "")
    text = text.replace("’", "'")
    text = re.sub(r"[^a-zA-Z0-9'ñáéíóúü\s]", " ", text)
    text = text.lower()
    # Some prompt-conditioned outputs leak a plain "en us" after tag cleanup.
    text = re.sub(r"\ben\s+us\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def duration_sec(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def patch_nemotron_prompt(default_lang: str) -> None:
    import inspect
    import nemo.collections.asr.data.audio_to_text_lhotse_prompt_index as mod

    for _, cls in vars(mod).items():
        if inspect.isclass(cls) and hasattr(cls, "_get_prompt_index"):
            old_fn = cls._get_prompt_index

            def new_get_prompt_index(self, prompt_key, _old_fn=old_fn):
                if prompt_key is None or str(prompt_key).lower() == "none":
                    prompt_key = default_lang
                return _old_fn(self, prompt_key)

            cls._get_prompt_index = new_get_prompt_index


def extract_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return extract_text(value[0])
    if hasattr(value, "text"):
        return value.text
    if isinstance(value, dict):
        return value.get("text") or str(value)
    return str(value)


def split_audio(wav_path: Path, out_dir: Path, chunk_sec: float, force: bool) -> list[Path]:
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.wav"))
    if existing:
        return existing

    pattern = str(out_dir / f"{wav_path.stem}_%03d.wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(wav_path),
        "-f", "segment",
        "-segment_time", str(chunk_sec),
        "-reset_timestamps", "1",
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("*.wav"))


def transcribe_chunk(model, wav_path: Path, language: str) -> str:
    try:
        model.set_inference_prompt(language)
    except Exception:
        pass
    with torch.no_grad():
        output = model.transcribe([str(wav_path)], batch_size=1, verbose=False)
    return clean_text(extract_text(output))


def best_monotonic_window(
    full_words: list[str],
    draft_words: list[str],
    cursor: int,
) -> tuple[int, int, float]:
    """Find the best reference window, never starting before *cursor*."""
    if not draft_words or cursor >= len(full_words):
        return cursor, cursor, 0.0

    expected_len = len(draft_words)
    min_len = max(2, int(expected_len * 0.60))
    max_len = min(len(full_words) - cursor, int(expected_len * 1.45) + 6)
    search_end = min(len(full_words), cursor + expected_len + 45)

    best_score = -1.0
    best_start = cursor
    best_end = min(cursor + expected_len, len(full_words))
    draft_text = " ".join(draft_words)

    for start in range(cursor, search_end):
        for length in range(min_len, max_len + 1):
            end = start + length
            if end > len(full_words):
                break
            candidate = " ".join(full_words[start:end])
            score = SequenceMatcher(None, draft_text, candidate).ratio()
            # Slightly prefer earlier windows for equal scores.
            score -= (start - cursor) * 0.0002
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    return best_start, best_end, max(0.0, best_score)


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            use_case = row.get("use_case") or row.get("Use Case") or row.get("UseCase")
            transcript = row.get("transcript") or row.get("Transcript")
            if use_case and transcript:
                rows.append({"use_case": use_case.strip(), "transcript": transcript.strip()})
    if not rows:
        raise ValueError(f"No usable rows in {csv_path}")
    return rows


def find_wav_for_use_case(use_case: str, wav_dir: Path) -> Path:
    target = norm_name(use_case)
    wavs = list(wav_dir.glob("*.wav")) + list(wav_dir.glob("*.WAV"))
    for wav in wavs:
        if norm_name(wav.stem) == target:
            return wav
    for wav in wavs:
        stem = norm_name(wav.stem)
        if target in stem or stem in target:
            return wav
    aliases = {
        "account_not_found_bank_issue": "bank_issue",
        "cobra_coverage_faq": "cobra_coverage",
    }
    alias = aliases.get(target)
    if alias:
        for wav in wavs:
            if norm_name(wav.stem) == alias:
                return wav
    raise FileNotFoundError(f"No WAV found for use case: {use_case}")


def _numeric_candidates(tokens: list[str], start_cursor: int = 0) -> Iterable[tuple[int, int, str]]:
    max_window = 7
    for start in range(start_cursor, len(tokens)):
        if tokens[start] not in NUMBER_WORDS and not tokens[start].isdigit():
            continue
        for end in range(start + 1, min(len(tokens), start + max_window) + 1):
            span = tokens[start:end]
            if any(t not in NUMBER_WORDS and not t.isdigit() for t in span):
                break
            parsed = parse_number_phrase(span)
            if parsed:
                yield start, end, parsed


def _same_number(candidate: str, target: str) -> bool:
    if candidate == target:
        return True
    try:
        return int(candidate) == int(target)
    except ValueError:
        return False


def spokenize_reference_numbers(reference: str, draft: str) -> tuple[str, int]:
    """Use draft number words when they represent a reference digit literal."""
    ref_tokens = reference.split()
    draft_tokens = draft.split()
    output: list[str] = []
    unmatched = 0
    draft_cursor = 0
    i = 0

    while i < len(ref_tokens):
        if not ref_tokens[i].isdigit():
            output.append(ref_tokens[i])
            i += 1
            continue

        start = i
        while i < len(ref_tokens) and ref_tokens[i].isdigit():
            i += 1
        run = ref_tokens[start:i]
        target = "".join(run) if all(len(x) == 1 for x in run) and len(run) > 1 else run[0]

        found: Optional[tuple[int, int]] = None
        for cand_start, cand_end, parsed in _numeric_candidates(draft_tokens, draft_cursor):
            if _same_number(parsed, target):
                found = (cand_start, cand_end)
                break

        if found:
            cand_start, cand_end = found
            output.extend(draft_tokens[cand_start:cand_end])
            draft_cursor = cand_end
        else:
            output.extend(run)
            unmatched += 1

    return " ".join(output), unmatched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--wav-dir", default="data/audio_16k")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out-dir", default="data/audio_chunks")
    parser.add_argument("--manifest", default="data/manifests/aligned_chunk_manifest.json")
    parser.add_argument("--audit", default="data/manifests/alignment_audit.csv")
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--chunk-sec", type=float, default=8.0)
    parser.add_argument("--min-score", type=float, default=0.78)
    parser.add_argument("--min-word-rate", type=float, default=0.6)
    parser.add_argument("--max-word-rate", type=float, default=5.5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    patch_nemotron_prompt(args.language)
    print("[load]", args.base_model)
    model = nemo_asr.models.ASRModel.restore_from(args.base_model, map_location="cuda")
    model = model.cuda().eval()
    entity_corrector = DomainEntityCorrector()

    manifest_rows: list[dict] = []
    audit_rows: list[dict] = []

    for item in read_csv_rows(Path(args.csv)):
        use_case = item["use_case"]
        full_text = clean_text(item["transcript"])
        full_words = full_text.split()
        wav = find_wav_for_use_case(use_case, Path(args.wav_dir))
        chunk_dir = Path(args.out_dir) / norm_name(use_case)
        chunks = split_audio(wav, chunk_dir, args.chunk_sec, args.force)

        print(f"\n[use_case] {use_case}\n[wav] {wav}\n[chunks] {len(chunks)}")
        cursor = 0

        for index, chunk in enumerate(chunks):
            duration = duration_sec(chunk)
            if duration < 1.0:
                continue

            draft = transcribe_chunk(model, chunk, args.language)
            draft, _ = entity_corrector.correct(draft)
            draft = clean_text(draft)
            draft_words = draft.split()
            start, end, score = best_monotonic_window(full_words, draft_words, cursor)
            aligned = " ".join(full_words[start:end])
            cursor = max(cursor, end)

            # Ticket IDs are best kept in the exact spoken/base-ASR form; for
            # ordinary chunks, use the human reference while preserving the
            # spoken number style found in the draft.
            if re.search(r"\bt\s+k\s+t\b", aligned):
                target = draft
                unmatched_numbers = 0
                target_source = "corrected_draft_ticket"
            else:
                target, unmatched_numbers = spokenize_reference_numbers(aligned, draft)
                target, _ = entity_corrector.correct(target)
                target = clean_text(target)
                target_source = "aligned_reference_spoken_numbers"

            word_rate = len(target.split()) / max(duration, 0.001)
            reasons: list[str] = []
            if not draft:
                reasons.append("empty_draft")
            if not target:
                reasons.append("empty_target")
            if score < args.min_score:
                reasons.append("low_alignment_score")
            if not (args.min_word_rate <= word_rate <= args.max_word_rate):
                reasons.append("abnormal_word_rate")
            if unmatched_numbers:
                reasons.append("unmatched_reference_number")

            accepted = not reasons
            if accepted:
                manifest_rows.append(
                    {
                        "audio_filepath": str(chunk.resolve()),
                        "duration": round(duration, 3),
                        "text": target,
                        "target_lang": args.language,
                        "language": args.language,
                        "use_case": use_case,
                    }
                )

            audit_rows.append(
                {
                    "use_case": use_case,
                    "chunk": str(chunk),
                    "duration": round(duration, 3),
                    "match_score": round(score, 4),
                    "word_rate": round(word_rate, 3),
                    "accepted": accepted,
                    "rejection_reasons": ";".join(reasons),
                    "target_source": target_source,
                    "unmatched_numbers": unmatched_numbers,
                    "draft_asr": draft,
                    "aligned_reference": aligned,
                    "training_text": target,
                }
            )
            status = "KEEP" if accepted else "DROP:" + ",".join(reasons)
            print(f"  {index + 1}/{len(chunks)} score={score:.2f} {status} | {target[:90]}")

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(audit_rows)

    kept = len(manifest_rows)
    total = len(audit_rows)
    print("\n[done]")
    print("manifest:", manifest_path)
    print("audit:", audit_path)
    print(f"kept: {kept}/{total}")
    if kept < 20:
        raise RuntimeError(
            "Too few safe chunks were produced. Review alignment_audit.csv and add or correct audio/transcripts before fine-tuning."
        )


if __name__ == "__main__":
    main()
