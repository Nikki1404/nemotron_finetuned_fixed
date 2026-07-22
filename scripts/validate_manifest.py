#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--allow-digits", action="store_true")
    parser.add_argument("--require-files", action="store_true")
    args = parser.parse_args()

    path = Path(args.manifest)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    warnings: list[str] = []
    seen_audio: set[str] = set()

    for index, row in enumerate(rows, 1):
        audio = Path(row.get("audio_filepath", ""))
        text = str(row.get("text", "")).strip()
        duration = float(row.get("duration", 0.0) or 0.0)
        target_lang = row.get("target_lang")

        if not text:
            errors.append(f"row {index}: empty text")
        if not target_lang:
            errors.append(f"row {index}: missing target_lang")
        if duration <= 0:
            errors.append(f"row {index}: invalid duration {duration}")
        if duration > args.max_duration:
            errors.append(f"row {index}: duration {duration:.2f}s exceeds {args.max_duration}s")
        if not args.allow_digits and re.search(r"\b\d+\b", text):
            errors.append(
                f"row {index}: numeric display token found in training text; use spoken words instead: {text[:120]}"
            )
        if str(audio) in seen_audio:
            warnings.append(f"row {index}: duplicate audio path {audio}")
        seen_audio.add(str(audio))
        if args.require_files and not audio.exists():
            errors.append(f"row {index}: missing audio file {audio}")
        elif audio.exists():
            try:
                actual = audio_duration(audio)
                if abs(actual - duration) > 0.25:
                    warnings.append(
                        f"row {index}: manifest duration {duration:.3f}s differs from WAV {actual:.3f}s"
                    )
            except Exception as exc:
                errors.append(f"row {index}: cannot read WAV {audio}: {exc}")

        word_rate = len(text.split()) / max(duration, 0.001)
        if word_rate < 0.5 or word_rate > 6.0:
            errors.append(f"row {index}: abnormal word rate {word_rate:.2f} words/s")

    print(f"Manifest: {path}")
    print(f"Rows: {len(rows)}")
    for item in warnings:
        print("[warning]", item)
    for item in errors:
        print("[error]", item)
    if errors:
        print(f"FAILED: {len(errors)} errors, {len(warnings)} warnings")
        sys.exit(1)
    print(f"PASSED: 0 errors, {len(warnings)} warnings")


if __name__ == "__main__":
    main()
