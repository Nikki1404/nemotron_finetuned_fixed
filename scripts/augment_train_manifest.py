#!/usr/bin/env python3
"""Create conservative train-only audio augmentation.

Only the training split is augmented. Validation and test remain clean.
The safe profile avoids aggressive volume boosts and excessive duplicates.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import wave
from pathlib import Path


def duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value.lower()).strip("_") or "unknown"


def augmentation_configs(profile: str):
    safe = [
        ("speed097", ["-filter:a", "atempo=0.97"]),
        ("speed103", ["-filter:a", "atempo=1.03"]),
        (
            "tel8k",
            [
                "-ar", "8000", "-ac", "1",
                "-af", "highpass=f=300,lowpass=f=3400",
            ],
        ),
        (
            "soft_noise",
            [
                "-filter_complex",
                "[0:a]volume=0.98[a];"
                "anoisesrc=color=pink:amplitude=0.0012[n];"
                "[a][n]amix=inputs=2:duration=first:dropout_transition=0",
            ],
        ),
    ]
    if profile == "safe":
        return safe
    return safe + [
        ("volm3", ["-filter:a", "volume=-3dB"]),
        ("speed094", ["-filter:a", "atempo=0.94"]),
        ("speed106", ["-filter:a", "atempo=1.06"]),
    ]


def augment_audio(src: Path, out_dir: Path, profile: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, options in augmentation_configs(profile):
        out = out_dir / f"{src.stem}_{name}.wav"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
        cmd += options
        cmd += ["-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(out)]
        try:
            run(cmd)
            outputs.append((name, out.resolve()))
        except subprocess.CalledProcessError:
            print(f"[warn] augmentation failed: {name} {src}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-audio-dir", default="data/audio_aug")
    parser.add_argument("--keep-original", action="store_true")
    parser.add_argument("--profile", choices=["safe", "extended"], default="safe")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.train_manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final = []
    for index, row in enumerate(rows, 1):
        src = Path(row["audio_filepath"])
        if not src.exists():
            print(f"[skip missing] {src}")
            continue
        if args.keep_original:
            original = dict(row)
            original["augmentation"] = "original"
            final.append(original)

        out_dir = Path(args.out_audio_dir) / safe_name(row.get("use_case", "unknown"))
        print(f"[{index}/{len(rows)}] augmenting {src}")
        for aug_name, aug_path in augment_audio(src, out_dir, args.profile):
            new = dict(row)
            new["audio_filepath"] = str(aug_path)
            new["duration"] = round(duration_sec(aug_path), 3)
            new["augmentation"] = aug_name
            final.append(new)

    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in final) + "\n",
        encoding="utf-8",
    )
    print("Original rows:", len(rows))
    print("Final rows:", len(final))
    print("Saved:", out)


if __name__ == "__main__":
    main()
