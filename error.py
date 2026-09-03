docker rm -f nemotron-trainer 2>/dev/null || true; docker run --rm --name nemotron-trainer --gpus all --ipc=host -e PYTHONPATH=/workspace -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TOKENIZERS_PARALLELISM=false -v "$PWD:/workspace" -v "$PWD/ft_models:/srv/models" -w /workspace nemotron_finetuned_3.5 bash -lc 'set -o pipefail; bash scripts/run_safe_finetuning.sh 2>&1 | tee logs/safe_finetuning.log'
sed -i '/^cd \/workspace$/a export PYTHONPATH=/workspace:${PYTHONPATH:-}' scripts/run_safe_finetuning.sh
head -n 10 scripts/run_safe_finetuning.sh
docker run --rm --gpus all -e PYTHONPATH=/workspace -v "$PWD:/workspace" -w /workspace nemotron_finetuned_3.5 python3.11 -c 'from app.asr_number_normalizer import NUMBER_WORDS,parse_number_phrase; from app.transcript_postprocessor import DomainEntityCorrector; print("App imports successful")'
docker run --rm -e PYTHONPATH=/workspace -v "$PWD:/workspace" -w /workspace nemotron_finetuned_3.5 bash -lc 'pwd; ls -la app; ls -la scripts; test -f app/__init__.py && echo "app package exists"'
mkdir -p ft_models logs results/safe_training data/manifests data/audio_16k data/audio_chunks data/audio_aug
docker rm -f nemotron-trainer 2>/dev/null || true; docker run --rm --name nemotron-trainer --gpus all --ipc=host -e PYTHONPATH=/workspace -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TOKENIZERS_PARALLELISM=false -v "$PWD:/workspace" -v "$PWD/ft_models:/srv/models" -w /workspace nemotron_finetuned_3.5 bash -lc 'set -o pipefail; bash scripts/run_safe_finetuning.sh 2>&1 | tee logs/safe_finetuning.log'
tail -f logs/safe_finetuning.log

python3 -c 'from pathlib import Path; p=Path("scripts/finetune_nemotron.py"); s=p.read_text(); s=s.replace("filename=\"best-{epoch:02d}-{val_loss:.4f}\"","filename=\"best-{epoch:02d}-{val_wer:.4f}\""); s=s.replace("monitor=\"val_loss\"","monitor=\"val_wer\""); s=s.replace("\"best_val_loss\": float(checkpoint.best_model_score)","\"best_val_wer\": float(checkpoint.best_model_score)"); s=s.replace("verbose=True,\n    )","verbose=True,\n        check_on_train_epoch_end=False,\n    )"); p.write_text(s); print("Changed checkpoint and early stopping metric to val_wer")'
grep -nE 'filename=|monitor=|best_val_|check_on_train' scripts/finetune_nemotron.py
grep -n 'val_loss' scripts/finetune_nemotron.py || echo "No val_loss callback references remain"
docker rm -f nemotron-trainer 2>/dev/null || true; docker run --rm --name nemotron-trainer --gpus all --ipc=host -e PYTHONPATH=/workspace -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TOKENIZERS_PARALLELISM=false -e NUMBA_CACHE_DIR=/tmp/numba_cache -v "$PWD:/workspace" -v "$PWD/ft_models:/srv/models" -w /workspace nemotron_finetuned_3.5 bash -lc 'set -euo pipefail; apt-get update; apt-get install -y --no-install-recommends cuda-nvvm-12-4; rm -rf /var/lib/apt/lists/*; python3.11 -m pip uninstall -y numba-cuda numba llvmlite >/dev/null 2>&1 || true; python3.11 -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4" "llvmlite==0.43.0" "numba==0.60.0"; rm -rf /root/.cache/numba /tmp/numba_cache /srv/models/finetuned_nemotron_candidate_checkpoints; export CUDA_HOME=/usr/local/cuda; NVVM="$(find /usr/local -path "*/nvvm/lib64/libnvvm.so" -print -quit)"; test -n "$NVVM" || { echo "ERROR: libnvvm.so missing"; exit 1; }; export LD_LIBRARY_PATH="$(dirname "$NVVM"):/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"; python3.11 -c "import ctypes,numba,llvmlite,numpy; from numba import cuda; ctypes.CDLL(\"$NVVM\"); print(\"NumPy:\",numpy.__version__); print(\"Numba:\",numba.__version__); print(\"llvmlite:\",llvmlite.__version__); print(\"CUDA available:\",cuda.is_available()); print(\"libNVVM:\",\"$NVVM\")"; echo "===== FINE-TUNING ====="; python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_aligned_aug_manifest.json --val-manifest data/manifests/val_aligned_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/finetuned_nemotron_candidate.nemo --freeze-mode decoder_only --max-epochs 4 --batch-size 1 --accumulate-grad-batches 8 --lr 1e-6 --language en-US --precision bf16-mixed --num-workers 0 --max-duration 20 --patience 2 --seed 42 2>&1 | tee logs/retry_finetuning.log; echo "===== CANDIDATE EVALUATION ====="; python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_nemotron_candidate.nemo --manifest data/manifests/test_aligned_manifest.json --language en-US --output-jsonl results/safe_training/finetuned_test.jsonl; echo "===== DEPLOYMENT GATE ====="; python3.11 scripts/evaluation_gate.py --base results/safe_training/base_test.jsonl --candidate results/safe_training/finetuned_test.jsonl --entity inspira --max-raw-regression 2.0 --max-semantic-regression 0.5 --report results/safe_training/deployment_gate.json; echo "===== PROMOTING MODEL ====="; cp /srv/models/finetuned_nemotron_candidate.nemo /srv/models/finetuned_nemotron_final.nemo; cp /srv/models/finetuned_nemotron_candidate.training_summary.json /srv/models/finetuned_nemotron_final.training_summary.json; ls -lh /srv/models/finetuned_nemotron_final.nemo; echo "FINE-TUNING AND EVALUATION COMPLETED"'
ls -lh ft_models/finetuned_nemotron_candidate.nemo ft_models/finetuned_nemotron_final.nemo
python3 -m json.tool results/safe_training/deployment_gate.json
python3 -c 'import json; r=json.load(open("results/safe_training/deployment_gate.json")); print("DEPLOYMENT PASSED" if r["passed"] else "DEPLOYMENT FAILED"); print(json.dumps(r,indent=2))'
us-central1-docker.pkg.dev/emr-dgt-autonomous-uctr1-snbx/asr-nemotron-3

docker build -t nemotron_finetuned .
docker run -d \
  --name nemotron-base \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  -p 8002:8002 \
  -v "$PWD/audio_logs/base:/srv/audio_logs" \
  -e MODEL_NAME=/srv/nemotron-3.5-asr-streaming-0.6b.nemo \
  -e ENABLE_POSTPROCESSING=false \
  nemotron_finetuned \
  uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8002 \
    --ws-ping-interval 20 \
    --ws-ping-timeout 120


#!/usr/bin/env python3

import argparse
import asyncio
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np
import resampy
import websockets

SERVER_URL = (
    "wss://nemotron-3-5-150916788856."
    "us-central1.run.app/asr/realtime-custom-vad"
)

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2

# Large-file settings
DEFAULT_SEGMENT_SEC = 45
DEFAULT_OVERLAP_SEC = 1
DEFAULT_SPEED = 1.0
DEFAULT_EOF_WAIT_SEC = 120

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

_LANG_TAG_RE = re.compile(r"<[a-z]{2}-[A-Z]{2}>\s*")


def clean_text(text: str) -> str:
    return _LANG_TAG_RE.sub("", text or "").strip()


def print_info(msg: str):
    print(f"{CYAN}[info]{RESET} {msg}")


def print_error(msg: str):
    print(f"{RED}[error]{RESET} {msg}")


def print_partial(text: str):
    sys.stdout.write(f"\r{YELLOW}[partial]{RESET} {text}    ")
    sys.stdout.flush()


def print_final(text: str):
    sys.stdout.write(f"\r{GREEN}{BOLD}[final]{RESET} {GREEN}{text}{RESET}\n")
    sys.stdout.flush()


def upsample_if_needed(pcm: bytes, client_sample_rate: int) -> bytes:
    """
    Resample mono PCM16 audio to 16 kHz.
    Handles both upsampling and downsampling.
    """
    if not pcm or client_sample_rate == SAMPLE_RATE:
        return pcm

    print_info(
        f"Resampling audio from {client_sample_rate}Hz -> {SAMPLE_RATE}Hz"
    )

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    y = resampy.resample(x, client_sample_rate, SAMPLE_RATE)
    y = np.clip(y, -1.0, 1.0)

    return (y * 32767.0).astype(np.int16).tobytes()


def load_wav_16k_mono(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if width != 2:
        raise ValueError(
            f"{path.name}: only 16-bit PCM WAV is supported; got {width * 8}-bit"
        )

    audio = np.frombuffer(raw, dtype=np.int16)

    if channels > 1:
        if len(audio) % channels != 0:
            raise ValueError(f"{path.name}: invalid interleaved PCM data")

        audio = (
            audio.reshape(-1, channels)
            .astype(np.float32)
            .mean(axis=1)
            .clip(-32768, 32767)
            .astype(np.int16)
        )

    raw_16k = upsample_if_needed(audio.tobytes(), sr)
    duration = len(raw_16k) / 2 / SAMPLE_RATE

    print_info(
        f"Audio: {sr}Hz {channels}ch {width * 8}bit "
        f"{duration:.1f}s -> 16000Hz mono PCM16"
    )

    return raw_16k, duration


def split_audio(
    pcm: bytes,
    segment_sec: float,
    overlap_sec: float,
) -> list[tuple[int, int, bytes]]:
    """
    Split 16 kHz mono PCM16 into larger logical segments.

    Returns:
        [(start_sample, end_sample, pcm_bytes), ...]
    """
    samples = np.frombuffer(pcm, dtype=np.int16)

    segment_samples = int(segment_sec * SAMPLE_RATE)
    overlap_samples = int(overlap_sec * SAMPLE_RATE)
    step_samples = segment_samples - overlap_samples

    if segment_samples <= 0:
        raise ValueError("--segment-sec must be > 0")

    if overlap_samples < 0:
        raise ValueError("--overlap-sec cannot be negative")

    if step_samples <= 0:
        raise ValueError("--overlap-sec must be smaller than --segment-sec")

    segments = []

    start = 0

    while start < len(samples):
        end = min(start + segment_samples, len(samples))
        segment = samples[start:end].tobytes()

        segments.append((start, end, segment))

        if end >= len(samples):
            break

        start += step_samples

    return segments


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def merge_transcripts(
    accumulated: str,
    new_text: str,
    max_overlap_words: int = 30,
) -> str:
    """
    Remove exact word overlap caused by overlapped audio segments.
    """
    accumulated = accumulated.strip()
    new_text = new_text.strip()

    if not accumulated:
        return new_text

    if not new_text:
        return accumulated

    old_words = accumulated.split()
    new_words = new_text.split()

    max_n = min(
        max_overlap_words,
        len(old_words),
        len(new_words),
    )

    for n in range(max_n, 0, -1):
        left = [normalize_word(w) for w in old_words[-n:]]
        right = [normalize_word(w) for w in new_words[:n]]

        if left == right:
            return " ".join(old_words + new_words[n:])

    return accumulated + " " + new_text


async def transcribe_segment(
    pcm: bytes,
    url: str,
    language: str,
    speed: float,
    eof_wait: int,
    segment_num: int,
    total_segments: int,
) -> str:
    """
    Send ONE logical segment through ONE fresh WebSocket.

    The segment itself is still sent in 100 ms binary chunks.
    """
    final_texts: list[str] = []
    server_done = asyncio.Event()

    print_info(
        f"[segment {segment_num}/{total_segments}] opening fresh WebSocket"
    )

    async with websockets.connect(
        url,
        ping_interval=None,
        close_timeout=10,
        max_size=None,
    ) as ws:

        await ws.send(
            json.dumps(
                {
                    "backend": "nemotron",
                    "sample_rate": SAMPLE_RATE,
                    "language": language,
                }
            )
        )

        async def receiver():
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type = msg.get("type", "")
                text = clean_text(msg.get("text", ""))

                if event_type == "partial":
                    if text:
                        print_partial(text)

                elif event_type == "final":
                    if text:
                        final_texts.append(text)
                        print_final(text)

                elif event_type == "done":
                    server_done.set()
                    return

                elif event_type == "error":
                    raise RuntimeError(
                        text or f"Server error: {msg}"
                    )

            if not server_done.is_set():
                raise ConnectionError(
                    "WebSocket closed before server sent done"
                )

        recv_task = asyncio.create_task(receiver())

        try:
            chunks = [
                pcm[i:i + CHUNK_BYTES]
                for i in range(0, len(pcm), CHUNK_BYTES)
            ]

            started = time.monotonic()

            for i, chunk in enumerate(chunks):
                if recv_task.done():
                    exc = recv_task.exception()
                    if exc:
                        raise exc
                    if not server_done.is_set():
                        raise ConnectionError(
                            "Receiver stopped before segment finished"
                        )

                await ws.send(chunk)

                expected = (
                    ((i + 1) * CHUNK_MS / 1000.0)
                    / speed
                )
                actual = time.monotonic() - started
                sleep_for = expected - actual

                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

            await ws.send(json.dumps({"type": "eof"}))

            print_info(
                f"[segment {segment_num}/{total_segments}] "
                f"EOF sent; waiting for server done"
            )

            try:
                await asyncio.wait_for(
                    server_done.wait(),
                    timeout=eof_wait,
                )
            except asyncio.TimeoutError as e:
                raise TimeoutError(
                    f"segment {segment_num}: no server done within "
                    f"{eof_wait}s"
                ) from e

            # Let receiver exit naturally after done.
            await recv_task

        finally:
            if not recv_task.done():
                recv_task.cancel()

            try:
                await recv_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    return "\n".join(final_texts).strip()


async def transcribe_large_file(
    wav_path: Path,
    url: str,
    language: str,
    segment_sec: float,
    overlap_sec: float,
    speed: float,
    eof_wait: int,
    save: bool = True,
) -> str | None:
    try:
        pcm, duration = load_wav_16k_mono(wav_path)
    except Exception as e:
        print_error(str(e))
        return None

    segments = split_audio(
        pcm,
        segment_sec=segment_sec,
        overlap_sec=overlap_sec,
    )

    print_info(
        f"{wav_path.name}: {duration:.1f}s -> "
        f"{len(segments)} segment(s), "
        f"{segment_sec:.1f}s each, "
        f"{overlap_sec:.1f}s overlap"
    )

    merged = ""

    for idx, (start, end, segment_pcm) in enumerate(segments, start=1):
        start_sec = start / SAMPLE_RATE
        end_sec = end / SAMPLE_RATE

        print("\n" + "=" * 72)
        print_info(
            f"Segment {idx}/{len(segments)}: "
            f"{start_sec:.1f}s -> {end_sec:.1f}s"
        )
        print("=" * 72)

        try:
            segment_text = await transcribe_segment(
                pcm=segment_pcm,
                url=url,
                language=language,
                speed=speed,
                eof_wait=eof_wait,
                segment_num=idx,
                total_segments=len(segments),
            )

        except Exception as e:
            print_error(
                f"{wav_path.name}: segment {idx} failed: {e}"
            )
            print_error(
                "Final transcript was NOT saved because one segment failed."
            )
            return None

        merged = merge_transcripts(
            merged,
            segment_text,
        )

    merged = merged.strip()

    if save:
        out_path = wav_path.with_suffix(".txt")
        out_path.write_text(
            merged + ("\n" if merged else ""),
            encoding="utf-8",
        )
        print_info(f"Transcript saved: {out_path}")

    return merged


async def run_folder(args):
    folder = Path(args.folder)

    if not folder.exists() or not folder.is_dir():
        print_error(f"Folder not found: {folder}")
        return

    pattern = "**/*.wav" if args.recursive else "*.wav"

    files = sorted(
        [p for p in folder.glob(pattern) if p.is_file()],
        key=lambda p: str(p).lower(),
    )

    if not files:
        print_error(f"No WAV files found in {folder}")
        return

    print_info(f"Found {len(files)} WAV file(s)")

    succeeded = 0
    failed = 0

    for i, wav_path in enumerate(files, start=1):
        print("\n" + "#" * 72)
        print_info(f"[file {i}/{len(files)}] {wav_path}")
        print("#" * 72)

        result = await transcribe_large_file(
            wav_path=wav_path,
            url=args.url,
            language=args.language,
            segment_sec=args.segment_sec,
            overlap_sec=args.overlap_sec,
            speed=args.speed,
            eof_wait=args.eof_wait,
            save=True,
        )

        if result is None:
            failed += 1
        else:
            succeeded += 1

    print("\n" + "=" * 72)
    print_info(
        f"Folder complete. Total={len(files)} "
        f"Succeeded={succeeded} Failed={failed}"
    )


async def run_single(args):
    wav_path = Path(args.file)

    if not wav_path.exists():
        print_error(f"File not found: {wav_path}")
        return

    await transcribe_large_file(
        wav_path=wav_path,
        url=args.url,
        language=args.language,
        segment_sec=args.segment_sec,
        overlap_sec=args.overlap_sec,
        speed=args.speed,
        eof_wait=args.eof_wait,
        save=args.save,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Nemotron large-file chunked WebSocket transcription client"
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument("--file", metavar="PATH")
    mode.add_argument("--folder", metavar="PATH")

    parser.add_argument(
        "--language",
        default="en-US",
    )

    parser.add_argument(
        "--url",
        default=SERVER_URL,
    )

    parser.add_argument(
        "--segment-sec",
        type=float,
        default=DEFAULT_SEGMENT_SEC,
        help=f"Logical audio segment size (default: {DEFAULT_SEGMENT_SEC}s)",
    )

    parser.add_argument(
        "--overlap-sec",
        type=float,
        default=DEFAULT_OVERLAP_SEC,
        help=f"Overlap between logical segments (default: {DEFAULT_OVERLAP_SEC}s)",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=f"Streaming speed; 1.0 = realtime (default: {DEFAULT_SPEED})",
    )

    parser.add_argument(
        "--eof-wait",
        type=int,
        default=DEFAULT_EOF_WAIT_SEC,
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save .txt for single-file mode",
    )

    args = parser.parse_args()

    if args.segment_sec <= 0:
        parser.error("--segment-sec must be > 0")

    if args.overlap_sec < 0:
        parser.error("--overlap-sec cannot be negative")

    if args.overlap_sec >= args.segment_sec:
        parser.error("--overlap-sec must be smaller than --segment-sec")

    if args.speed <= 0:
        parser.error("--speed must be > 0")

    if args.eof_wait <= 0:
        parser.error("--eof-wait must be > 0")

    if args.recursive and not args.folder:
        parser.error("--recursive can only be used with --folder")

    if args.folder:
        asyncio.run(run_folder(args))
    else:
        asyncio.run(run_single(args))


if __name__ == "__main__":
    main()
