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
import mimetypes
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import resampy
import websockets


# ---------------------------------------------------------------------
# Server / audio configuration
# ---------------------------------------------------------------------

# SERVER_URL = "ws://localhost:8002/asr/realtime-custom-vad"
SERVER_URL = (
    "wss://nemotron-3-5-150916788856."
    "us-central1.run.app/asr/realtime-custom-vad"
)

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2

# File streaming:
#   1.0 = realtime
#   2.0 = twice realtime
#   4.0 = four times realtime
#
# For this streaming ASR endpoint, 2x is a safer batch default than dumping
# the complete audio as fast as the client can send it.
DEFAULT_SEND_SPEED = 2.0

# Maximum time to wait for the server's explicit {"type": "done"} after EOF.
DEFAULT_EOF_WAIT_SEC = 240

# WebSocket rotation safeguard.
DEFAULT_ROTATE_SOFT_SEC = 180
DEFAULT_ROTATE_HARD_SEC = 210

RECONNECT_BACKOFF_SEC = 1.5

# Number of complete-file retries if the WebSocket dies before server "done".
DEFAULT_FILE_RETRIES = 2

EOF_SENTINEL = object()

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

_LANG_TAG_RE = re.compile(r"<[a-z]{2}-[A-Z]{2}>\s*")


# ---------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    return _LANG_TAG_RE.sub("", text or "").strip()


def print_partial(text: str):
    sys.stdout.write(f"\r{YELLOW}[partial]{RESET} {text}    ")
    sys.stdout.flush()


def print_final(text: str, ttfb_ms=None):
    ttfb_str = f"  {DIM}(TTFB {ttfb_ms}ms){RESET}" if ttfb_ms else ""
    sys.stdout.write(
        f"\r{GREEN}{BOLD}[final]  {RESET}{GREEN}{text}{RESET}{ttfb_str}\n"
    )
    sys.stdout.flush()


def print_corrections(corrections):
    if not corrections:
        return
    sys.stdout.write(
        f"  {MAGENTA}[corrections]{RESET} {DIM}{corrections}{RESET}\n"
    )
    sys.stdout.flush()


def print_info(msg: str):
    print(f"{CYAN}[info]{RESET} {msg}")


def print_error(msg: str):
    print(f"{RED}[error]{RESET} {msg}")


def http_host_from_ws_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = "https" if parsed.scheme in ("wss", "https") else "http"
    return f"{scheme}://{parsed.netloc}"


# ---------------------------------------------------------------------
# Audio normalization
# ---------------------------------------------------------------------

def upsample_if_needed(pcm: bytes, client_sample_rate: int) -> bytes:
    """
    Resample mono PCM16 audio to SAMPLE_RATE (16 kHz).

    Despite the historical function name, this performs both upsampling
    and downsampling:
        8 kHz  -> 16 kHz
        48 kHz -> 16 kHz
        16 kHz -> unchanged
    """
    if not pcm or client_sample_rate == SAMPLE_RATE:
        return pcm

    print_info(
        f"Resampling audio from {client_sample_rate}Hz → {SAMPLE_RATE}Hz"
    )

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    y = resampy.resample(
        x,
        client_sample_rate,
        SAMPLE_RATE,
    )

    y = np.clip(y, -1.0, 1.0)

    return (y * 32767.0).astype(np.int16).tobytes()


def load_wav_as_16k_mono_pcm16(wav_path: Path):
    """
    Read a PCM16 WAV, convert multi-channel audio to mono, then resample
    to 16 kHz PCM16.

    Returns:
        raw_bytes, original_sr, n_channels, sample_width, duration_sec
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            file_sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw_audio = wf.readframes(n_frames)
    except (wave.Error, EOFError) as e:
        raise ValueError(f"Could not read WAV file: {e}") from e

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit audio"
        )

    if file_sr <= 0:
        raise ValueError(f"Invalid sample rate: {file_sr}")

    audio_i16 = np.frombuffer(raw_audio, dtype=np.int16)

    if n_channels > 1:
        if len(audio_i16) % n_channels != 0:
            raise ValueError("Invalid interleaved PCM channel data")

        audio_i16 = (
            audio_i16.reshape(-1, n_channels)
            .astype(np.float32)
            .mean(axis=1)
            .clip(-32768, 32767)
            .astype(np.int16)
        )

    raw_bytes = upsample_if_needed(
        audio_i16.tobytes(),
        file_sr,
    )

    duration_sec = len(np.frombuffer(raw_bytes, dtype=np.int16)) / SAMPLE_RATE

    return (
        raw_bytes,
        file_sr,
        n_channels,
        sample_width,
        duration_sec,
    )


# ---------------------------------------------------------------------
# WebSocket receive / one connection leg
# ---------------------------------------------------------------------

async def _receive_loop(
    ws,
    got_final: asyncio.Event,
    server_done: asyncio.Event,
    final_texts: list[str],
):
    """
    Receive ASR events.

    IMPORTANT:
    server_done is set ONLY when the server explicitly sends:
        {"type": "done"}

    A dropped WebSocket is NOT considered successful completion.
    """
    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            ev_type = msg.get("type", "")
            text = clean_text(msg.get("text", ""))
            ttfb = msg.get("t_start")
            corrections = msg.get("corrections")

            if ev_type == "partial":
                if text:
                    print_partial(text)

            elif ev_type == "final":
                if text:
                    print_final(text, ttfb)
                    print_corrections(corrections)
                    final_texts.append(text)

                got_final.set()

            elif ev_type == "done":
                server_done.set()
                return

            elif ev_type == "error":
                raise RuntimeError(
                    text or f"Server returned ASR error: {msg}"
                )

    except asyncio.CancelledError:
        raise

    except websockets.exceptions.ConnectionClosed as e:
        raise ConnectionError(
            f"WebSocket closed before server completion: {e}"
        ) from e


async def _wait_for_server_done(
    recv_task: asyncio.Task,
    server_done: asyncio.Event,
    eof_wait: int,
):
    """
    Wait for either:
      1. explicit server "done", or
      2. receiver failure.

    This avoids waiting the entire EOF timeout after the WebSocket has
    already died.
    """
    done_wait_task = asyncio.create_task(server_done.wait())

    try:
        finished, _ = await asyncio.wait(
            {recv_task, done_wait_task},
            timeout=eof_wait,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not finished:
            raise TimeoutError(
                f"Timed out after {eof_wait}s waiting for server done"
            )

        if recv_task in finished:
            # Propagate receiver errors immediately.
            exc = recv_task.exception()

            if exc is not None:
                raise exc

            # Receiver exited cleanly but no explicit server "done".
            if not server_done.is_set():
                raise ConnectionError(
                    "Receiver stopped before server sent done"
                )

        if server_done.is_set():
            return

        raise ConnectionError(
            "Server connection ended without a done event"
        )

    finally:
        if not done_wait_task.done():
            done_wait_task.cancel()

        try:
            await done_wait_task
        except asyncio.CancelledError:
            pass


async def _run_one_leg(
    url,
    language,
    queue,
    stop_all_event,
    session_num,
    rotate_soft,
    rotate_hard,
    final_texts,
    eof_wait,
):
    """
    Run one WebSocket connection leg.

    Returns:
        True  -> input EOF was reached and server confirmed done
        False -> connection was rotated and more input remains
    """
    print_info(f"[session {session_num}] connecting to {url}")

    # ping_interval=None disables client-side keepalive ping timeouts.
    #
    # This is intentional for the batch-file case because the server may
    # spend a significant amount of time doing inference after receiving
    # buffered audio. The previous 20s ping timeout was closing otherwise
    # active ASR sessions before transcription completed.
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

        got_final = asyncio.Event()
        server_done = asyncio.Event()

        recv_task = asyncio.create_task(
            _receive_loop(
                ws,
                got_final,
                server_done,
                final_texts,
            )
        )

        leg_start = time.monotonic()
        input_finished = False
        reason = "stopped"

        try:
            while True:
                if stop_all_event.is_set():
                    reason = "stopped"
                    break

                # Detect receiver failure immediately while we're still sending.
                if recv_task.done():
                    exc = recv_task.exception()
                    if exc is not None:
                        raise exc

                    if server_done.is_set():
                        return input_finished

                    raise ConnectionError(
                        "Receiver stopped before input completed"
                    )

                elapsed = time.monotonic() - leg_start

                if elapsed >= rotate_hard:
                    reason = "hard_rotate"
                    break

                if (
                    elapsed >= rotate_soft
                    and got_final.is_set()
                ):
                    reason = "soft_rotate"
                    break

                try:
                    chunk = await asyncio.wait_for(
                        queue.get(),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    continue

                if chunk is EOF_SENTINEL:
                    input_finished = True
                    reason = "input_complete"
                    break

                try:
                    await ws.send(chunk)

                except websockets.exceptions.ConnectionClosed as e:
                    # Put unsent audio back so a rotated/reconnected leg
                    # doesn't silently lose this chunk.
                    await queue.put(chunk)

                    raise ConnectionError(
                        f"WebSocket closed while sending audio: {e}"
                    ) from e

            # Tell the server to flush the current leg.
            try:
                await ws.send(json.dumps({"type": "eof"}))
            except websockets.exceptions.ConnectionClosed as e:
                raise ConnectionError(
                    f"WebSocket closed while sending EOF: {e}"
                ) from e

            # Wait for explicit server "done".
            await _wait_for_server_done(
                recv_task,
                server_done,
                eof_wait,
            )

            if reason == "soft_rotate":
                print_info(
                    f"[session {session_num}] rotating connection "
                    f"after finalized utterance"
                )

            elif reason == "hard_rotate":
                print_info(
                    f"[session {session_num}] force-rotating connection "
                    f"at safety cutoff"
                )

            return input_finished

        finally:
            if not recv_task.done():
                recv_task.cancel()

            try:
                await recv_task
            except asyncio.CancelledError:
                pass
            except Exception:
                # The actual error has already been handled / propagated above.
                pass


async def stream_forever(
    url,
    language,
    queue,
    stop_all_event,
    rotate_soft,
    rotate_hard,
    final_texts=None,
    eof_wait=DEFAULT_EOF_WAIT_SEC,
):
    """
    Stream queue contents across one or more WebSocket legs.

    Connection rotation is preserved, but a genuine connection failure is
    propagated to the caller instead of being mistaken for successful EOF.
    """
    if final_texts is None:
        final_texts = []

    session_num = 0
    input_finished = False

    while not stop_all_event.is_set() and not input_finished:
        session_num += 1

        input_finished = await _run_one_leg(
            url=url,
            language=language,
            queue=queue,
            stop_all_event=stop_all_event,
            session_num=session_num,
            rotate_soft=rotate_soft,
            rotate_hard=rotate_hard,
            final_texts=final_texts,
            eof_wait=eof_wait,
        )


# ---------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------

async def _transcribe_file_attempt(
    raw_bytes: bytes,
    language: str,
    realtime: bool,
    send_speed: float,
    url: str,
    rotate_soft: int,
    rotate_hard: int,
    eof_wait: int,
):
    """
    One complete transcription attempt.

    Returns the complete list of confirmed final utterances only if the
    server reaches explicit "done".
    """
    chunks = [
        raw_bytes[i:i + CHUNK_BYTES]
        for i in range(0, len(raw_bytes), CHUNK_BYTES)
    ]

    queue: asyncio.Queue = asyncio.Queue()
    stop_all_event = asyncio.Event()
    final_texts: list[str] = []

    effective_speed = 1.0 if realtime else send_speed

    async def producer():
        t_start = time.monotonic()

        for i, chunk in enumerate(chunks):
            await queue.put(chunk)

            expected_elapsed = (
                ((i + 1) * CHUNK_MS / 1000.0)
                / effective_speed
            )

            actual_elapsed = time.monotonic() - t_start
            sleep_for = expected_elapsed - actual_elapsed

            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

        print_info(
            "File sent — sending EOF and waiting for final results..."
        )

        await queue.put(EOF_SENTINEL)

    prod_task = asyncio.create_task(producer())

    try:
        await stream_forever(
            url=url,
            language=language,
            queue=queue,
            stop_all_event=stop_all_event,
            rotate_soft=rotate_soft,
            rotate_hard=rotate_hard,
            final_texts=final_texts,
            eof_wait=eof_wait,
        )

        return final_texts

    finally:
        stop_all_event.set()

        if not prod_task.done():
            prod_task.cancel()

        try:
            await prod_task
        except asyncio.CancelledError:
            pass


async def run_file(
    path: str,
    language: str,
    realtime: bool,
    url: str,
    rotate_soft: int,
    rotate_hard: int,
    save_transcript: bool = False,
    eof_wait: int = DEFAULT_EOF_WAIT_SEC,
    send_speed: float = DEFAULT_SEND_SPEED,
    file_retries: int = DEFAULT_FILE_RETRIES,
):
    wav_path = Path(path)

    if not wav_path.exists():
        print_error(f"File not found: {path}")
        return None

    if not wav_path.is_file():
        print_error(f"Not a file: {path}")
        return None

    try:
        (
            raw_bytes,
            file_sr,
            n_channels,
            sample_width,
            audio_sec,
        ) = load_wav_as_16k_mono_pcm16(wav_path)

    except ValueError as e:
        print_error(f"{wav_path.name}: {e}")
        return None

    print_info(f"File: {wav_path.name}")
    print_info(
        f"Audio: {file_sr}Hz {n_channels}ch "
        f"{sample_width * 8}bit {audio_sec:.1f}s"
    )
    print_info(f"Language: {language}")
    print_info(
        f"Send speed: {'1.0x realtime' if realtime else f'{send_speed:.2f}x realtime'}"
    )
    print_info(f"Connecting to {url}\n")

    total_start = time.monotonic()

    attempts = max(1, file_retries + 1)

    for attempt in range(1, attempts + 1):
        if attempt > 1:
            print_info(
                f"Retrying entire file from the beginning "
                f"(attempt {attempt}/{attempts})..."
            )

        try:
            final_texts = await _transcribe_file_attempt(
                raw_bytes=raw_bytes,
                language=language,
                realtime=realtime,
                send_speed=send_speed,
                url=url,
                rotate_soft=rotate_soft,
                rotate_hard=rotate_hard,
                eof_wait=eof_wait,
            )

            transcript = "\n".join(final_texts).strip()

            elapsed = time.monotonic() - total_start
            rtf = elapsed / audio_sec if audio_sec > 0 else 0.0

            if save_transcript:
                transcript_path = wav_path.with_suffix(".txt")

                transcript_path.write_text(
                    transcript + ("\n" if transcript else ""),
                    encoding="utf-8",
                )

                print_info(
                    f"Transcript saved: {transcript_path}"
                )

            print_info(
                f"\nDone. Audio={audio_sec:.1f}s "
                f"Wall={elapsed:.2f}s RTF={rtf:.2f}x"
            )

            return transcript

        except (
            ConnectionError,
            TimeoutError,
            OSError,
            websockets.exceptions.WebSocketException,
        ) as e:
            print_error(
                f"{wav_path.name}: transcription attempt "
                f"{attempt}/{attempts} failed: {e}"
            )

            if attempt < attempts:
                await asyncio.sleep(RECONNECT_BACKOFF_SEC)
                continue

            # Never save a normal .txt for an incomplete transcription.
            print_error(
                f"{wav_path.name}: server never confirmed complete "
                f"transcription after {attempts} attempt(s)."
            )

            return None


async def run_folder(
    folder: str,
    language: str,
    realtime: bool,
    url: str,
    rotate_soft: int,
    rotate_hard: int,
    recursive: bool = False,
    eof_wait: int = DEFAULT_EOF_WAIT_SEC,
    send_speed: float = DEFAULT_SEND_SPEED,
    file_retries: int = DEFAULT_FILE_RETRIES,
):
    folder_path = Path(folder)

    if not folder_path.exists():
        print_error(f"Folder not found: {folder}")
        return

    if not folder_path.is_dir():
        print_error(f"Not a folder: {folder}")
        return

    pattern = "**/*.wav" if recursive else "*.wav"

    wav_files = sorted(
        (
            p
            for p in folder_path.glob(pattern)
            if p.is_file()
        ),
        key=lambda p: str(p).lower(),
    )

    if not wav_files:
        print_error(
            f"No WAV files found in: {folder_path}"
        )
        return

    print_info(
        f"Found {len(wav_files)} WAV file(s) in {folder_path}"
    )
    print_info(
        "A .txt file is written only after the server explicitly "
        "confirms transcription completion.\n"
    )

    succeeded = 0
    failed = 0

    for index, wav_path in enumerate(
        wav_files,
        start=1,
    ):
        print("\n" + "=" * 72)
        print_info(
            f"[{index}/{len(wav_files)}] Processing: {wav_path}"
        )
        print("=" * 72)

        try:
            transcript = await run_file(
                path=str(wav_path),
                language=language,
                realtime=realtime,
                url=url,
                rotate_soft=rotate_soft,
                rotate_hard=rotate_hard,
                save_transcript=True,
                eof_wait=eof_wait,
                send_speed=send_speed,
                file_retries=file_retries,
            )

            if transcript is None:
                failed += 1
            else:
                succeeded += 1

        except KeyboardInterrupt:
            raise

        except Exception as e:
            failed += 1
            print_error(
                f"Unexpected failure processing "
                f"{wav_path.name}: {e}"
            )

    print("\n" + "=" * 72)
    print_info(
        f"Folder complete. "
        f"Total={len(wav_files)} "
        f"Succeeded={succeeded} "
        f"Failed={failed}"
    )


# ---------------------------------------------------------------------
# Mic mode
# ---------------------------------------------------------------------

async def run_mic(
    language: str,
    url: str,
    rotate_soft: int,
    rotate_hard: int,
    eof_wait: int,
):
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "sounddevice not installed. "
            "Run: pip install sounddevice"
        )
        sys.exit(1)

    print_info(f"Connecting to {url}")
    print_info(f"Language: {language}")
    print_info("Speak into your microphone. Press Ctrl+C to stop.\n")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop_all_event = asyncio.Event()

    def audio_callback(
        indata,
        frames,
        time_info,
        status,
    ):
        pcm = (
            indata[:, 0] * 32767
        ).astype("int16").tobytes()

        loop.call_soon_threadsafe(
            queue.put_nowait,
            pcm,
        )

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(
            SAMPLE_RATE * CHUNK_MS / 1000
        ),
        callback=audio_callback,
    ):
        try:
            await stream_forever(
                url=url,
                language=language,
                queue=queue,
                stop_all_event=stop_all_event,
                rotate_soft=rotate_soft,
                rotate_hard=rotate_hard,
                eof_wait=eof_wait,
            )

        except KeyboardInterrupt:
            print_info("Stopping...")
            stop_all_event.set()


# ---------------------------------------------------------------------
# OpenAI-compatible HTTP endpoint
# ---------------------------------------------------------------------

def _build_multipart_form(
    fields: dict,
    file_field: str,
    file_path: Path,
):
    boundary = uuid.uuid4().hex
    CRLF = "\r\n"
    body = bytearray()

    for name, value in fields.items():
        if value is None:
            continue

        body.extend(
            f"--{boundary}{CRLF}".encode()
        )

        body.extend(
            (
                f'Content-Disposition: form-data; '
                f'name="{name}"{CRLF}{CRLF}'
            ).encode()
        )

        body.extend(
            f"{value}{CRLF}".encode()
        )

    filename = file_path.name

    content_type = (
        mimetypes.guess_type(filename)[0]
        or "application/octet-stream"
    )

    body.extend(
        f"--{boundary}{CRLF}".encode()
    )

    body.extend(
        (
            f'Content-Disposition: form-data; '
            f'name="{file_field}"; '
            f'filename="{filename}"{CRLF}'
            f"Content-Type: {content_type}{CRLF}{CRLF}"
        ).encode()
    )

    with open(file_path, "rb") as f:
        body.extend(f.read())

    body.extend(CRLF.encode())

    body.extend(
        f"--{boundary}--{CRLF}".encode()
    )

    return (
        bytes(body),
        f"multipart/form-data; boundary={boundary}",
    )


async def run_openai_http(
    path: str,
    language: str,
    url: str,
    model: str,
    response_format: str,
):
    wav_path = Path(path)

    if not wav_path.exists():
        print_error(f"File not found: {path}")
        return

    http_host = http_host_from_ws_url(url)
    endpoint = f"{http_host}/v1/audio/transcriptions"

    print_info(f"File: {wav_path.name}")
    print_info(f"Language: {language}")
    print_info(f"Model: {model}")
    print_info(f"Response format: {response_format}")
    print_info(f"POST {endpoint}\n")

    body, content_type = _build_multipart_form(
        fields={
            "model": model,
            "language": language,
            "response_format": response_format,
        },
        file_field="file",
        file_path=wav_path,
    )

    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
    )

    req.add_header(
        "Content-Type",
        content_type,
    )

    req.add_header(
        "Content-Length",
        str(len(body)),
    )

    t_start = time.monotonic()
    loop = asyncio.get_running_loop()

    def _do_request():
        try:
            with urllib.request.urlopen(
                req,
                timeout=300,
            ) as resp:
                return resp.status, resp.read()

        except urllib.error.HTTPError as e:
            return e.code, e.read()

        except urllib.error.URLError as e:
            return None, str(e).encode()

    status, raw_resp = await loop.run_in_executor(
        None,
        _do_request,
    )

    elapsed = time.monotonic() - t_start

    if status is None:
        print_error(
            f"Request failed: "
            f"{raw_resp.decode(errors='replace')}"
        )
        return

    if status != 200:
        print_error(
            f"HTTP {status}: "
            f"{raw_resp.decode(errors='replace')}"
        )
        return

    if response_format == "text":
        print_final(
            raw_resp.decode(
                "utf-8",
                errors="replace",
            )
        )

    else:
        try:
            data = json.loads(raw_resp)

        except json.JSONDecodeError:
            print_error(
                "Could not parse JSON response"
            )
            return

        print_final(
            data.get("text", "")
        )

        if response_format == "verbose_json":
            print_info(
                f"duration={data.get('duration')}s "
                f"language={data.get('language')}"
            )

    print_info(
        f"\nDone. Wall={elapsed:.2f}s"
    )


# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------

async def check_health(url: str):
    try:
        http_host = http_host_from_ws_url(url)

        with urllib.request.urlopen(
            f"{http_host}/health",
            timeout=5,
        ) as r:
            data = json.loads(r.read())

        print_info(
            f"Server health: {data}"
        )

        return True

    except Exception as e:
        print(
            f"[warn] Health check failed: {e} "
            f"(server may still be starting)"
        )

        return False


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Nemotron ASR client "
            "(WebSocket + OpenAI-compatible HTTP)"
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--mic",
        action="store_true",
        help="Realtime WebSocket streaming from microphone",
    )

    mode.add_argument(
        "--file",
        metavar="PATH",
        help="Send one WAV file",
    )

    mode.add_argument(
        "--folder",
        metavar="PATH",
        help=(
            "Process WAV files in a folder and save "
            "one transcript per WAV"
        ),
    )

    parser.add_argument(
        "--language",
        default="en-US",
        help="Language code (default: en-US)",
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
        help=(
            "[--file/--folder] send audio at 1x realtime "
            "instead of --speed"
        ),
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SEND_SPEED,
        help=(
            "[--file/--folder] file streaming speed when "
            f"--realtime is not used (default: {DEFAULT_SEND_SPEED}x)"
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "[--folder only] also process WAV files "
            "inside subfolders"
        ),
    )

    parser.add_argument(
        "--url",
        default=SERVER_URL,
        help="WebSocket ASR URL",
    )

    parser.add_argument(
        "--health",
        action="store_true",
        help="Run health check before transcription",
    )

    parser.add_argument(
        "--eof-wait",
        type=int,
        default=DEFAULT_EOF_WAIT_SEC,
        help=(
            "[--file/--folder] maximum seconds to wait "
            "for server done after EOF "
            f"(default: {DEFAULT_EOF_WAIT_SEC})"
        ),
    )

    parser.add_argument(
        "--file-retries",
        type=int,
        default=DEFAULT_FILE_RETRIES,
        help=(
            "[--file/--folder] number of complete-file retries "
            "after premature WebSocket failure "
            f"(default: {DEFAULT_FILE_RETRIES})"
        ),
    )

    parser.add_argument(
        "--rotate-after",
        type=int,
        default=DEFAULT_ROTATE_SOFT_SEC,
        help=(
            "seconds before rotating at the next "
            "finalized utterance"
        ),
    )

    parser.add_argument(
        "--rotate-hard",
        type=int,
        default=DEFAULT_ROTATE_HARD_SEC,
        help="hard WebSocket rotation cutoff",
    )

    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="disable WebSocket rotation",
    )

    # OpenAI-compatible HTTP mode.
    parser.add_argument(
        "--openai",
        action="store_true",
        help=(
            "Use /v1/audio/transcriptions instead "
            "of WebSocket. Requires --file."
        ),
    )

    parser.add_argument(
        "--model",
        default="nemotron-3.5-asr-streaming-0.6b",
        help="[--openai only] model id",
    )

    parser.add_argument(
        "--response-format",
        default="json",
        choices=[
            "json",
            "text",
            "verbose_json",
        ],
        help="[--openai only] response format",
    )

    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed must be greater than 0")

    if args.eof_wait <= 0:
        parser.error("--eof-wait must be greater than 0")

    if args.file_retries < 0:
        parser.error("--file-retries cannot be negative")

    if args.recursive and not args.folder:
        parser.error(
            "--recursive can only be used with --folder"
        )

    if args.openai and not args.file:
        parser.error(
            "--openai requires --file"
        )

    if (
        not args.no_rotate
        and args.rotate_after >= args.rotate_hard
    ):
        parser.error(
            "--rotate-after must be less than --rotate-hard"
        )

    rotate_soft = (
        10**9
        if args.no_rotate
        else args.rotate_after
    )

    rotate_hard = (
        10**9
        if args.no_rotate
        else args.rotate_hard
    )

    # Keep the existing behavior of checking health before use.
    asyncio.run(
        check_health(args.url)
    )

    if args.openai:
        asyncio.run(
            run_openai_http(
                path=args.file,
                language=args.language,
                url=args.url,
                model=args.model,
                response_format=args.response_format,
            )
        )

    elif args.mic:
        asyncio.run(
            run_mic(
                language=args.language,
                url=args.url,
                rotate_soft=rotate_soft,
                rotate_hard=rotate_hard,
                eof_wait=args.eof_wait,
            )
        )

    elif args.folder:
        asyncio.run(
            run_folder(
                folder=args.folder,
                language=args.language,
                realtime=args.realtime,
                url=args.url,
                rotate_soft=rotate_soft,
                rotate_hard=rotate_hard,
                recursive=args.recursive,
                eof_wait=args.eof_wait,
                send_speed=args.speed,
                file_retries=args.file_retries,
            )
        )

    else:
        asyncio.run(
            run_file(
                path=args.file,
                language=args.language,
                realtime=args.realtime,
                url=args.url,
                rotate_soft=rotate_soft,
                rotate_hard=rotate_hard,
                save_transcript=False,
                eof_wait=args.eof_wait,
                send_speed=args.speed,
                file_retries=args.file_retries,
            )
        )


if __name__ == "__main__":
    main()
