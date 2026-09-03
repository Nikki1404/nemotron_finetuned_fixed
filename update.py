docker build -t nemotron_finetuned .
docker rm -f nemotron-base nemotron-ft-en nemotron-trainer 2>/dev/null || true
mkdir -p ft_models logs results/safe_training data/manifests data/audio_16k data/audio_chunks data/audio_aug audio_logs/base audio_logs/finetuned
docker run --rm --name nemotron-trainer --gpus all --ipc=host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TOKENIZERS_PARALLELISM=false -v "$PWD:/workspace" -v "$PWD/ft_models:/srv/models" -w /workspace nemotron_finetuned bash -lc 'set -o pipefail; bash scripts/run_safe_finetuning.sh 2>&1 | tee logs/safe_finetuning.log'
python3 -m json.tool results/safe_training/deployment_gate.json
ls -lh ft_models/finetuned_nemotron_final.nemo
docker run -d --name nemotron-base --restart unless-stopped --gpus all --ipc=host -p 8002:8002 -v "$PWD/audio_logs/base:/srv/audio_logs" -e MODEL_NAME=/srv/nemotron-3.5-asr-streaming-0.6b.nemo nemotron_finetuned
docker run -d --name nemotron-ft-en --restart unless-stopped --gpus all --ipc=host -p 8003:8002 -v "$PWD/ft_models:/srv/models:ro" -v "$PWD/audio_logs/finetuned:/srv/audio_logs" -e MODEL_NAME=/srv/models/finetuned_nemotron_final.nemo -e DOMAIN_VOCAB_PATH=/srv/config/domain_vocabulary.json nemotron_finetuned
curl -s http://localhost:8002/health && echo
curl -s http://localhost:8003/health && echo
nvidia-smi


#client.py-
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

import websockets

#SERVER_URL = "ws://localhost:8002/asr/realtime-custom-vad"
SERVER_URL = "wss://nemotron-3-5-150916788856.us-central1.run.app/asr/realtime-custom-vad"

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2

# --- WebSocket rotation workaround (avoids Cloud Run's ~2-minute cutoff) ---
# The real fix is server-side (see Dockerfile CMD --ws-ping-timeout and the
# Cloud Run service's --timeout). This is a band-aid: proactively close and
# reopen the WebSocket before the server/infra kills it out from under us.
DEFAULT_ROTATE_SOFT_SEC = 180   # rotate at the next finalized utterance after this
DEFAULT_ROTATE_HARD_SEC = 210   # force-rotate here regardless, leaving time to flush before ~240s
EOF_WAIT_SEC = 20               # how long to wait for "done" after sending eof during rotation
RECONNECT_BACKOFF_SEC = 1.5

EOF_SENTINEL = object()

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

_LANG_TAG_RE = re.compile(r"<[a-z]{2}-[A-Z]{2}>\s*")


def clean_text(text: str) -> str:
    return _LANG_TAG_RE.sub("", text or "").strip()


def print_partial(text: str):
    sys.stdout.write(f"\r{YELLOW}[partial]{RESET} {text}    ")
    sys.stdout.flush()


def print_final(text: str, ttfb_ms=None):
    ttfb_str = f"  {DIM}(TTFB {ttfb_ms}ms){RESET}" if ttfb_ms else ""
    sys.stdout.write(f"\r{GREEN}{BOLD}[final]  {RESET}{GREEN}{text}{RESET}{ttfb_str}\n")
    sys.stdout.flush()


def print_corrections(corrections):
    if not corrections:
        return
    sys.stdout.write(f"  {MAGENTA}[corrections]{RESET} {DIM}{corrections}{RESET}\n")
    sys.stdout.flush()


def print_info(msg: str):
    print(f"{CYAN}[info]{RESET} {msg}")


def http_host_from_ws_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = "https" if parsed.scheme in ("wss", "https") else "http"
    return f"{scheme}://{parsed.netloc}"


# ---------------------------------------------------------------------
# Shared streaming core (used by both --mic and --file):
# pulls PCM16 chunks off an asyncio.Queue and sends them over a WebSocket,
# rotating the connection before the server-side timeout kills it, and
# reconnecting transparently. Chunks pulled off the queue but not yet
# successfully sent are requeued so no audio is silently dropped.
# ---------------------------------------------------------------------
async def _receive_loop(ws, got_final: asyncio.Event, leg_done: asyncio.Event):
    """
    Reads server events for one WebSocket leg. Only ever prints a line as
    [final] when the server actually sends a 'final' event — an unconfirmed
    trailing [partial] is never promoted to a final result, including when
    a leg ends due to rotation, a dropped connection, or Ctrl+C.
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
                got_final.set()

            elif ev_type == "done":
                leg_done.set()
                break

            elif ev_type == "error":
                print(f"\n[server error] {text}")

    except websockets.exceptions.ConnectionClosed as e:
        print_info(f"[receive] connection closed: {e}")
    except Exception as e:
        print(f"\n[receive error] {e}")
    finally:
        leg_done.set()


async def _run_one_leg(url, language, queue, stop_all_event, session_num, rotate_soft, rotate_hard):
    """
    Runs a single WebSocket connection ("leg") until it's time to rotate,
    the caller asked to stop, or the input source is fully drained.
    Returns True if the input source is fully finished (EOF_SENTINEL seen).
    """
    print_info(f"[session {session_num}] connecting to {url}")

    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps({
            "backend": "nemotron",
            "sample_rate": SAMPLE_RATE,
            "language": language,
        }))

        got_final = asyncio.Event()
        leg_done = asyncio.Event()
        recv_task = asyncio.create_task(_receive_loop(ws, got_final, leg_done))

        leg_start = time.time()
        input_finished = False
        reason = "stopped"
        elapsed = 0.0

        try:
            while True:
                if stop_all_event.is_set():
                    reason = "stopped"
                    break

                elapsed = time.time() - leg_start

                if elapsed >= rotate_hard:
                    reason = "hard_rotate"
                    break

                if elapsed >= rotate_soft and got_final.is_set():
                    reason = "soft_rotate"
                    break

                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if chunk is EOF_SENTINEL:
                    input_finished = True
                    reason = "input_complete"
                    break

                try:
                    await ws.send(chunk)
                except websockets.exceptions.ConnectionClosed:
                    await queue.put(chunk)  # don't lose this chunk — retry on the next leg
                    raise

        except KeyboardInterrupt:
            reason = "stopped"
            stop_all_event.set()

        # Graceful flush before tearing down this leg.
        try:
            await ws.send(json.dumps({"type": "eof"}))
        except Exception:
            pass

        try:
            await asyncio.wait_for(leg_done.wait(), timeout=EOF_WAIT_SEC)
        except asyncio.TimeoutError:
            print_info(f"[session {session_num}] timed out waiting for flush")

        recv_task.cancel()
        try:
            await recv_task
        except Exception:
            pass

        if reason == "soft_rotate":
            print_info(f"[session {session_num}] rotating connection at {elapsed:.0f}s (after a finalized utterance)")
        elif reason == "hard_rotate":
            print_info(f"[session {session_num}] force-rotating connection at {elapsed:.0f}s (safety cutoff, mid-utterance)")

        return input_finished


async def stream_forever(url, language, queue, stop_all_event, rotate_soft, rotate_hard):
    session_num = 0
    input_finished = False

    while not stop_all_event.is_set() and not input_finished:
        session_num += 1
        try:
            input_finished = await _run_one_leg(
                url, language, queue, stop_all_event, session_num, rotate_soft, rotate_hard
            )
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print_info(f"[session {session_num}] connection dropped ({e}) — reconnecting")
            await asyncio.sleep(RECONNECT_BACKOFF_SEC)


# ---------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------
async def run_file(path: str, language: str, realtime: bool, url: str, rotate_soft: int, rotate_hard: int):
    wav_path = Path(path)

    if not wav_path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        file_sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_audio = wf.readframes(n_frames)

    print_info(f"File: {wav_path.name}")
    print_info(
        f"Audio: {file_sr}Hz {n_channels}ch {sample_width * 8}bit "
        f"{n_frames / file_sr:.1f}s"
    )
    print_info(f"Language: {language}")
    print_info(f"Realtime simulation: {realtime}")
    print_info(f"Connecting to {url}\n")

    import numpy as np

    audio_i16 = np.frombuffer(raw_audio, dtype=np.int16)

    if n_channels == 2:
        audio_i16 = audio_i16.reshape(-1, 2).mean(axis=1).astype(np.int16)

    if file_sr != SAMPLE_RATE:
        print_info(f"Resampling {file_sr}Hz → {SAMPLE_RATE}Hz")

        try:
            import resampy
        except ImportError:
            print("resampy not installed. Run: pip install resampy")
            sys.exit(1)

        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        audio_f32 = resampy.resample(audio_f32, file_sr, SAMPLE_RATE)
        audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)

    raw_bytes = audio_i16.tobytes()
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
    chunk_bytes = chunk_samples * 2

    chunks = [
        raw_bytes[i:i + chunk_bytes]
        for i in range(0, len(raw_bytes), chunk_bytes)
    ]

    queue: asyncio.Queue = asyncio.Queue()
    stop_all_event = asyncio.Event()

    async def producer():
        t_start = time.time()
        for i, chunk in enumerate(chunks):
            await queue.put(chunk)

            if realtime:
                expected_elapsed = (i + 1) * CHUNK_MS / 1000.0
                actual_elapsed = time.time() - t_start
                sleep_for = expected_elapsed - actual_elapsed
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
            else:
                await asyncio.sleep(0.001)

        print_info("File sent — sending EOF and waiting for final results...")
        await queue.put(EOF_SENTINEL)

    t_start = time.time()
    prod_task = asyncio.create_task(producer())

    try:
        await stream_forever(url, language, queue, stop_all_event, rotate_soft, rotate_hard)
    except KeyboardInterrupt:
        print_info("Interrupted while sending audio")
        stop_all_event.set()

    prod_task.cancel()
    try:
        await prod_task
    except asyncio.CancelledError:
        pass

    elapsed = time.time() - t_start
    audio_sec = len(audio_i16) / SAMPLE_RATE
    rtf = elapsed / audio_sec if audio_sec > 0 else 0

    print_info(f"\nDone. Audio={audio_sec:.1f}s Wall={elapsed:.2f}s RTF={rtf:.2f}x")


# ---------------------------------------------------------------------
# Mic mode
# ---------------------------------------------------------------------
async def run_mic(language: str, url: str, rotate_soft: int, rotate_hard: int):
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    print_info(f"Connecting to {url}")
    print_info(f"Language: {language}")
    print_info(
        f"Auto-rotating the WebSocket every ~{rotate_soft}-{rotate_hard}s "
        f"to work around the server's connection limit."
    )
    print_info("Speak into your microphone. Press Ctrl+C to stop.\n")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop_all_event = asyncio.Event()

    def audio_callback(indata, frames, time_info, status):
        pcm = (indata[:, 0] * 32767).astype("int16").tobytes()
        loop.call_soon_threadsafe(queue.put_nowait, pcm)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
        callback=audio_callback,
    ):
        # The mic keeps filling `queue` via the callback above regardless of
        # WebSocket connection state, so no audio is lost while rotating.
        try:
            await stream_forever(url, language, queue, stop_all_event, rotate_soft, rotate_hard)
        except KeyboardInterrupt:
            print_info("Stopping...")
            stop_all_event.set()


# ---------------------------------------------------------------------
# OpenAI-compatible HTTP endpoint test (/v1/audio/transcriptions)
# ---------------------------------------------------------------------
def _build_multipart_form(fields: dict, file_field: str, file_path: Path):
    """
    Minimal stdlib multipart/form-data encoder (no 'requests' dependency),
    matching what FastAPI's File()/Form() parameters on the server expect.
    """
    boundary = uuid.uuid4().hex
    CRLF = "\r\n"
    body = bytearray()

    for name, value in fields.items():
        if value is None:
            continue
        body.extend(f"--{boundary}{CRLF}".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{name}"{CRLF}{CRLF}'.encode()
        )
        body.extend(f"{value}{CRLF}".encode())

    filename = file_path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    body.extend(f"--{boundary}{CRLF}".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"{CRLF}'
            f"Content-Type: {content_type}{CRLF}{CRLF}"
        ).encode()
    )

    with open(file_path, "rb") as f:
        body.extend(f.read())

    body.extend(CRLF.encode())
    body.extend(f"--{boundary}--{CRLF}".encode())

    content_type_header = f"multipart/form-data; boundary={boundary}"
    return bytes(body), content_type_header


async def run_openai_http(
    path: str,
    language: str,
    url: str,
    model: str,
    response_format: str,
):
    """
    Tests the OpenAI-compatible /v1/audio/transcriptions endpoint.
    Sends the whole file in one multipart POST (non-streaming), the same
    way LiveKit or any OpenAI-SDK-compatible client would.
    """
    wav_path = Path(path)

    if not wav_path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

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

    req = urllib.request.Request(endpoint, data=body, method="POST")
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(body)))

    t_start = time.time()
    loop = asyncio.get_running_loop()

    def _do_request():
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except urllib.error.URLError as e:
            return None, str(e).encode()

    status, raw_resp = await loop.run_in_executor(None, _do_request)
    elapsed = time.time() - t_start

    if status is None:
        print(f"[error] Request failed: {raw_resp.decode(errors='replace')}")
        return

    if status != 200:
        print(f"[error] HTTP {status}: {raw_resp.decode(errors='replace')}")
        return

    if response_format == "text":
        print_final(raw_resp.decode("utf-8", errors="replace"))
    else:
        try:
            data = json.loads(raw_resp)
        except json.JSONDecodeError:
            print(f"[warn] Could not parse JSON response: {raw_resp[:500]!r}")
            return

        print_final(data.get("text", ""))

        if response_format == "verbose_json":
            print_info(
                f"duration={data.get('duration')}s language={data.get('language')}"
            )

    print_info(f"\nDone. Wall={elapsed:.2f}s")


async def check_health(url: str):
    try:
        http_host = http_host_from_ws_url(url)

        with urllib.request.urlopen(f"{http_host}/health", timeout=5) as r:
            data = json.loads(r.read())

        print_info(f"Server health: {data}")
        return True

    except Exception as e:
        print(f"[warn] Health check failed: {e} (server may still be starting)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Nemotron ASR client (WebSocket + OpenAI-compatible HTTP)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true", help="Realtime WebSocket streaming from microphone")
    mode.add_argument("--file", metavar="PATH", help="Send a WAV file (WebSocket streaming by default)")

    parser.add_argument("--language", default="en-US")
    parser.add_argument("--realtime", action="store_true", help="[--file only] Pace chunks to simulate realtime playback over the WebSocket")
    parser.add_argument("--url", default=SERVER_URL, help="WebSocket URL (wss://.../asr/realtime-custom-vad)")
    parser.add_argument("--health", action="store_true")

    # Connection rotation workaround (mic/file WebSocket modes)
    parser.add_argument(
        "--rotate-after", type=int, default=DEFAULT_ROTATE_SOFT_SEC,
        help=f"[--mic/--file] seconds after which to rotate the WS connection at the next finalized "
             f"utterance boundary (default: {DEFAULT_ROTATE_SOFT_SEC})",
    )
    parser.add_argument(
        "--rotate-hard", type=int, default=DEFAULT_ROTATE_HARD_SEC,
        help=f"[--mic/--file] hard cutoff in seconds to force-rotate even mid-utterance "
             f"(default: {DEFAULT_ROTATE_HARD_SEC})",
    )
    parser.add_argument(
        "--no-rotate", action="store_true",
        help="[--mic/--file] disable connection rotation entirely (old behavior — will hit the server timeout on long calls)",
    )

    # OpenAI-compatible HTTP endpoint test
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use the OpenAI-compatible HTTP /v1/audio/transcriptions endpoint "
             "instead of the realtime WebSocket. Requires --file (not --mic).",
    )
    parser.add_argument(
        "--model",
        default="nemotron-3.5-asr-streaming-0.6b",
        help="[--openai only] model id, must match what the server expects",
    )
    parser.add_argument(
        "--response-format",
        default="json",
        choices=["json", "text", "verbose_json"],
        help="[--openai only] matches OpenAI's response_format param",
    )

    args = parser.parse_args()

    if args.openai and args.mic:
        parser.error("--openai requires --file (HTTP file upload); it can't be used with --mic")

    if args.rotate_after >= args.rotate_hard:
        parser.error("--rotate-after must be less than --rotate-hard")

    rotate_soft = 10**9 if args.no_rotate else args.rotate_after
    rotate_hard = 10**9 if args.no_rotate else args.rotate_hard

    if args.health:
        asyncio.run(check_health(args.url))
        return

    asyncio.run(check_health(args.url))

    if args.openai:
        asyncio.run(
            run_openai_http(
                args.file, args.language, args.url, args.model, args.response_format
            )
        )
    elif args.mic:
        asyncio.run(run_mic(args.language, args.url, rotate_soft, rotate_hard))
    else:
        asyncio.run(run_file(args.file, args.language, args.realtime, args.url, rotate_soft, rotate_hard))


if __name__ == "__main__":
    main()
