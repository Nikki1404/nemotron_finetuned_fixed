#!/usr/bin/env python3
import argparse
import asyncio
import json
import re
import sys
import time
import wave
from pathlib import Path
from urllib.parse import urlparse

import websockets

SERVER_URL = "ws://localhost:8003/asr/realtime-custom-vad"
SAMPLE_RATE = 16000
CHUNK_MS = 30
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CLEAR_LINE = "\r\033[2K"
_LANG_TAG_RE = re.compile(r"<[a-z]{2}-[A-Z]{2}>\s*")


def clean_text(text):
    return _LANG_TAG_RE.sub("", text or "").strip()


def print_partial(text):
    # Clear and replace the same terminal line. This prevents cumulative
    # partial hypotheses from looking like repeated transcript paragraphs.
    sys.stdout.write(f"{CLEAR_LINE}{YELLOW}[partial]{RESET} {text}")
    sys.stdout.flush()


def print_final(text, ttfb_ms=None):
    suffix = f"  {DIM}(TTFB {ttfb_ms}ms){RESET}" if ttfb_ms is not None else ""
    sys.stdout.write(
        f"{CLEAR_LINE}{GREEN}{BOLD}[final]  {RESET}{GREEN}{text}{RESET}{suffix}\n"
    )
    sys.stdout.flush()


def print_info(message):
    print(f"{CYAN}[info]{RESET} {message}")


async def receive_loop(ws, stop_event):
    last_partial = ""
    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            typ = msg.get("type", "")
            text = clean_text(msg.get("text", ""))
            ttfb = msg.get("ttfb_ms", msg.get("t_start"))

            if typ == "partial" and text:
                if text != last_partial:
                    last_partial = text
                    print_partial(text)
            elif typ == "final" and text:
                print_final(text, ttfb)
                last_partial = ""
            elif typ == "done":
                stop_event.set()
                break
            elif typ == "error":
                print(f"\n[server error] {text or msg.get('error', '')}")
    except Exception as exc:
        print(f"\n[receive error] {exc}")
    finally:
        stop_event.set()


async def run_file(path, language, realtime, url):
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

    if sample_width != 2:
        raise ValueError("Client currently expects 16-bit PCM WAV input")

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
    if n_channels > 1:
        audio_i16 = audio_i16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    if file_sr != SAMPLE_RATE:
        print_info(f"Resampling {file_sr}Hz → {SAMPLE_RATE}Hz")
        import resampy

        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        audio_f32 = resampy.resample(audio_f32, file_sr, SAMPLE_RATE)
        audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)

    raw_bytes = audio_i16.tobytes()
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
    chunk_bytes = chunk_samples * 2
    chunks = [raw_bytes[i : i + chunk_bytes] for i in range(0, len(raw_bytes), chunk_bytes)]

    t_start = time.time()
    async with websockets.connect(url, ping_interval=None, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {"backend": "nemotron", "sample_rate": SAMPLE_RATE, "language": language}
            )
        )
        stop_event = asyncio.Event()
        recv_task = asyncio.create_task(receive_loop(ws, stop_event))

        for i, chunk in enumerate(chunks):
            await ws.send(chunk)
            if realtime:
                sleep_for = ((i + 1) * CHUNK_MS / 1000.0) - (time.time() - t_start)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
            else:
                await asyncio.sleep(0.001)

        print_info("\nFile sent — sending EOF and waiting for final results...")
        await ws.send(json.dumps({"type": "eof"}))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=90.0)
        except asyncio.TimeoutError:
            print_info("Timeout waiting for final response")
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    elapsed = time.time() - t_start
    audio_sec = len(audio_i16) / SAMPLE_RATE
    rtf = elapsed / audio_sec if audio_sec > 0 else 0
    print_info(f"\nDone. Audio={audio_sec:.1f}s Wall={elapsed:.2f}s RTF={rtf:.2f}x")


async def run_mic(language, url):
    import sounddevice as sd

    print_info(f"Connecting to {url}")
    print_info(f"Language: {language}")
    print_info("Speak into your microphone. Press Ctrl+C to stop.\n")

    async with websockets.connect(url, ping_interval=None, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {"backend": "nemotron", "sample_rate": SAMPLE_RATE, "language": language}
            )
        )
        stop_event = asyncio.Event()
        recv_task = asyncio.create_task(receive_loop(ws, stop_event))
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                print(f"\n[audio warning] {status}")
            pcm = (indata[:, 0] * 32767).astype("int16").tobytes()
            loop.call_soon_threadsafe(queue.put_nowait, pcm)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
                callback=callback,
            ):
                while not stop_event.is_set():
                    try:
                        await ws.send(await asyncio.wait_for(queue.get(), timeout=0.5))
                    except asyncio.TimeoutError:
                        continue
        except KeyboardInterrupt:
            print_info("Stopping...")

        await ws.send(json.dumps({"type": "eof"}))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=90.0)
        except asyncio.TimeoutError:
            print_info("Timeout waiting for final response")
        recv_task.cancel()


async def check_health(url):
    try:
        import urllib.request

        parsed = urlparse(url)
        scheme = "https" if parsed.scheme == "wss" else "http"
        with urllib.request.urlopen(f"{scheme}://{parsed.netloc}/health", timeout=5) as response:
            data = json.loads(response.read())
        print_info(f"Server health: {data}")
        return True
    except Exception as exc:
        print(f"[warn] Health check failed: {exc} (server may still be starting)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Nemotron ASR WebSocket client")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true")
    mode.add_argument("--file", metavar="PATH")
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--url", default=SERVER_URL)
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()

    if args.health:
        asyncio.run(check_health(args.url))
        return
    asyncio.run(check_health(args.url))
    asyncio.run(
        run_mic(args.language, args.url)
        if args.mic
        else run_file(args.file, args.language, args.realtime, args.url)
    )


if __name__ == "__main__":
    main()
