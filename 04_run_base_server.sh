#!/usr/bin/env bash
set -euo pipefail
mkdir -p audio_logs

docker run --gpus all -it --rm \
  -p 8003:8003 \
  -v "$PWD/audio_logs:/srv/audio_logs" \
  -e MODEL_NAME=/srv/nemotron-3.5-asr-streaming-0.6b.nemo \
  -e DOMAIN_VOCAB_PATH=/srv/config/domain_vocabulary.json \
  -e VAD_START_MARGIN=1.8 \
  -e VAD_MIN_NOISE_RMS=0.002 \
  -e PRE_SPEECH_MS=500 \
  -e NEMO_END_SILENCE_MS=900 \
  -e FINALIZE_PAD_MS=800 \
  -e CONTEXT_RIGHT=2 \
  -e NEMO_MAX_SYMBOLS=15 \
  -e PARTIAL_EMIT_INTERVAL_MS=300 \
  -e PARTIAL_MIN_NEW_CHARS=4 \
  nemotron_finetuned \
  uvicorn app.main:app --host 0.0.0.0 --port 8003 \
    --ws-ping-interval 20 --ws-ping-timeout 120
