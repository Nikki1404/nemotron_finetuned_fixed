from dataclasses import dataclass, replace
import os


@dataclass(frozen=True)
class Config:
    # ── Core ASR selection ──────────────────────────────────────────────────
    asr_backend: str = os.getenv("ASR_BACKEND", "nemotron")
    model_name: str = os.getenv("MODEL_NAME", "")
    device: str = os.getenv("DEVICE", "cuda")

    # ── Common audio ────────────────────────────────────────────────────────
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # ── Common VAD ──────────────────────────────────────────────────────────
    vad_frame_ms: int = int(os.getenv("VAD_FRAME_MS", "20"))
    vad_start_margin: float = float(os.getenv("VAD_START_MARGIN", "1.5"))
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.0015"))
    pre_speech_ms: int = int(os.getenv("PRE_SPEECH_MS", "700"))

    # ── Partial output throttling ───────────────────────────────────────────
    partial_emit_interval_ms: int = int(os.getenv("PARTIAL_EMIT_INTERVAL_MS", "300"))
    partial_min_new_chars: int = int(os.getenv("PARTIAL_MIN_NEW_CHARS", "4"))

    # ── Global safety constraint ────────────────────────────────────────────
    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))

    log_level: str = os.getenv("LOG_LEVEL", "INFO")



MODEL_MAP = {
    # Points to the .nemo file saved during Docker build.
    # Falls back to HuggingFace download if running outside Docker.
    "nemotron": os.getenv(
        "MODEL_NAME",
        "/srv/nemotron-3.5-asr-streaming-0.6b.nemo"
    ),
}


def load_config() -> Config:
    cfg = Config()

    if not cfg.model_name:
        cfg = replace(cfg, model_name=MODEL_MAP.get(cfg.asr_backend, ""))

    print(
        f"DEBUG: Startup cfg.model_name='{cfg.model_name}' "
        f"cfg.asr_backend='{cfg.asr_backend}'"
    )

    return cfg
