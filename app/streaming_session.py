import time
from app.vad import AdaptiveEnergyVAD


class StreamingSession:
    def __init__(self, engine, cfg):
        self.engine = engine
        self.cfg = cfg
        self.vad = AdaptiveEnergyVAD(
            cfg.sample_rate,
            cfg.vad_frame_ms,
            cfg.vad_start_margin,
            cfg.vad_min_noise_rms,
            cfg.pre_speech_ms,
        )
        self.session = engine.new_session(max_buffer_ms=cfg.max_utt_ms)
        self.frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000) * 2
        self.raw_buf = bytearray()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.t_utt_start = None
        self.t_first_partial = None
        self.silence_ms = 0
        self.last_partial = ""
        self.last_emitted_partial = ""
        self.last_partial_emit_monotonic = 0.0

    @staticmethod
    def _common_prefix_len(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _should_emit_partial(self, text: str) -> bool:
        text = (text or "").strip()
        if not text or text == self.last_emitted_partial:
            return False
        if not self.last_emitted_partial:
            return True

        now = time.monotonic()
        elapsed_ms = (now - self.last_partial_emit_monotonic) * 1000.0
        prefix = self._common_prefix_len(self.last_emitted_partial, text)
        changed_chars = max(
            len(text) - prefix,
            len(self.last_emitted_partial) - prefix,
        )

        # Emit immediately on a meaningful hypothesis rewrite; otherwise
        # throttle cumulative partials so logs do not contain the same long
        # sentence every 20 ms.
        rewrite = prefix < int(0.75 * min(len(text), len(self.last_emitted_partial)))
        enough_growth = changed_chars >= self.cfg.partial_min_new_chars
        timed_out = elapsed_ms >= self.cfg.partial_emit_interval_ms
        sentence_boundary = text.endswith((".", "?", "!"))
        return rewrite or sentence_boundary or (enough_growth and timed_out)

    def process_chunk(self, pcm: bytes) -> list:
        events = []
        self.raw_buf.extend(pcm)
        while len(self.raw_buf) >= self.frame_bytes:
            frame = bytes(self.raw_buf[: self.frame_bytes])
            del self.raw_buf[: self.frame_bytes]
            is_speech, pre = self.vad.push_frame(frame)
            self.silence_ms = 0 if is_speech else self.silence_ms + self.cfg.vad_frame_ms

            frame_already_in_pre_roll = False
            if pre and not self.utt_started:
                self.utt_started = True
                self.utt_audio_ms = int(len(pre) / 2 / self.cfg.sample_rate * 1000)
                self.t_utt_start = time.time()
                self.t_first_partial = None
                self.last_partial = ""
                self.last_emitted_partial = ""
                self.last_partial_emit_monotonic = 0.0
                self.session.accept_pcm16(pre)
                # AdaptiveEnergyVAD includes the triggering frame in pre-roll.
                # Do not append the same frame a second time.
                frame_already_in_pre_roll = True

            if not self.utt_started:
                continue

            if not frame_already_in_pre_roll:
                self.session.accept_pcm16(frame)
                self.utt_audio_ms += self.cfg.vad_frame_ms

            if self.engine.caps.partials:
                text = self.session.step_if_ready()
                if text:
                    self.last_partial = text
                    if self._should_emit_partial(text):
                        if self.t_first_partial is None:
                            self.t_first_partial = time.time()
                        self.last_emitted_partial = text
                        self.last_partial_emit_monotonic = time.monotonic()
                        ttfb_ms = int((self.t_first_partial - self.t_utt_start) * 1000)
                        events.append(("partial", text, ttfb_ms))

            if (
                not is_speech
                and self.utt_audio_ms >= self.engine.min_utt_ms
                and self.silence_ms >= self.engine.end_silence_ms
            ):
                final = self.session.finalize(self.engine.finalize_pad_ms)
                if not final and self.last_partial:
                    final = self.last_partial
                if final:
                    ttfb_ms = (
                        int((self.t_first_partial - self.t_utt_start) * 1000)
                        if self.t_first_partial
                        else None
                    )
                    events.append(("final", final, ttfb_ms))
                self.reset()
        return events

    def flush(self) -> list:
        events = []
        if not self.utt_started:
            return events
        final = self.session.finalize(self.engine.finalize_pad_ms)
        if not final and self.last_partial:
            final = self.last_partial
        if final:
            ttfb_ms = (
                int((self.t_first_partial - self.t_utt_start) * 1000)
                if self.t_first_partial
                else None
            )
            events.append(("final", final, ttfb_ms))
        self.reset()
        return events

    def reset(self):
        self.vad.reset()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0
        self.last_partial = ""
        self.last_emitted_partial = ""
        self.last_partial_emit_monotonic = 0.0
