"""Safe domain and number post-processing for ASR output."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Optional

from app.asr_number_normalizer import ContextualNumberNormalizer


@dataclass
class Correction:
    kind: str
    original: str
    replacement: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "original": self.original,
            "replacement": self.replacement,
            "reason": self.reason,
        }


@dataclass
class ProcessedTranscript:
    text: str
    raw_text: str
    corrections: list[Correction] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.text != self.raw_text

    def correction_dicts(self) -> list[dict[str, str]]:
        return [item.as_dict() for item in self.corrections]


class DomainEntityCorrector:
    """Exact, vocabulary-driven corrections with optional context guards.

    This is intentionally not a fuzzy global spell checker.  A phrase is
    changed only when it exactly matches a configured variant.  Therefore the
    ordinary verb "inspire" is not changed unless it appears in the configured
    domain phrase "Inspire Financial" or "Inspire debit card".
    """

    def __init__(self, vocabulary_path: Optional[str] = None) -> None:
        default_path = Path(__file__).resolve().parent.parent / "config" / "domain_vocabulary.json"
        self.vocabulary_path = Path(
            vocabulary_path
            or os.getenv("DOMAIN_VOCAB_PATH", str(default_path))
        )
        self.entities = self._load_entities()

    def _load_entities(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.vocabulary_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid domain vocabulary JSON: {self.vocabulary_path}: {exc}") from exc
        entities = payload.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError("domain_vocabulary.json must contain an 'entities' list")
        return entities

    @staticmethod
    def _context_matches(text: str, required: list[str]) -> bool:
        if not required:
            return True
        low = text.lower()
        return any(term.lower() in low for term in required)

    def correct(self, text: str) -> tuple[str, list[Correction]]:
        if not text or not self.entities:
            return text, []

        output = text
        corrections: list[Correction] = []
        for entity in self.entities:
            canonical = str(entity.get("canonical", "")).strip()
            variants = entity.get("variants", [])
            required_context = [str(x) for x in entity.get("required_context", [])]
            if not canonical or not isinstance(variants, list):
                continue
            if not self._context_matches(output, required_context):
                continue

            for variant in variants:
                variant = str(variant).strip()
                if not variant:
                    continue
                pattern = re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)", re.IGNORECASE)
                matches = list(pattern.finditer(output))
                if not matches:
                    continue
                original_matches = [m.group(0) for m in matches]
                output = pattern.sub(canonical, output)
                for original in original_matches:
                    corrections.append(
                        Correction(
                            kind="domain_entity",
                            original=original,
                            replacement=canonical,
                            reason="exact configured variant with required context",
                        )
                    )
        return output, corrections


class TranscriptPostProcessor:
    def __init__(self, vocabulary_path: Optional[str] = None) -> None:
        self.entity_corrector = DomainEntityCorrector(vocabulary_path)
        self.number_normalizer = ContextualNumberNormalizer()

    def reset(self) -> None:
        self.number_normalizer.reset()

    def process(self, text: str, *, is_final: bool) -> ProcessedTranscript:
        raw = text or ""
        corrected, corrections = self.entity_corrector.correct(raw)
        numbered = self.number_normalizer.process(corrected, is_final=is_final)
        if numbered != corrected:
            corrections.append(
                Correction(
                    kind="number_normalization",
                    original=corrected,
                    replacement=numbered,
                    reason="unambiguous number phrase or strong numeric context",
                )
            )
        return ProcessedTranscript(
            text=numbered,
            raw_text=raw,
            corrections=corrections,
        )
