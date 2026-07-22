#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr

from app.asr_number_normalizer import normalize_asr_numbers
from app.transcript_postprocessor import DomainEntityCorrector

LANG_TAG_RE = re.compile(r"<[a-z]{2}-[A-Z]{2}>", re.IGNORECASE)


def patch_prompt(default_lang: str) -> None:
    import nemo.collections.asr.data.audio_to_text_lhotse_prompt_index as mod

    for _, cls in vars(mod).items():
        if inspect.isclass(cls) and hasattr(cls, "_get_prompt_index"):
            old = cls._get_prompt_index

            def patched(self, prompt_key, _old=old):
                if prompt_key is None or str(prompt_key).lower() == "none":
                    prompt_key = default_lang
                return _old(self, prompt_key)

            cls._get_prompt_index = patched


def clean_output(text: str) -> str:
    text = LANG_TAG_RE.sub(" ", text or "")
    text = re.sub(r"\ben\s+us\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def norm(text: str) -> str:
    text = clean_output(text).lower()
    text = re.sub(r"[^a-z0-9ñáéíóúü\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref).split(), norm(hyp).split()
    return 0.0 if not r and not h else (100.0 if not r else 100.0 * edit_distance(r, h) / len(r))


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    return 0.0 if not r and not h else (100.0 if not r else 100.0 * edit_distance(r, h) / len(r))


def get_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return get_text(value[0])
    if hasattr(value, "text"):
        return value.text
    if isinstance(value, dict):
        return value.get("text") or value.get("pred_text") or str(value)
    return str(value)


def semantic_form(text: str, corrector: DomainEntityCorrector) -> str:
    text = clean_output(text)
    text, _ = corrector.correct(text)
    return normalize_asr_numbers(text, force=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-jsonl", default="eval_predictions.jsonl")
    args = parser.parse_args()

    patch_prompt(args.language)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    print(f"[eval] Loading model: {args.model}")
    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location=args.device).to(args.device)
    model.eval()
    corrector = DomainEntityCorrector()

    outputs = []
    raw_wer_total = raw_cer_total = semantic_wer_total = 0.0
    for index, row in enumerate(rows, 1):
        audio = row["audio_filepath"]
        lang = row.get("target_lang") or row.get("language") or args.language
        try:
            model.set_inference_prompt(lang)
        except Exception:
            pass
        print(f"[eval] {index}/{len(rows)} {audio} lang={lang}")
        with torch.no_grad():
            prediction = clean_output(get_text(model.transcribe([audio], batch_size=1, verbose=False)))
        reference = row.get("text", "")
        raw_w = wer(reference, prediction)
        raw_c = cer(reference, prediction)
        semantic_ref = semantic_form(reference, corrector)
        semantic_pred = semantic_form(prediction, corrector)
        semantic_w = wer(semantic_ref, semantic_pred)
        raw_wer_total += raw_w
        raw_cer_total += raw_c
        semantic_wer_total += semantic_w
        print(f"Raw WER: {raw_w:.2f}% | Semantic WER: {semantic_w:.2f}% | CER: {raw_c:.2f}%")
        print(f"PRED: {prediction[:300]}")
        outputs.append(
            {
                "audio_filepath": audio,
                "language": lang,
                "reference": reference,
                "prediction": prediction,
                "semantic_reference": semantic_ref,
                "semantic_prediction": semantic_pred,
                "wer": raw_w,
                "semantic_wer": semantic_w,
                "cer": raw_c,
            }
        )

    count = max(1, len(rows))
    Path(args.output_jsonl).write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in outputs) + "\n",
        encoding="utf-8",
    )
    print("\n========== SUMMARY ==========")
    print(f"Files: {len(rows)}")
    print(f"Average raw WER: {raw_wer_total / count:.2f}%")
    print(f"Average semantic WER: {semantic_wer_total / count:.2f}%")
    print(f"Average CER: {raw_cer_total / count:.2f}%")
    print(f"Saved predictions: {args.output_jsonl}")


if __name__ == "__main__":
    main()
