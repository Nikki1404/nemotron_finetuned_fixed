#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

BASE_MODEL=/srv/nemotron-3.5-asr-streaming-0.6b.nemo
FINAL_MODEL=/srv/models/finetuned_nemotron_final.nemo
CSV=data/inspira_transcripts.csv

mkdir -p /srv/models logs results/safe_training data/manifests

printf '\n===== 1. PREPARE 16 KHZ AUDIO =====\n'
python3.11 scripts/prepare_dataset.py \
  --csv "$CSV" \
  --wav-dir raw_wavs \
  --out-dir data

printf '\n===== 2. BUILD SAFE MONOTONIC CHUNKS =====\n'
rm -rf data/audio_chunks data/audio_aug
python3.11 scripts/auto_align_chunks_with_base_asr.py \
  --csv "$CSV" \
  --wav-dir data/audio_16k \
  --base-model "$BASE_MODEL" \
  --out-dir data/audio_chunks \
  --manifest data/manifests/aligned_chunk_manifest.json \
  --audit data/manifests/alignment_audit.csv \
  --language en-US \
  --chunk-sec 8 \
  --min-score 0.78 \
  --force

printf '\n===== 3. SPLIT BY WHOLE CALL / USE CASE =====\n'
python3.11 scripts/split_by_usecase_manifest.py \
  --input data/manifests/aligned_chunk_manifest.json \
  --out-dir data/manifests \
  --seed 42

printf '\n===== 4. VALIDATE CLEAN MANIFESTS =====\n'
for manifest in \
  data/manifests/train_aligned_manifest.json \
  data/manifests/val_aligned_manifest.json \
  data/manifests/test_aligned_manifest.json; do
  python3.11 scripts/validate_manifest.py \
    --manifest "$manifest" \
    --max-duration 20 \
    --require-files
done

printf '\n===== 5. TRAIN-ONLY SAFE AUGMENTATION =====\n'
python3.11 scripts/augment_train_manifest.py \
  --train-manifest data/manifests/train_aligned_manifest.json \
  --out-manifest data/manifests/train_aligned_aug_manifest.json \
  --out-audio-dir data/audio_aug \
  --profile safe \
  --keep-original

python3.11 scripts/validate_manifest.py \
  --manifest data/manifests/train_aligned_aug_manifest.json \
  --max-duration 20 \
  --require-files

printf '\n===== 6. BASELINE ON UNTOUCHED TEST CALLS =====\n'
python3.11 scripts/evaluate_manifest.py \
  --model "$BASE_MODEL" \
  --manifest data/manifests/test_aligned_manifest.json \
  --language en-US \
  --output-jsonl results/safe_training/base_test.jsonl

printf '\n===== 7. CONSERVATIVE DECODER-ONLY FINE-TUNING =====\n'
rm -rf /srv/models/finetuned_nemotron_candidate_checkpoints
python3.11 scripts/finetune_nemotron.py \
  --train-manifest data/manifests/train_aligned_aug_manifest.json \
  --val-manifest data/manifests/val_aligned_manifest.json \
  --base-model "$BASE_MODEL" \
  --output-nemo /srv/models/finetuned_nemotron_candidate.nemo \
  --freeze-mode decoder_only \
  --max-epochs 4 \
  --batch-size 1 \
  --accumulate-grad-batches 8 \
  --lr 1e-6 \
  --language en-US \
  --precision bf16-mixed \
  --num-workers 0 \
  --max-duration 20 \
  --patience 2 \
  --seed 42

printf '\n===== 8. FINAL TEST EVALUATION =====\n'
python3.11 scripts/evaluate_manifest.py \
  --model /srv/models/finetuned_nemotron_candidate.nemo \
  --manifest data/manifests/test_aligned_manifest.json \
  --language en-US \
  --output-jsonl results/safe_training/finetuned_test.jsonl

printf '\n===== 9. HELD-OUT DEPLOYMENT GATE =====\n'
python3.11 scripts/evaluation_gate.py \
  --base results/safe_training/base_test.jsonl \
  --candidate results/safe_training/finetuned_test.jsonl \
  --entity inspira \
  --max-raw-regression 2.0 \
  --max-semantic-regression 0.5 \
  --report results/safe_training/deployment_gate.json

printf '\n===== 10. PROMOTE VALIDATED CANDIDATE =====\n'
cp /srv/models/finetuned_nemotron_candidate.nemo "$FINAL_MODEL"
cp /srv/models/finetuned_nemotron_candidate.training_summary.json \
   /srv/models/finetuned_nemotron_final.training_summary.json
ls -lh "$FINAL_MODEL"

echo
printf 'Training complete. Review these before deployment:\n'
printf '  data/manifests/alignment_audit.csv\n'
printf '  results/safe_training/base_test.jsonl\n'
printf '  results/safe_training/finetuned_test.jsonl\n'
