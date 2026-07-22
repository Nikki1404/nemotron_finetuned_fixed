# Nemotron Fine-Tuning Fix Report

## Root causes found in the uploaded project

1. The observed malformed IDs were primarily produced by the old post-processor, not directly by Nemotron:

   - `twenty thirty four` -> `203004`
   - `twelve forty three` -> `124003`
   - `fifteen thirty eight` -> `153008`
   - `two thousand thirty four` -> `23004`
   - `twelve hundred thirty four` -> `12304`
   - `twenty forty three` -> `204003`
   - `twelve thirty four` -> `123004`

2. Numeric training labels did not match the speech. Audio saying number words was aligned to labels containing digits. The corrected pipeline trains spoken forms and performs display normalization only after recognition.

3. The old whole-call manifests had durations of approximately 53, 53, 63, 89, 89, 97, and 191 seconds, while the trainer used `max_duration=60`. Four calls were therefore outside the training duration limit in that path.

4. The old chunk aligner searched from `cursor - 10`, allowing a new chunk to reuse words already assigned to an earlier chunk.

5. Old aligned data quality:

   - 67 aligned rows
   - 28 rows with numeric display tokens
   - 1 empty row
   - 6 rows below alignment score 0.78
   - 315 augmented train rows generated from only 45 aligned rows

6. The repeated partial text was a cumulative-hypothesis display/state issue. The model returns the current full partial hypothesis; the client/logging path repeatedly printed it as a new line.

7. The streaming finalizer consumed only one pending encoder shift, which could truncate the end of an utterance.

## Main corrections

- Replaced the number normalizer with a conservative, contextual implementation.
- Added exact vocabulary-driven domain correction using `config/domain_vocabulary.json`.
- Preserved `raw_text` and correction metadata for every changed event.
- Added stateful numeric context across turns, such as a member-ID question followed by a numeric answer.
- Throttled cumulative partial events and replaced the same terminal line in `client.py`.
- Removed duplicate VAD trigger-frame ingestion.
- Drained all pending encoder shifts during finalization.
- Made `NEMO_MAX_SYMBOLS` configurable.
- Rebuilt chunk alignment as strictly monotonic and quality-filtered.
- Reconstructed spoken numeric labels from base-ASR drafts.
- Added manifest validation and train-only conservative augmentation.
- Added validation-best checkpointing, early stopping, low learning rate, and decoder/joint-only fine-tuning.
- Added baseline and fine-tuned held-out evaluation with raw and semantic WER.

## Verified locally

- All Python files compile.
- All shell scripts pass syntax validation.
- Post-processing tests pass: `3 passed`.
- Required number formats produce the expected values.
- Ordinary phrases such as `about two days ago` and `one to three business days` remain unchanged.
- `I want to inspire financial confidence` remains unchanged.
- `Hello thank you for calling Inspire Financial` becomes `Hello thank you for calling Inspira Financial`.

## Not performed here

The uploaded ZIP did not include the seven WAV files or a `.nemo` model, and this environment did not run the GPU fine-tuning job. Run `scripts/run_safe_finetuning.sh` on the EC2 GPU instance after copying the WAV files into `raw_wavs/`.
- Added a held-out deployment gate. A candidate is not promoted when raw WER regresses by more than 2 absolute points, semantic WER regresses by more than 0.5 points, or exact `Inspira` recall is lower than the base model.
