# Nemotron 3.5 Inspira ASR — Corrected Fine-Tuning Project

This project keeps the same FastAPI WebSocket architecture, the same
`AdaptiveEnergyVAD`, the same Nemotron streaming engine, and the same Docker
workflow. It fixes the data-label, number-normalization, entity-correction,
partial-output, and final-flush problems found in the uploaded project.

## What was wrong in the previous project

1. **Four of five whole-call training files were silently excluded.**
   `finetune_nemotron.py` used `max_duration=60`, while the old training calls
   were approximately 63, 89, 97, and 191 seconds. Only the 53-second
   Verification Code call remained in that path.

2. **Audio and numeric labels did not match.**
   The audio said forms such as `twenty forty three`, but manifests used `2043`.
   The model was therefore asked to learn a display-format transformation as
   part of acoustic recognition. The old custom normalizer then converted
   `twenty forty three` to `204003` and `twelve thirty four` to `123004`.

3. **Chunk alignment could move backward.**
   A chunk was allowed to align up to ten words before the previous cursor.
   That duplicated boundary words and shifted labels across adjacent chunks.

4. **Unsafe chunks were still used.**
   The old alignment included an empty chunk, six chunks below a 0.78 match
   score, and many labels containing numeric display tokens.

5. **Partials were cumulative.**
   Nemotron returns the entire current hypothesis. Captured terminal output
   made each cumulative update look like a repeated transcript. Partials are
   now throttled and rendered by replacing one terminal line.

6. **Finalization consumed only one pending encoder shift.**
   EOF could leave trailing chunks undrained. Finalization now drains all
   remaining shifts, improving last-word completion.

## Important design rule

The model learns **what was spoken**:

```text
Audio: twenty forty three
Training label: twenty forty three
Runtime display: 2043
```

Do not train this pair:

```text
Audio: twenty forty three
Training label: 2043
```

With only seven calls, arbitrary number generalization cannot safely come from
fine-tuning. The deterministic contextual normalizer handles display forms,
while fine-tuning focuses on acoustic/domain phrases such as `Inspira`.

## Safe post-processing behavior

The post-processor changes a domain entity only when an exact configured
variant and its required context match. For example:

```text
Thank you for calling Inspire Financial
-> Thank you for calling Inspira Financial
```

It does not globally replace the ordinary word `inspire`:

```text
I want to inspire financial confidence
-> unchanged
```

Number conversion is restricted to a numeric response or a strong context such
as member ID, SSN, phone ending, ticket number, or verification code. Ordinary
phrases remain unchanged:

```text
about two days ago            -> unchanged
one to three business days    -> unchanged
```

Supported required forms:

```text
twenty thirty four                  -> 2034
twelve forty three                  -> 1243
fifteen thirty eight                -> 1538
double five                         -> 55
triple five double nine four zero   -> 5559940
two thousand thirty four            -> 2034
twelve hundred thirty four          -> 1234
```

The current Nemotron streaming call does not expose reliable token-level
acoustic confidence in this server path. Therefore the “confidence guard” here
is a conservative lexical/context guard. If the conditions are not strong, the
original transcript is returned unchanged. Every applied correction is also
included in the event as `raw_text` and `corrections` for auditing.

# A-to-Z commands

## 1. EC2 prerequisites

Use an NVIDIA GPU instance with a recent driver and Docker GPU support.

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If the third command fails, install NVIDIA Container Toolkit before continuing.

## 2. Enter the project

```bash
cd /path/to/nemotron_finetuned_fixed
```

Place these files under `raw_wavs/`:

```text
withdraw_money.wav
Card_lost.wav
Card_Delivery_Status.wav
COBRA_coverage.wav
Profile_Update.wav
bank_issue.wav
Verification_Code_Issue.wav
```

Verify:

```bash
find raw_wavs -maxdepth 1 -type f -name '*.wav' -printf '%f\n' | sort
```

## 3. Build the image

Normal EC2 build without a proxy:

```bash
docker build --progress=plain -t nemotron_finetuned .
```

Corporate proxy build, only when required:

```bash
docker build --progress=plain \
  --build-arg HTTP_PROXY=http://163.116.128.80:8080 \
  --build-arg HTTPS_PROXY=http://163.116.128.80:8080 \
  -t nemotron_finetuned .
```

Verify the image:

```bash
docker images | grep nemotron_finetuned
```

## 4. Enter the training container

```bash
chmod +x 01_enter_training_container.sh \
  02_prepare_baseline_train_eval_inside_container.sh \
  03_run_finetuned_server.sh \
  04_run_base_server.sh \
  scripts/run_safe_finetuning.sh

./01_enter_training_container.sh
```

You are now inside the container. Verify GPU and model:

```bash
nvidia-smi
ls -lh /srv/nemotron-3.5-asr-streaming-0.6b.nemo
cd /workspace
```

## 5. Run the complete safe pipeline

Inside the container:

```bash
bash scripts/run_safe_finetuning.sh 2>&1 | tee logs/safe_finetuning.log
```

The script performs:

1. conversion to 16 kHz mono PCM;
2. base-ASR-assisted monotonic chunk alignment;
3. spoken-form numeric label reconstruction;
4. quality filtering and audit CSV generation;
5. whole-call train/validation/test splitting;
6. train-only speed, telephony, and low-noise augmentation;
7. baseline test evaluation;
8. decoder/joint-only fine-tuning with low learning rate;
9. validation early stopping and best-checkpoint restoration;
10. final held-out test evaluation;
11. a deployment gate that blocks raw-WER, semantic-WER, or `Inspira` recall regression;
12. promotion to `ft_models/finetuned_nemotron_final.nemo` only after the gate passes.

## 6. Check data quality before trusting the model

Still inside the container:

```bash
python3.11 - <<'PY'
import csv
from collections import Counter

with open('data/manifests/alignment_audit.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print('total:', len(rows))
print('accepted:', sum(r['accepted'].lower() == 'true' for r in rows))
print('rejections:', Counter(r['rejection_reasons'] for r in rows if r['accepted'].lower() != 'true'))
PY
```

Review rejected and borderline rows:

```bash
python3.11 - <<'PY'
import csv
with open('data/manifests/alignment_audit.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['accepted'].lower() != 'true' or float(row['match_score']) < 0.85:
            print('\n', row['chunk'])
            print('score:', row['match_score'], 'reason:', row['rejection_reasons'])
            print('draft:', row['draft_asr'])
            print('label:', row['training_text'])
PY
```

Do not manually force a bad row into training. Correct the source transcript or
add a manual segment annotation instead.

## 7. Compare baseline and fine-tuned outputs

```bash
cat results/safe_training/base_test.jsonl
cat results/safe_training/finetuned_test.jsonl
cat results/safe_training/deployment_gate.json
```

Useful summary:

```bash
python3.11 - <<'PY'
import json
from pathlib import Path

for name in ['base_test.jsonl', 'finetuned_test.jsonl']:
    rows = [json.loads(x) for x in (Path('results/safe_training') / name).read_text().splitlines() if x.strip()]
    print(name)
    print('  raw WER:', sum(x['wer'] for x in rows) / len(rows))
    print('  semantic WER:', sum(x['semantic_wer'] for x in rows) / len(rows))
PY
```

Exit the training container:

```bash
exit
```

Verify the model on the host:

```bash
ls -lh ft_models/finetuned_nemotron_final.nemo
cat ft_models/finetuned_nemotron_final.training_summary.json
```

## 8. Run unit tests for number and entity rules

On the host or inside the container after installing pytest:

```bash
python3 -m pip install pytest
pytest -q tests/test_postprocessing.py
```

Expected result:

```text
3 passed
```

## 9. Run the fine-tuned realtime server

```bash
mkdir -p audio_logs
./03_run_finetuned_server.sh
```

In another terminal:

```bash
curl http://localhost:8003/health
```

## 10. Test the exact WAV in realtime mode

```bash
python3 -m venv .client-venv
source .client-venv/bin/activate
pip install -r requirements-client.txt

python client.py \
  --file raw_wavs/bank_issue.wav \
  --language en-US \
  --realtime \
  --url ws://localhost:8003/asr/realtime-custom-vad
```

For the withdrawal file:

```bash
python client.py \
  --file raw_wavs/withdraw_money.wav \
  --language en-US \
  --realtime \
  --url ws://localhost:8003/asr/realtime-custom-vad
```

## 11. Microphone test

On a computer with an input device and PortAudio:

Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
source .client-venv/bin/activate
pip install -r requirements-client.txt
python client.py --mic --language en-US \
  --url ws://EC2_PUBLIC_IP:8003/asr/realtime-custom-vad
```

macOS:

```bash
brew install portaudio
source .client-venv/bin/activate
pip install -r requirements-client.txt
python client.py --mic --language en-US \
  --url ws://EC2_PUBLIC_IP:8003/asr/realtime-custom-vad
```

Open TCP port 8003 only to the required client IP, or use an SSH tunnel:

```bash
ssh -L 8003:localhost:8003 ubuntu@EC2_PUBLIC_IP
```

Then use `ws://localhost:8003/...` locally.

## 12. Compare the base model using the same corrected runtime

Stop the fine-tuned server, then run:

```bash
./04_run_base_server.sh
```

Test the same audio and compare the saved sessions under `audio_logs/`.

## Recommended data expansion

Seven calls are enough for a POC, not a production domain adaptation. Add:

- at least 50–100 manually verified utterances containing `Inspira` from
  different speakers and accents;
- separate number audio covering pair-style, digit-by-digit, hundred,
  thousand, repeated digits, leading zeroes, and alphanumeric tickets;
- real 8 kHz telephony captures, codec variation, and realistic background
  noise;
- labels that exactly match the spoken form;
- whole-speaker or whole-call holdout splits.

Do not generate new numeric labels by simply changing the transcript while
reusing the old audio. Audio and text must always say the same thing.
