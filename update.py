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
