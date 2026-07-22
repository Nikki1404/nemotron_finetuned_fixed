docker rm -f nemotron-trainer 2>/dev/null || true; docker run --rm --name nemotron-trainer --gpus all --ipc=host -e PYTHONPATH=/workspace -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TOKENIZERS_PARALLELISM=false -v "$PWD:/workspace" -v "$PWD/ft_models:/srv/models" -w /workspace nemotron_finetuned_3.5 bash -lc 'set -o pipefail; bash scripts/run_safe_finetuning.sh 2>&1 | tee logs/safe_finetuning.log'
sed -i '/^cd \/workspace$/a export PYTHONPATH=/workspace:${PYTHONPATH:-}' scripts/run_safe_finetuning.sh
head -n 10 scripts/run_safe_finetuning.sh
docker run --rm --gpus all -e PYTHONPATH=/workspace -v "$PWD:/workspace" -w /workspace nemotron_finetuned_3.5 python3.11 -c 'from app.asr_number_normalizer import NUMBER_WORDS,parse_number_phrase; from app.transcript_postprocessor import DomainEntityCorrector; print("App imports successful")'
docker run --rm -e PYTHONPATH=/workspace -v "$PWD:/workspace" -w /workspace nemotron_finetuned_3.5 bash -lc 'pwd; ls -la app; ls -la scripts; test -f app/__init__.py && echo "app package exists"'
mkdir -p ft_models logs results/safe_training data/manifests data/audio_16k data/audio_chunks data/audio_aug
docker rm -f nemotron-trainer 2>/dev/null || true; docker run --rm --name nemotron-trainer --gpus all --ipc=host -e PYTHONPATH=/workspace -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TOKENIZERS_PARALLELISM=false -v "$PWD:/workspace" -v "$PWD/ft_models:/srv/models" -w /workspace nemotron_finetuned_3.5 bash -lc 'set -o pipefail; bash scripts/run_safe_finetuning.sh 2>&1 | tee logs/safe_finetuning.log'
tail -f logs/safe_finetuning.log

docker rm -f nemotron-trainer 2>/dev/null || true; docker run --rm --name nemotron-trainer --gpus all --ipc=host -e PYTHONPATH=/workspace -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TOKENIZERS_PARALLELISM=false -e NUMBA_CACHE_DIR=/tmp/numba_cache -v "$PWD:/workspace" -v "$PWD/ft_models:/srv/models" -w /workspace nemotron_finetuned_3.5 bash -lc 'set -euo pipefail; apt-get update; apt-get install -y --no-install-recommends cuda-nvvm-12-4; rm -rf /var/lib/apt/lists/*; python3.11 -m pip uninstall -y numba-cuda numba llvmlite || true; python3.11 -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4" "llvmlite==0.43.0" "numba==0.60.0"; rm -rf /root/.cache/numba /tmp/numba_cache; export CUDA_HOME=/usr/local/cuda; NVVM="$(find /usr/local -path "*/nvvm/lib64/libnvvm.so" -print -quit)"; test -n "$NVVM" || { echo "ERROR: libnvvm.so missing"; exit 1; }; export LD_LIBRARY_PATH="$(dirname "$NVVM"):/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"; python3.11 -c "import ctypes,numba,llvmlite,numpy; from numba import cuda; ctypes.CDLL(\"$NVVM\"); print(\"NumPy:\",numpy.__version__); print(\"Numba:\",numba.__version__); print(\"llvmlite:\",llvmlite.__version__); print(\"Numba CUDA:\",cuda.__file__); print(\"CUDA available:\",cuda.is_available()); print(\"libNVVM:\",\"$NVVM\")"; set -o pipefail; bash scripts/run_safe_finetuning.sh 2>&1 | tee logs/safe_finetuning.log'

getting this 
                                                                      [NeMo I 2026-07-22 03:13:00 wer:318]
    dation DataLoader 0:  71%|███████▏  | 5/7 [00:02<00:01,  1.77it/s]
[NeMo I 2026-07-22 03:13:00 wer:319] WER reference:ending in forty six seventy eight please check your messages and try again okay i will check is there anything else i can help you with
[NeMo I 2026-07-22 03:13:00 wer:320] WER predicted:Ending in forty six seventy eight, please check your messages and try again. <en-US> Okay, I will check. <en-US> Is there anything else I can help you with
                                                                      [NeMo I 2026-07-22 03:13:00 wer:318]
    dation DataLoader 0:  86%|████████▌ | 6/7 [00:02<00:00,  2.03it/s]
[NeMo I 2026-07-22 03:13:00 wer:319] WER reference:no thanks thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 03:13:00 wer:320] WER predicted:No thanks. <en-US> Thank you for calling Inspira Financial. <en-US> Have a nice day. <en-US>
                                                                      [NeMo I 2026-07-22 03:13:00 asr_model:198] CUDA graphs disabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer███| 7/7 [00:03<00:00,  2.26it/s]
Epoch 0: 100%|██████████| 260/260 [01:25<00:00,  3.06it/s, v_num=2][NeMo I 2026-07-22 03:13:00 asr_model:185] CUDA graphs enabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer
Traceback (most recent call last):
  File "/workspace/scripts/finetune_nemotron.py", line 250, in <module>
    main()
  File "/workspace/scripts/finetune_nemotron.py", line 213, in main
    trainer.fit(model)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 538, in fit
    call._call_and_handle_interrupt(
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 47, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 574, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 981, in _run
    results = self._run_stage()
              ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1025, in _run_stage
    self.fit_loop.run()
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/fit_loop.py", line 206, in run
    self.on_advance_end()
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/fit_loop.py", line 378, in on_advance_end
    call._call_callback_hooks(trainer, "on_train_epoch_end", monitoring_callbacks=True)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 218, in _call_callback_hooks
    fn(trainer, trainer.lightning_module, *args, **kwargs)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/callbacks/early_stopping.py", line 190, in on_train_epoch_end
    self._run_early_stopping_check(trainer)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/callbacks/early_stopping.py", line 202, in _run_early_stopping_check
    if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/callbacks/early_stopping.py", line 153, in _validate_condition_metric
    raise RuntimeError(error_msg)
RuntimeError: Early stopping conditioned on metric `val_loss` which is not available. Pass in or modify your `EarlyStopping` callback to use any of the following: `train_loss`, `learning_rate`, `global_step`, `training_batch_wer`, `val_wer`
Epoch 0: 100%|██████████| 260/260 [01:26<00:00,  3.00it/s, v_num=2]
