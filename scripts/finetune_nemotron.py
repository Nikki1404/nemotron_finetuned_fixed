#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import types
from pathlib import Path
from typing import Any


def load_model(model_path: str, language: str):
    import nemo.collections.asr as nemo_asr

    cls = nemo_asr.models.EncDecRNNTBPEModelWithPrompt
    if model_path.endswith(".nemo") or Path(model_path).exists():
        model = cls.restore_from(model_path, map_location="cpu")
    else:
        model = cls.from_pretrained(model_path, map_location="cpu")
    for fn in ("set_inference_prompt", "set_default_prompt"):
        try:
            getattr(model, fn)(language)
        except Exception as exc:
            print(f"[warn] {fn}({language}) failed: {exc}")
    return model


def set_freeze_mode(model: Any, freeze_mode: str) -> None:
    if freeze_mode == "none":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    if freeze_mode == "decoder_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name, module in model.named_modules():
            if any(key in name.lower() for key in ("decoder", "joint", "prompt_kernel")):
                for parameter in module.parameters(recurse=True):
                    parameter.requires_grad = True
        return
    raise ValueError(f"Unknown freeze_mode: {freeze_mode}")


def count_trainable(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"[params] total={total:,} trainable={trainable:,} ({trainable / max(1, total) * 100:.2f}%)")
    return total, trainable


def get_prompt_index(model: Any, language: str) -> int:
    prompt_dict = model.cfg.train_ds.prompt_dictionary
    if language not in prompt_dict:
        raise ValueError(f"Language {language} not in prompt dictionary")
    return int(prompt_dict[language])


def patch_batch_prompt_indices(model: Any, prompt_index: int) -> None:
    old_train = model.training_step
    old_val = model.validation_step

    def add_prompt(batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            import torch

            signal, signal_len, transcript, transcript_len = batch
            prompt_indices = torch.full(
                (signal.shape[0],),
                prompt_index,
                dtype=torch.long,
                device=signal.device,
            )
            return signal, signal_len, transcript, transcript_len, prompt_indices
        return batch

    def new_training_step(self, batch, batch_idx):
        return old_train(add_prompt(batch), batch_idx)

    def new_validation_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            return old_val(add_prompt(batch), batch_idx, dataloader_idx)
        except TypeError:
            return old_val(add_prompt(batch), batch_idx)

    model.training_step = types.MethodType(new_training_step, model)
    model.validation_step = types.MethodType(new_validation_step, model)
    print(f"[patch] Added prompt_indices={prompt_index} when batch has four items")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-nemo", required=True)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--freeze-mode", default="decoder_only", choices=["decoder_only", "none"])
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from omegaconf import OmegaConf

    model = load_model(args.base_model, args.language)
    prompt_index = get_prompt_index(model, args.language)
    print(f"[language] {args.language} prompt_index={prompt_index}")
    set_freeze_mode(model, args.freeze_mode)
    total_params, trainable_params = count_trainable(model)
    patch_batch_prompt_indices(model, prompt_index)

    common = {
        "sample_rate": 16000,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "max_duration": args.max_duration,
        "min_duration": 0.5,
        "is_tarred": False,
        "use_lhotse": False,
    }
    train_cfg = OmegaConf.create(
        {
            **common,
            "manifest_filepath": str(Path(args.train_manifest).resolve()),
            "batch_size": args.batch_size,
            "shuffle": True,
        }
    )
    val_cfg = OmegaConf.create(
        {
            **common,
            "manifest_filepath": str(Path(args.val_manifest).resolve()),
            "batch_size": 1,
            "shuffle": False,
        }
    )
    model.setup_training_data(train_data_config=train_cfg)
    model.setup_validation_data(val_data_config=val_cfg)

    model.cfg.optim = OmegaConf.create(
        {
            "name": "adamw",
            "lr": args.lr,
            "betas": [0.9, 0.98],
            "weight_decay": 0.001,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": 20,
                "min_lr": args.lr / 20.0,
            },
        }
    )

    output = Path(args.output_nemo)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output.parent / f"{output.stem}_checkpoints"
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience,
        min_delta=0.001,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        max_epochs=args.max_epochs,
        precision=args.precision if torch.cuda.is_available() else "32-true",
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=1,
        callbacks=[checkpoint, early_stop],
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        deterministic=True,
    )
    model.set_trainer(trainer)
    print("[train] Starting conservative fine-tuning...")
    trainer.fit(model)

    best_path = checkpoint.best_model_path
    if best_path:
        print(f"[best] Restoring validation-best checkpoint: {best_path}")
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[warn] missing keys while restoring best checkpoint: {len(missing)}")
        if unexpected:
            print(f"[warn] unexpected keys while restoring best checkpoint: {len(unexpected)}")

    model.save_to(str(output))
    summary = {
        "base_model": args.base_model,
        "output_nemo": str(output),
        "train_manifest": str(Path(args.train_manifest).resolve()),
        "val_manifest": str(Path(args.val_manifest).resolve()),
        "freeze_mode": args.freeze_mode,
        "learning_rate": args.lr,
        "max_epochs": args.max_epochs,
        "best_checkpoint": best_path,
        "best_val_loss": float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "seed": args.seed,
    }
    summary_path = output.with_suffix(".training_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] Fine-tuned model saved to: {output}")
    print(f"[done] Training summary saved to: {summary_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
