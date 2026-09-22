#!/usr/bin/env python3
"""DPO LoRA (Unsloth) de Qwen3-1.7B sur paires de préférence triage.

Usage :
    ./scripts/train.sh scripts/train_dpo.py --pilot ...
    # ou via train_dpo.sh (wrapper dédié)

Format des paires (TRL conversationnel) :
    {"prompt": [system, user], "chosen": [assistant], "rejected": [assistant]}
"""
# ruff: noqa: E402, I001  (imports Unsloth/TRL après le patch ci-dessous)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# --- Fix TRL 0.24 + transformers v5 ---
# _is_package_available renvoie désormais un tuple (bool, version) ; TRL 0.24 stocke ce
# tuple et `is_*_available()` le renvoie (truthy même si False). On remet des booléens.
import trl.import_utils as _trl_iu

for _v in dir(_trl_iu):
    if _v.endswith("_available") and not _v.startswith("is_"):
        _val = getattr(_trl_iu, _v)
        if isinstance(_val, tuple):
            setattr(_trl_iu, _v, bool(_val[0]))

from unsloth import FastModel, PatchDPOTrainer  # isort: skip  # noqa: E402

PatchDPOTrainer()

from datasets import Dataset  # noqa: E402
from trl import DPOTrainer, DPOConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO LoRA Qwen3-1.7B (Unsloth)")
    ap.add_argument("--data", default=str(PROCESSED_DIR / "triage" / "dpo_pairs.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--resume-from", default="", help="Adapter LoRA SFT (ex. models/lora_stage2_v2)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--bf16", action="store_true", help="LoRA bf16 (requis Qwen3.5)")
    ap.add_argument("--target-all-linear", action="store_true", help="LoRA all-linear (DeltaNet)")
    ap.add_argument("--output-dir", default="models/lora_dpo")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="Sous-échantillon 20 paires (validation)")
    args = ap.parse_args()

    pairs = [json.loads(line) for line in Path(args.data).open(encoding="utf-8")]
    if args.pilot:
        pairs = pairs[:20]
    print(f"{len(pairs)} paires DPO")

    load_kwargs: dict = {"max_seq_length": args.max_seq_length}
    if args.bf16:
        load_kwargs["load_in_4bit"] = False
        load_kwargs["load_in_16bit"] = True
    else:
        load_kwargs["load_in_4bit"] = True
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.resume_from if args.resume_from else args.model,
        **load_kwargs,
    )
    if not args.resume_from:
        target_modules = "all-linear" if args.target_all_linear else [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ]
        model = FastModel.get_peft_model(
            model,
            r=args.r,
            lora_alpha=args.r,
            lora_dropout=0,
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    dataset = Dataset.from_list(pairs)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(
            output_dir=args.output_dir,
            beta=args.beta,
            max_length=args.max_seq_length,
            max_prompt_length=1024,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_steps=5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter DPO sauvegardé dans {args.output_dir}")

    if args.merge:
        merged_dir = args.output_dir + "_merged"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"Modèle fusionné sauvegardé dans {merged_dir}")


if __name__ == "__main__":
    main()
