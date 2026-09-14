#!/usr/bin/env python3
"""SFT LoRA de Qwen3-1.7B avec Unsloth.

Usage :
    .venv-train/bin/python scripts/train_sft.py --pilot                 # pilot run (200 vignettes, 1 epoch)
    .venv-train/bin/python scripts/train_sft.py --epochs 2 --merge      # run complet + fusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from unsloth import FastModel  # isort: skip  (avant trl/transformers)

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        records.extend(json.loads(line) for line in p.open(encoding="utf-8"))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT LoRA Qwen3-1.7B (Unsloth)")
    ap.add_argument("--pilot", action="store_true", help="Petit run de validation")
    ap.add_argument("--data", choices=["all", "base", "triage"], default="all",
                    help="Données : all (base+triage×3) / base / triage")
    ap.add_argument("--resume-from", default="", help="Reprendre depuis un adaptateur LoRA")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--merge", action="store_true", help="Fusionner le LoRA après entraînement")
    ap.add_argument("--output-dir", default="models/lora_pilot")
    args = ap.parse_args()

    # --- Données ---
    if args.pilot:
        records = load_records([PROCESSED_DIR / "triage" / "sft_vignettes.jsonl"])[:200]
    elif args.data == "base":
        records = load_records([PROCESSED_DIR / "final" / "sft_train.jsonl"])
    elif args.data == "triage":
        records = load_records([
            PROCESSED_DIR / "triage" / "sft_vignettes.jsonl",
            PROCESSED_DIR / "triage" / "sft_dialogues_dressed.jsonl",
        ])
    else:  # all
        base = load_records([PROCESSED_DIR / "final" / "sft_train.jsonl"])
        triage = load_records([
            PROCESSED_DIR / "triage" / "sft_vignettes.jsonl",
            PROCESSED_DIR / "triage" / "sft_dialogues_dressed.jsonl",
        ])
        records = base + triage * 3

    print(f"{len(records)} exemples d'entraînement")

    # --- Modèle + LoRA ---
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.resume_from if args.resume_from else "Qwen/Qwen3-1.7B",
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    if not args.resume_from:
        model = FastModel.get_peft_model(
            model,
            r=args.r,
            lora_alpha=args.r,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    texts = [
        tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        for r in records
    ]
    dataset = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=args.output_dir,
            max_length=args.max_seq_length,
            dataset_text_field="text",
            per_device_train_batch_size=2,
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

    # --- Sauvegarde ---
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter LoRA sauvegardé dans {args.output_dir}")

    if args.merge:
        merged_dir = args.output_dir + "_merged"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"Modèle fusionné sauvegardé dans {merged_dir}")


if __name__ == "__main__":
    main()
