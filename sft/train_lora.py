#!/usr/bin/env python3
"""LoRA fine-tune Qwen3-4B as the Stage-2 remark classifier.

The model is trained to emit a single "1"/"0" right after the chat template's
generation prompt (thinking disabled), with loss only on that answer. That is
exactly the position sft/score.py reads at inference time.

Usage (3 GPUs):
  torchrun --nproc_per_node 3 sft/train_lora.py \
      --data-dir results/sft_data --output-dir results/sft_qwen4b
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

ANSWER = {0: "0", 1: "1"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4, help="per device")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    end_ids = tok("<|im_end|>", add_special_tokens=False).input_ids

    def encode(rec):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": rec["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        p = tok(prompt, add_special_tokens=False).input_ids
        a = tok(ANSWER[int(rec["label"])], add_special_tokens=False).input_ids + end_ids
        p = p[-(args.max_len - len(a)):]
        return {"input_ids": p + a, "labels": [-100] * len(p) + a}

    ds = load_dataset("json", data_files={
        "train": str(args.data_dir / "train.jsonl"),
        "val": str(args.data_dir / "val.jsonl"),
    })
    cols = ds["train"].column_names
    ds = ds.map(encode, remove_columns=cols, num_proc=8)

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for k, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[k, :L] = torch.tensor(b["input_ids"])
            lab[k, :L] = torch.tensor(b["labels"])
            att[k, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))

    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        group_by_length=True,
        report_to=[],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"],
                      eval_dataset=ds["val"], data_collator=collate)
    result = trainer.train()
    eval_metrics = trainer.evaluate()   # collective under DDP: every rank must call it

    if trainer.is_world_process_zero():
        adapter = args.output_dir / "adapter"
        trainer.model.save_pretrained(str(adapter))
        tok.save_pretrained(str(adapter))
        summary = {
            "train": result.metrics,
            "eval": eval_metrics,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "args": {k: str(v) for k, v in vars(args).items()},
            "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        }
        (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    main()
