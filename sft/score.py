#!/usr/bin/env python3
"""Score SFT records with Qwen3 (base or +LoRA): p(answer = "1") per record.

Reads the next-token distribution right after the generation prompt and
normalises over the "0"/"1" tokens, so no decoding is needed. Shards across
GPUs with torchrun; rank 0 merges shards into one {record id: p1} JSON.

Usage:
  torchrun --nproc_per_node 3 sft/score.py --data results/sft_data/test.jsonl \
      [--adapter results/sft_qwen4b/adapter] --out results/sft_eval/qwen_sft_test.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1536)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    id0 = tok("0", add_special_tokens=False).input_ids[0]
    id1 = tok("1", add_special_tokens=False).input_ids[0]

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model.eval()

    recs = [json.loads(l) for l in args.data.open(encoding="utf-8")]
    mine = recs[rank::world]
    texts = [
        tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        for r in mine
    ]
    order = sorted(range(len(texts)), key=lambda k: len(texts[k]))

    out: dict[str, float] = {}
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), args.batch):
            ks = order[b:b + args.batch]
            enc = tok([texts[k] for k in ks], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_len, add_special_tokens=False)
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[:, -1, :]
            p1 = torch.softmax(logits[:, [id0, id1]].float(), dim=-1)[:, 1].tolist()
            for k, p in zip(ks, p1):
                out[mine[k]["id"]] = p
            if rank == 0 and (b // args.batch) % 50 == 0:
                print(f"  rank0 {b}/{len(order)}  {time.time() - t0:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shard = args.out.with_suffix(f".rank{rank}.json")
    tmp = shard.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.rename(shard)
    if rank == 0:
        shards = [args.out.with_suffix(f".rank{r}.json") for r in range(world)]
        while not all(sp.exists() for sp in shards):
            time.sleep(2)
        merged: dict[str, float] = {}
        for sp in shards:
            merged.update(json.loads(sp.read_text()))
            sp.unlink()
        args.out.write_text(json.dumps(merged))
        print(f"wrote {len(merged)} scores to {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    main()
