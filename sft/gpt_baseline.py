#!/usr/bin/env python3
"""Run the production Stage-2 classifier (gpt-4.1-mini, per-line) on the SFT windows.

Uses exactly the windows in <split>_meetings.json, so its predictions line up
one-to-one with the Qwen scores. Writes {record id: 0/1} as JSON.

Usage:
  python scripts/_run_with_env.py sft/gpt_baseline.py \
      --data-dir results/sft_data --split test --out results/sft_eval/gpt41mini_test.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import infer_ic2s2_best as best  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--chunk-ctx", type=int, default=5)
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()

    info = json.loads((args.data_dir / f"{args.split}_meetings.json").read_text())
    utts_by_mk: dict[str, list] = {}
    for fp in sorted(args.splits_dir.glob(f"*_{args.split}.json")):
        for mk, utts in json.loads(fp.read_text(encoding="utf-8")).items():
            if mk in info:
                utts_by_mk.setdefault(mk, [best.sanitize(u) for u in utts])

    units = []
    for mk, m in info.items():
        utts = utts_by_mk[mk]
        for section, key in (("Public Comment", "win_pc"), ("Public Hearing", "win_ph")):
            for run in best.contiguous_runs(m[key]):
                for block in best.chunk_run(run, args.chunk):
                    lo, hi = block[0], block[-1]
                    ctx = [j for j in range(max(0, lo - args.chunk_ctx),
                                            min(len(utts), hi + args.chunk_ctx + 1))
                           if j not in block]
                    units.append((section, mk, utts, block, ctx))

    def run_unit(unit):
        section, mk, utts, block, ctx = unit
        out = best.call(best.prompt_c_perline(best.SECTION_CFG[section], utts, block, ctx),
                        args.model)
        got = {int(a): int(b) for a, b in re.findall(r"(\d+)\s*:\s*([01])", out)}
        best.COVERAGE[0] += len(block)
        best.COVERAGE[1] += sum(1 for i in block if i in got)
        return {f"{mk}|{section}|{i}": got.get(i, 0) for i in block}

    preds: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for d in pool.map(run_unit, units):
            preds.update(d)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preds))
    pin, pout, calls = best.USAGE
    rate_in, rate_out = best.PRICE.get(args.model, (0, 0))
    print(f"{len(units)} calls, {len(preds)} labels, coverage "
          f"{best.COVERAGE[1]}/{best.COVERAGE[0]}, "
          f"cost ${(pin * rate_in + pout * rate_out) / 1e6:.2f}")


if __name__ == "__main__":
    main()
