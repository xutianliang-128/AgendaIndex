#!/usr/bin/env python3
"""Compare Stage-2 classifiers on identical windows.

Each system is a {record id: score} JSON (0/1 labels or p1 probabilities). A
remark in the PC window predicted 1 counts as PC, in the PH window as PH;
everything outside the windows is 0. Reports:
  - end_to_end: whole meeting vs gold (structure misses count as FN)
  - within_window: only utterances inside the predicted window
for all test cities, the IC2S2 subset, and per city.

Probabilistic systems are thresholded at 0.5 and at the F1-optimal threshold
found on the val split (pass --val-scores with matching names).

Usage:
  python sft/evaluate.py --data-dir results/sft_data \
      --system qwen_zero=results/sft_eval/qwen_zero_test.json \
      --system qwen_sft=results/sft_eval/qwen_sft_test.json \
      --system gpt41mini=results/sft_eval/gpt41mini_test.json \
      --val-scores qwen_zero=results/sft_eval/qwen_zero_val.json \
      --val-scores qwen_sft=results/sft_eval/qwen_sft_val.json \
      --out results/sft_eval/report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

IC2S2 = ["SEA", "OAK", "RCH", "AA", "LS", "RO", "JS", "AP", "PE", "IN"]
SECTIONS = (("pc", "Public Comment"), ("ph", "Public Hearing"))


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def counts(info: dict, scores: dict, thr: float, cities=None):
    """{scope: {kind: [tp, fp, fn]}} for scope in end_to_end / within_window."""
    c = {s: {k: [0, 0, 0] for k, _ in SECTIONS} for s in ("end_to_end", "within_window")}
    for mk, m in info.items():
        if cities is not None and m["city"] not in cities:
            continue
        for kind, section in SECTIONS:
            gold = m[f"gt_{kind}"]
            win = set(m[f"win_{kind}"])
            pred = [0] * m["n"]
            for i in win:
                pred[i] = int(scores.get(f"{mk}|{section}|{i}", 0) >= thr)
            for i in range(m["n"]):
                g, p = gold[i], pred[i]
                for scope in ("end_to_end", "within_window"):
                    if scope == "within_window" and i not in win:
                        continue
                    t = c[scope][kind]
                    t[0] += g and p
                    t[1] += (not g) and p
                    t[2] += g and (not p)
    return c


def report(info, scores, thr, cities=None):
    c = counts(info, scores, thr, cities)
    return {s: {k: prf(*v) for k, v in d.items()} for s, d in c.items()}


def best_threshold(info, scores) -> float:
    """Threshold maximising mean end-to-end F1 of PC and PH."""
    best, best_f = 0.5, -1.0
    for t in [x / 100 for x in range(5, 96, 5)]:
        r = report(info, scores, t)["end_to_end"]
        f = (r["pc"]["f1"] + r["ph"]["f1"]) / 2
        if f > best_f:
            best, best_f = t, f
    return best


def parse_pairs(items):
    return dict(x.split("=", 1) for x in items or [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--system", action="append", required=True)
    ap.add_argument("--val-scores", action="append")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    info = json.loads((args.data_dir / f"{args.split}_meetings.json").read_text())
    val_info = json.loads((args.data_dir / "val_meetings.json").read_text())
    systems = {k: json.loads(Path(v).read_text()) for k, v in parse_pairs(args.system).items()}
    vals = {k: json.loads(Path(v).read_text()) for k, v in parse_pairs(args.val_scores).items()}

    cities_present = sorted({m["city"] for m in info.values()})
    ic2s2 = [c for c in IC2S2 if c in cities_present]
    runs = {}
    for name, sc in systems.items():
        runs[f"{name}@0.5"] = (sc, 0.5)
        if name in vals:
            t = best_threshold(val_info, vals[name])
            runs[f"{name}@val{t:.2f}"] = (sc, t)

    out = {"ic2s2_cities_in_split": ic2s2, "n_meetings": len(info),
           "n_meetings_ic2s2": sum(m["city"] in ic2s2 for m in info.values()), "runs": {}}
    for run, (sc, t) in runs.items():
        out["runs"][run] = {
            "threshold": t,
            "all": report(info, sc, t),
            "ic2s2": report(info, sc, t, set(ic2s2)),
            "per_city": {c: report(info, sc, t, {c})["end_to_end"] for c in cities_present},
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    def row(label, r):
        e, w = r["end_to_end"], r["within_window"]
        return (f"{label:<26} PC {e['pc']['p']:.3f}/{e['pc']['r']:.3f}/{e['pc']['f1']:.3f}  "
                f"PH {e['ph']['p']:.3f}/{e['ph']['r']:.3f}/{e['ph']['f1']:.3f}   "
                f"| win PC {w['pc']['f1']:.3f} PH {w['ph']['f1']:.3f}")

    for scope in ("all", "ic2s2"):
        n = out["n_meetings"] if scope == "all" else out["n_meetings_ic2s2"]
        print(f"\n=== {scope} ({n} meetings)   end-to-end P/R/F1 ===")
        for run, d in out["runs"].items():
            print(row(run, d[scope]))
    print("\n=== per-city end-to-end F1 (PC / PH) ===")
    names = list(out["runs"])
    print(f"{'city':<6}" + "".join(f"{n[:22]:>24}" for n in names))
    for c in cities_present:
        cells = []
        for n in names:
            e = out["runs"][n]["per_city"][c]
            f = [f"{e[k]['f1']:.2f}" if e[k]["tp"] + e[k]["fp"] + e[k]["fn"] else "  - "
                 for k in ("pc", "ph")]
            cells.append(f"{f[0]} / {f[1]}")
        print(f"{c:<6}" + "".join(f"{x:>24}" for x in cells))


if __name__ == "__main__":
    main()
