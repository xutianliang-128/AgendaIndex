#!/usr/bin/env python3
"""Build SFT / evaluation records from annotated splits and cached structures.

One record per (section, utterance) inside a predicted public-speaking window,
using the same windows the pipeline itself scores. The training label is
"spoken by a member of the public" (PC or PH gold), because which bucket the
remark lands in is decided by the window, not by the classifier.

Also writes <split>_meetings.json with the full-meeting gold and the window
index sets, which evaluation needs for end-to-end scoring.

Usage:
  python sft/build_windows.py --splits-dir DATA/all_splits \
      --structure-dir CACHE/structures --out-dir results/sft_data
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agendaindex.section_taxonomy import classify_stage  # noqa: E402
from sft.prompt import build_prompt  # noqa: E402

SPLITS = ("train", "val", "test")
KIND_TO_SECTION = {"pc": "Public Comment", "ph": "Public Hearing"}


def is_one(v) -> bool:
    return v in (1, "1", True)


def windows(stages: list, n_utts: int) -> dict[str, dict[int, str]]:
    """section -> {utterance index -> heading of the stage that covers it}."""
    out: dict[str, dict[int, str]] = {"Public Comment": {}, "Public Hearing": {}}
    for s in stages or []:
        kind = classify_stage(s.get("stage_name"))
        if kind is None:
            continue
        try:
            lo, hi = int(s.get("start_index", 0)), int(s.get("end_index", 0))
        except (TypeError, ValueError):
            continue
        lo, hi = max(0, min(lo, hi)), min(n_utts - 1, max(lo, hi))
        for i in range(lo, hi + 1):
            out[KIND_TO_SECTION[kind]].setdefault(i, s.get("stage_name") or "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--structure-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ctx", type=int, default=5)
    args = ap.parse_args()

    structs: dict[str, list] = {}
    for sf in sorted(args.structure_dir.glob("*_meeting_structure.json")):
        structs.update(json.loads(sf.read_text(encoding="utf-8")))

    meetings: dict[str, tuple[str, list]] = {}
    for split in SPLITS:
        for fp in sorted(args.splits_dir.glob(f"*_{split}.json")):
            for mk, utts in json.loads(fp.read_text(encoding="utf-8")).items():
                meetings.setdefault(mk, (split, utts))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        recs, mtg_info, stats = [], {}, Counter()
        for mk, (sp, utts) in sorted(meetings.items()):
            if sp != split:
                continue
            if mk not in structs:
                stats["meetings_without_structure"] += 1
                continue
            city = mk.split("_")[0]
            win = windows(structs[mk], len(utts))
            gt_pc = [int(is_one(u.get("is_public_comment_gt"))) for u in utts]
            gt_ph = [int(is_one(u.get("is_public_hearing_gt"))) for u in utts]
            mtg_info[mk] = {
                "city": city,
                "n": len(utts),
                "gt_pc": gt_pc,
                "gt_ph": gt_ph,
                "win_pc": sorted(win["Public Comment"]),
                "win_ph": sorted(win["Public Hearing"]),
            }
            for section, idx_map in win.items():
                for i, stage in sorted(idx_map.items()):
                    label = int(gt_pc[i] or gt_ph[i])
                    recs.append({
                        "id": f"{mk}|{section}|{i}",
                        "city": city,
                        "meeting": mk,
                        "section": section,
                        "idx": i,
                        "label": label,
                        "prompt": build_prompt(utts, i, section, stage, ctx=args.ctx),
                    })
                    stats["records"] += 1
                    stats["positives"] += label
            stats["meetings"] += 1
        with (args.out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (args.out_dir / f"{split}_meetings.json").write_text(json.dumps(mtg_info), encoding="utf-8")
        print(split, dict(stats))


if __name__ == "__main__":
    main()
