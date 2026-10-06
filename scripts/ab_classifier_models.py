#!/usr/bin/env python3
"""A/B the per-utterance remark classifier across OpenAI models.

The structure (which utterances sit in a PC/PH window) is held fixed, so the
only thing that varies is the model answering the binary question. Scoring is
against the human annotation in all_splits, over the whole meeting: an
utterance counts as predicted-positive only when it is inside the window AND
the model calls it a public remark, which is exactly the production rule.

Usage:
  python scripts/_run_with_env.py scripts/ab_classifier_models.py --meetings 8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "all_splits_localgov_train_test_val" / "all_splits"
STRUCT_CACHE = ROOT / "results" / "all_splits_pred_cache" / "structures"

sys.path.insert(0, str(ROOT))
from pageindex.section_taxonomy import public_section_indices  # noqa: E402

# GPT-5 family rejects temperature != 1 on chat.completions and bills hidden
# reasoning tokens, so it needs different kwargs than the 4.x models.
REASONING_MODELS = ("gpt-5", "o1", "o3", "o4")

SECTION_CFG = {
    "Public Comment": {
        "label": "Public Comment",
        "desc": "an actual PUBLIC COMMENT (a citizen/member of the public addressing council)",
        "key": "is_public_comment",
        "gt": "is_public_comment_gt",
    },
    "Public Hearing": {
        "label": "Public Hearing",
        "desc": (
            "an actual PUBLIC REMARK during the hearing (a citizen/member of the public "
            "speaking on the hearing item)"
        ),
        "key": "is_public_remark",
        "gt": "is_public_hearing_gt",
    },
}


PROMPT_VARIANT = "v1"


def build_prompt(cfg, speaker, text, start, end) -> str:
    if PROMPT_VARIANT == "v1":
        return f"""You are a binary classifier. This utterance is from the "{cfg['label']}" section of a city council meeting. Decide if it is {cfg['desc']} or not (e.g. chair/moderator/staff saying "next speaker", "thank you", reading rules, or other procedural talk).

Use only the following information (no other fields):
- speaker_id: {speaker}
- text: {text}
- time: start={start} end={end} (seconds)

Return JSON: {{ "{cfg['key']}": 0 or 1 }}. 1 = citizen/member-of-public remark, 0 = not (e.g. moderator/chair/staff)."""

    # v2: the section window usually also contains the staff presentation and the
    # council's own deliberation on the item. Those dominate the false positives,
    # so they are called out explicitly instead of only procedural filler.
    return f"""You are a binary classifier. This utterance comes from the "{cfg['label']}" portion of a city council meeting. Decide if it is {cfg['desc']}.

Answer 1 ONLY for someone who is not part of the city government addressing the body: a resident, business owner, applicant, or other member of the public who has come forward to speak. Such a speaker typically states their name and address, speaks in one longer uninterrupted turn, and petitions or urges the council.

Answer 0 for everyone on the government's side of the dais, including:
- the mayor, chair, or any council member, whether running the meeting or debating the item
- city staff, the city manager, attorney, clerk, police or fire chief, or a consultant presenting or answering questions
- procedural talk: calling the next speaker, thanking a speaker, reading the rules, opening or closing the hearing

Note that the staff presentation and the council's back-and-forth deliberation on the item often fall inside this same section. Those are 0, even when the content sounds conversational or personal.

Use only the following information (no other fields):
- speaker_id: {speaker}
- text: {text}
- time: start={start} end={end} (seconds)

Return JSON: {{ "{cfg['key']}": 0 or 1 }}."""


client = openai.OpenAI()
USAGE: dict[str, list[int]] = {}


def ask(model: str, prompt: str, key: str) -> int:
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if not model.startswith(REASONING_MODELS):
        kwargs["temperature"] = 0
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(**kwargs)
            u = resp.usage
            USAGE.setdefault(model, [0, 0, 0])
            USAGE[model][0] += u.prompt_tokens
            USAGE[model][1] += u.completion_tokens
            USAGE[model][2] += 1
            content = resp.choices[0].message.content or ""
            m = re.search(r'"' + re.escape(key) + r'"\s*:\s*(\d)', content)
            if m:
                return int(m.group(1))
            m = re.search(r"\b([01])\b", content)
            return int(m.group(1)) if m else 0
        except Exception as exc:  # noqa: BLE001
            if attempt == 5:
                print(f"    [{model}] giving up: {exc}", file=sys.stderr)
                return 0
            time.sleep(1.5 * (attempt + 1))
    return 0


def load_meetings(limit: int, per_city: int = 3, min_gt: int = 10) -> list[tuple[str, str, list]]:
    """Annotated meetings with GT positives, spread across as many cities as possible.

    Round-robin over cities so no single city's speaking style dominates the score.
    """
    by_city: dict[str, list] = {}
    for path in sorted(SPLITS.glob("*.json")):
        city = path.stem.split("_")[0]
        struct_file = STRUCT_CACHE / f"{city}_meeting_structure.json"
        if not struct_file.is_file():
            continue
        structures = json.loads(struct_file.read_text())
        for mk, utts in json.loads(path.read_text()).items():
            if mk not in structures:
                continue
            pc = sum(1 for u in utts if u.get("is_public_comment_gt") in (1, "1", True))
            ph = sum(1 for u in utts if u.get("is_public_hearing_gt") in (1, "1", True))
            if pc + ph >= min_gt:
                by_city.setdefault(city, []).append((city, mk, utts, ph))

    # Prefer meetings that carry Public Hearing labels; PH is the scarce signal.
    for city in by_city:
        by_city[city].sort(key=lambda x: -x[3])

    # The same meeting appears in both <CITY>_all.json and <CITY>_test.json.
    for city, items in by_city.items():
        seen, uniq = set(), []
        for it in items:
            if it[1] not in seen:
                seen.add(it[1])
                uniq.append(it)
        by_city[city] = uniq

    picked = []
    for rnd in range(per_city):
        for city in sorted(by_city):
            if rnd < len(by_city[city]) and len(picked) < limit:
                c, mk, utts, _ = by_city[city][rnd]
                picked.append((c, mk, utts))
    return picked


def metrics(y_true, y_pred) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    n = len(y_true)
    return {
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "support": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", type=int, default=8)
    ap.add_argument("--cities", nargs="*", default=None, help="Restrict to these city codes")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"])
    ap.add_argument("--all", action="store_true", help="Every annotated meeting, not a per-city sample")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--models", nargs="*", default=[
        "gpt-4o-2024-11-20", "gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1-nano", "gpt-5-nano",
    ])
    ap.add_argument("--out", default=str(ROOT / "results" / "ab_classifier_models.json"))
    args = ap.parse_args()

    globals()["PROMPT_VARIANT"] = args.prompt
    print(f"prompt variant: {args.prompt}")
    meetings = (load_meetings(10**9, per_city=10**6, min_gt=0) if args.all
                else load_meetings(args.meetings))
    if args.cities:
        meetings = [m for m in meetings if m[0] in set(args.cities)]
    print(f"A/B set: {len(meetings)} meetings from {len({c for c,_,_ in meetings})} cities")

    # Fixed structure -> fixed job list, shared by every model.
    jobs = []
    truth = {"Public Comment": [], "Public Hearing": []}
    in_window = {"Public Comment": [], "Public Hearing": []}
    owner = {"Public Comment": [], "Public Hearing": []}
    for city, mk, utts in meetings:
        structures = json.loads((STRUCT_CACHE / f"{city}_meeting_structure.json").read_text())
        pc_idx, ph_idx = public_section_indices(structures[mk])
        for section, idxs in (("Public Comment", pc_idx), ("Public Hearing", ph_idx)):
            cfg = SECTION_CFG[section]
            for i, u in enumerate(utts):
                truth[section].append(1 if u.get(cfg["gt"]) in (1, "1", True) else 0)
                in_window[section].append(i in idxs)
                owner[section].append((city, mk))
                if i in idxs:
                    jobs.append((section, len(truth[section]) - 1, u))
    for section in truth:
        g = sum(truth[section])
        inside = sum(1 for t, w in zip(truth[section], in_window[section]) if t and w)
        print(f"  {section}: {g} GT positives over {len(truth[section])} utterances; "
              f"{inside} inside a predicted window (structure recall ceiling {inside/g:.3f})")
    print(f"  classifier calls per model: {len(jobs)}")

    report = {}
    for model in args.models:
        print(f"\n--- {model} ---", flush=True)
        t0 = time.time()

        def one(job):
            section, slot, u = job
            cfg = SECTION_CFG[section]
            prompt = build_prompt(
                cfg, u.get("speaker", ""),
                str(u.get("text") or "").strip(), u.get("start"), u.get("end"),
            )
            return section, slot, ask(model, prompt, cfg["key"])

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            out = list(pool.map(one, jobs))

        preds = {s: [0] * len(v) for s, v in truth.items()}
        for section, slot, val in out:
            preds[section][slot] = val

        elapsed = time.time() - t0
        tin, tout, ncalls = USAGE.get(model, [0, 0, 0])
        report[model] = {
            "seconds": round(elapsed, 1),
            "calls": ncalls,
            "input_tokens": tin,
            "output_tokens": tout,
            "out_tokens_per_call": round(tout / ncalls, 1) if ncalls else 0,
        }
        for s in truth:
            # end_to_end: scored over the whole meeting, so structure misses count
            # as false negatives. within_window: classifier only, comparable to the
            # older within-section evaluations.
            report[model][s] = {
                "end_to_end": metrics(truth[s], preds[s]),
                "within_window": metrics(
                    [t for t, w in zip(truth[s], in_window[s]) if w],
                    [p for p, w in zip(preds[s], in_window[s]) if w],
                ),
            }
            e, w = report[model][s]["end_to_end"], report[model][s]["within_window"]
            print(f"  {s:15s} end-to-end  acc={e['accuracy']:.3f} F1={e['f1']:.3f} "
                  f"(P={e['precision']:.3f} R={e['recall']:.3f}) n={e['support']}")
            print(f"  {'':15s} in-window   acc={w['accuracy']:.3f} F1={w['f1']:.3f} "
                  f"(P={w['precision']:.3f} R={w['recall']:.3f}) n={w['support']}")
        print(f"  {elapsed:.0f}s, {report[model]['out_tokens_per_call']} output tok/call")

        per_meeting = {}
        for s_name in truth:
            for t, pr, w, key in zip(truth[s_name], preds[s_name], in_window[s_name], owner[s_name]):
                slot = per_meeting.setdefault(f"{key[0]}|{key[1]}|{s_name}", {"t": [], "p": []})
                slot["t"].append(t)
                slot["p"].append(pr)
        report[model]["per_meeting"] = {
            k: metrics(v["t"], v["p"]) for k, v in per_meeting.items()
        }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
