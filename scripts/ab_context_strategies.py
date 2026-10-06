#!/usr/bin/env python3
"""Compare ways of giving the remark classifier context, on identical section windows.

base  one LLM call per utterance, no context at all (what production does today)
A     one call per utterance, plus +/-N neighbouring utterances as context
B     one call per utterance, plus speaker-level facts computed from the meeting
      (how often this speaker talks, across how many agenda stages) - no extra LLM cost
C     one call per chunk of a window; the model returns the indices that are
      public remarks, so it sees the whole back-and-forth and the window text is
      sent once instead of once per utterance

The section windows come from the cached structures and are identical across
strategies, so the only variable is what the classifier gets to see.

Usage:
  python scripts/_run_with_env.py scripts/ab_context_strategies.py --meetings 100
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "all_splits_localgov_train_test_val" / "all_splits"
STRUCT_CACHE = ROOT / "results" / "all_splits_pred_cache" / "structures"

sys.path.insert(0, str(ROOT))
from pageindex.section_taxonomy import public_section_indices  # noqa: E402

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

# Shared v2 wording: the window also holds the staff presentation and the
# council's own deliberation, and those dominate the false positives.
RULES = """Answer 1 ONLY for someone who is not part of the city government addressing the body: a resident, business owner, applicant, or other member of the public who has come forward to speak. Such a speaker typically states their name and address, speaks in one longer uninterrupted turn, and petitions or urges the council.

Answer 0 for everyone on the government's side of the dais, including:
- the mayor, chair, or any council member, whether running the meeting or debating the item
- city staff, the city manager, attorney, clerk, police or fire chief, or a consultant presenting or answering questions
- procedural talk: calling the next speaker, thanking a speaker, reading the rules, opening or closing the hearing

The staff presentation and the council's back-and-forth deliberation on the item often fall inside this same section. Those are 0, even when the content sounds conversational or personal."""

client = openai.OpenAI()
USAGE: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
COVERAGE = [0, 0]   # (indices asked for, indices the model actually answered)


def call(strategy: str, prompt: str, model: str, max_retries: int = 6) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            u = resp.usage
            USAGE[strategy][0] += u.prompt_tokens
            USAGE[strategy][1] += u.completion_tokens
            USAGE[strategy][2] += 1
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                print(f"    [{strategy}] giving up: {exc}", file=sys.stderr)
                return ""
            time.sleep(1.5 * (attempt + 1))
    return ""


def parse_binary(out: str, key: str) -> int:
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*(\d)', out)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([01])\b", out)
    return int(m.group(1)) if m else 0


def fmt(u: dict) -> str:
    return f"{u.get('speaker','')}: {str(u.get('text') or '').strip()}"


# ---------------------------------------------------------------- strategies


def prompt_base(cfg, utts, i) -> str:
    u = utts[i]
    return f"""You are a binary classifier. This utterance comes from the "{cfg['label']}" portion of a city council meeting. Decide if it is {cfg['desc']}.

{RULES}

Use only the following information (no other fields):
- speaker_id: {u.get('speaker','')}
- text: {str(u.get('text') or '').strip()}
- time: start={u.get('start')} end={u.get('end')} (seconds)

Return JSON: {{ "{cfg['key']}": 0 or 1 }}."""


def prompt_a(cfg, utts, i, ctx: int) -> str:
    lo, hi = max(0, i - ctx), min(len(utts), i + ctx + 1)
    lines = []
    for j in range(lo, hi):
        mark = ">>> " if j == i else "    "
        lines.append(f"{mark}[{j}] {fmt(utts[j])}")
    u = utts[i]
    return f"""You are a binary classifier. The utterance marked ">>>" comes from the "{cfg['label']}" portion of a city council meeting. Decide if it is {cfg['desc']}.

{RULES}

Surrounding utterances are shown only as context. Judge ONLY the one marked ">>>".

{chr(10).join(lines)}

Target utterance: speaker_id={u.get('speaker','')}, time start={u.get('start')} end={u.get('end')}

Return JSON: {{ "{cfg['key']}": 0 or 1 }}."""


def prompt_b(cfg, utts, i, stats: dict) -> str:
    u = utts[i]
    s = stats[u.get("speaker", "")]
    return f"""You are a binary classifier. This utterance comes from the "{cfg['label']}" portion of a city council meeting. Decide if it is {cfg['desc']}.

{RULES}

Speaker behaviour across the whole meeting (a strong clue: officials recur throughout the agenda and take many short turns; members of the public appear only inside the public-speaking window, usually in one burst of consecutive turns):
- this speaker takes {s['turns']} of the meeting's {s['total']} turns ({s['share']:.0%})
- they appear in {s['stages']} of the {s['n_stages']} agenda stages
- {s['in_window']} of their turns fall inside this public-speaking window
- their longest run of consecutive turns here is {s['run']}

Utterance to judge:
- speaker_id: {u.get('speaker','')}
- text: {str(u.get('text') or '').strip()}
- time: start={u.get('start')} end={u.get('end')} (seconds)

Return JSON: {{ "{cfg['key']}": 0 or 1 }}."""


def prompt_c(cfg, utts, label_idx: list[int], ctx_idx: list[int]) -> str:
    lines = []
    for j in sorted(set(label_idx) | set(ctx_idx)):
        tag = "" if j in set(label_idx) else "   (context only)"
        lines.append(f"[{j}] {fmt(utts[j])}{tag}")
    lo, hi = min(label_idx), max(label_idx)
    return f"""Below is a verbatim stretch of the "{cfg['label']}" portion of a city council meeting, with each utterance numbered. Identify every utterance that is {cfg['desc']}.

{RULES}

Because you can see the whole exchange, use it: a member of the public normally speaks in one unbroken run of turns after being called, while officials trade short turns back and forth with each other.

{chr(10).join(lines)}

Consider only indices {lo} through {hi} (lines marked "context only" are background, do not label them).

Return JSON: {{ "public_remark_indices": [list of indices that are public remarks] }}. Return an empty list if there are none."""


def prompt_c_perline(cfg, utts, label_idx: list[int], ctx_idx: list[int]) -> str:
    """Same context as prompt_c, but every index must be answered explicitly.

    Returning only a list of hits lets the model quietly skip utterances, which
    showed up as a systematic recall loss; an exhaustive verdict per line forces
    it to take a position on each one.
    """
    lines = []
    for j in sorted(set(label_idx) | set(ctx_idx)):
        tag = "" if j in set(label_idx) else "   (context only)"
        lines.append(f"[{j}] {fmt(utts[j])}{tag}")
    lo, hi = min(label_idx), max(label_idx)
    return f"""Below is a verbatim stretch of the "{cfg['label']}" portion of a city council meeting, with each utterance numbered. For EVERY numbered utterance, decide whether it is {cfg['desc']}.

{RULES}

Because you can see the whole exchange, use it: a member of the public normally speaks in one unbroken run of turns after being called, while officials trade short turns back and forth with each other.

{chr(10).join(lines)}

Label indices {lo} through {hi} ({len(label_idx)} utterances; lines marked "context only" are background, skip them).

Output one line per index, in order, with no other text:
<index>: <0 or 1>

You must output exactly {len(label_idx)} lines."""


# ---------------------------------------------------------------- helpers


def contiguous_runs(idxs: set[int]) -> list[list[int]]:
    out, cur = [], []
    for i in sorted(idxs):
        if cur and i == cur[-1] + 1:
            cur.append(i)
        else:
            if cur:
                out.append(cur)
            cur = [i]
    if cur:
        out.append(cur)
    return out


def chunk_run(run: list[int], utts: list, size: int, mode: str) -> list[list[int]]:
    """Split one contiguous window into blocks for strategy C.

    "speaker" mode never cuts in the middle of a speaker's uninterrupted turn,
    which is the cue the model uses to tell a member of the public apart from
    officials trading short turns.
    """
    if len(run) <= size:
        return [run]
    if mode == "fixed":
        return [run[k:k + size] for k in range(0, len(run), size)]

    groups, cur = [], [run[0]]
    for prev, j in zip(run, run[1:]):
        same = utts[j].get("speaker") == utts[prev].get("speaker")
        if same:
            cur.append(j)
        else:
            groups.append(cur)
            cur = [j]
    groups.append(cur)

    blocks, cur = [], []
    for g in groups:
        if cur and len(cur) + len(g) > size:
            blocks.append(cur)
            cur = []
        cur.extend(g)
    if cur:
        blocks.append(cur)
    return blocks


def speaker_stats(utts: list, stages: list, window: set[int]) -> dict:
    total = len(utts)
    stage_of = {}
    for n, s in enumerate(stages):
        for j in range(int(s.get("start_index", 0)), int(s.get("end_index", 0)) + 1):
            stage_of[j] = n
    turns, in_stage, in_win = defaultdict(int), defaultdict(set), defaultdict(list)
    for j, u in enumerate(utts):
        sp = u.get("speaker", "")
        turns[sp] += 1
        if j in stage_of:
            in_stage[sp].add(stage_of[j])
        if j in window:
            in_win[sp].append(j)
    out = {}
    for sp in turns:
        runs = contiguous_runs(set(in_win[sp])) if in_win[sp] else []
        out[sp] = {
            "turns": turns[sp],
            "total": total,
            "share": turns[sp] / total if total else 0,
            "stages": len(in_stage[sp]),
            "n_stages": max(1, len(stages)),
            "in_window": len(in_win[sp]),
            "run": max((len(r) for r in runs), default=0),
        }
    return out


def metrics(y_true, y_pred) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(pr, 4), "recall": round(rc, 4),
        "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def load_meetings(limit: int) -> list:
    """Round-robin across cities so no single city dominates."""
    by_city = defaultdict(list)
    seen = set()
    for path in sorted(SPLITS.glob("*.json")):
        city = path.stem.split("_")[0]
        sf = STRUCT_CACHE / f"{city}_meeting_structure.json"
        if not sf.is_file():
            continue
        structs = json.loads(sf.read_text())
        for mk, utts in json.loads(path.read_text()).items():
            if mk in seen or mk not in structs:
                continue
            seen.add(mk)
            gt = sum(
                1 for u in utts
                if u.get("is_public_comment_gt") in (1, "1", True)
                or u.get("is_public_hearing_gt") in (1, "1", True)
            )
            by_city[city].append((city, mk, utts, structs[mk], gt))
    for c in by_city:
        by_city[c].sort(key=lambda x: -x[4])
    picked, rnd = [], 0
    while len(picked) < limit and any(len(v) > rnd for v in by_city.values()):
        for c in sorted(by_city):
            if len(by_city[c]) > rnd and len(picked) < limit:
                picked.append(by_city[c][rnd][:4])
        rnd += 1
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", type=int, default=100)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--ctx", type=int, default=3, help="neighbours each side for A")
    ap.add_argument("--chunk", type=int, default=40, help="utterances labelled per call in C")
    ap.add_argument("--chunk-ctx", type=int, default=5, help="context lines around a C chunk")
    ap.add_argument("--chunk-mode", default="fixed", choices=["fixed", "speaker"])
    ap.add_argument("--c-format", default="indices", choices=["indices", "perline"])
    ap.add_argument("--strategies", nargs="*", default=["base", "A", "B", "C"])
    ap.add_argument("--out", default=str(ROOT / "results" / "ab_context_strategies.json"))
    args = ap.parse_args()

    meetings = load_meetings(args.meetings)
    print(f"{len(meetings)} meetings from {len({m[0] for m in meetings})} cities, model={args.model}")

    # Shared ground truth + identical windows for every strategy.
    truth = {s: [] for s in SECTION_CFG}
    slot_of = {}            # (section, meeting, idx) -> slot
    tasks = []              # (section, meeting_key, utts, stages, window set)
    for city, mk, utts, stages in meetings:
        pc_idx, ph_idx = public_section_indices(stages)
        for section, idxs in (("Public Comment", pc_idx), ("Public Hearing", ph_idx)):
            cfg = SECTION_CFG[section]
            for i, u in enumerate(utts):
                slot_of[(section, mk, i)] = len(truth[section])
                truth[section].append(1 if u.get(cfg["gt"]) in (1, "1", True) else 0)
            if idxs:
                tasks.append((section, mk, utts, stages, idxs))

    n_window = sum(len(t[4]) for t in tasks)
    for s in truth:
        print(f"  {s}: {sum(truth[s])} GT positives over {len(truth[s])} utterances")
    print(f"  utterances inside a window: {n_window}")

    report = {}
    for strat in args.strategies:
        print(f"\n--- {strat} ---", flush=True)
        t0 = time.time()
        units = []
        if strat == "C":
            for section, mk, utts, stages, idxs in tasks:
                for run in contiguous_runs(idxs):
                    for block in chunk_run(run, utts, args.chunk, args.chunk_mode):
                        lo, hi = block[0], block[-1]
                        ctx = [j for j in range(max(0, lo - args.chunk_ctx),
                                                min(len(utts), hi + args.chunk_ctx + 1))
                               if j not in block]
                        units.append((section, mk, utts, block, ctx))
        else:
            for section, mk, utts, stages, idxs in tasks:
                stats = speaker_stats(utts, stages, idxs) if strat == "B" else None
                for i in sorted(idxs):
                    units.append((section, mk, utts, i, stats))

        def run_unit(unit):
            if strat == "C":
                section, mk, utts, block, ctx = unit
                cfg = SECTION_CFG[section]
                if args.c_format == "perline":
                    out = call(strat, prompt_c_perline(cfg, utts, block, ctx), args.model)
                    got = {int(a): int(b) for a, b in re.findall(r"(\d+)\s*:\s*([01])", out)}
                    COVERAGE[0] += len(block)
                    COVERAGE[1] += sum(1 for i in block if i in got)
                    return [(section, mk, i, got.get(i, 0)) for i in block]
                out = call(strat, prompt_c(cfg, utts, block, ctx), args.model)
                m = re.search(r"\[([\d,\s]*)\]", out)
                hits = set()
                if m:
                    hits = {int(x) for x in re.findall(r"\d+", m.group(1))}
                hits &= set(block)          # ignore anything outside the block
                return [(section, mk, i, 1 if i in hits else 0) for i in block]
            section, mk, utts, i, stats = unit
            cfg = SECTION_CFG[section]
            if strat == "base":
                p = prompt_base(cfg, utts, i)
            elif strat == "A":
                p = prompt_a(cfg, utts, i, args.ctx)
            else:
                p = prompt_b(cfg, utts, i, stats)
            return [(section, mk, i, parse_binary(call(strat, p, args.model), cfg["key"]))]

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run_unit, units))

        preds = {s: [0] * len(truth[s]) for s in truth}
        for group in results:
            for section, mk, i, val in group:
                preds[section][slot_of[(section, mk, i)]] = val

        tin, tout, ncalls = USAGE[strat]
        price = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4o-2024-11-20": (2.50, 10.0)}[args.model]
        report[strat] = {
            "seconds": round(time.time() - t0, 1),
            "llm_calls": ncalls,
            "input_tokens": tin,
            "output_tokens": tout,
            "cost_usd": round(tin / 1e6 * price[0] + tout / 1e6 * price[1], 4),
            **{s: metrics(truth[s], preds[s]) for s in truth},
        }
        r = report[strat]
        for s in truth:
            print(f"  {s:15s} F1={r[s]['f1']:.3f} (P={r[s]['precision']:.3f} R={r[s]['recall']:.3f})")
        if COVERAGE[0]:
            print(f"  parse coverage: {COVERAGE[1]:,}/{COVERAGE[0]:,} "
                  f"({COVERAGE[1]/COVERAGE[0]:.3%}) indices actually answered by the model")
        print(f"  {r['llm_calls']} calls, {r['input_tokens']:,} in / {r['output_tokens']:,} out, "
              f"${r['cost_usd']:.3f}, {r['seconds']:.0f}s")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
