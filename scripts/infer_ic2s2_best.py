#!/usr/bin/env python3
"""Run the winning configuration over the seven IC2S2 cities and score PC / PH.

Winning configuration, as measured by scripts/ab_context_strategies.py:
  stage 1  cached section windows from pageindex.section_taxonomy
  stage 2  strategy C, per-line output format, chunk 100, +/-5 context lines,
           v2 rules, gpt-4.1-mini

Meetings whose structure is not cached get one extracted here (gpt-4o, matching
how the rest of the cache was built) so every annotated meeting is scored.

Usage:
  python scripts/_run_with_env.py scripts/infer_ic2s2_best.py
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
from pageindex.section_taxonomy import classify_stage, public_section_indices  # noqa: E402

CITIES = ["SEA", "OAK", "RCH", "AA", "LS", "RO", "JS", "AP", "PE", "IN"]
CITY_NAMES = {
    "SEA": "Seattle, WA",
    "OAK": "Oakland, CA",
    "RCH": "Richmond, VA",
    "AA": "Ann Arbor, MI",
    "LS": "Lansing, MI",
    "RO": "Royal Oak, MI",
    "JS": "Jackson, MS",
    "AP": "Alpena, MI",
    "PE": "Perry, MI",
    "IN": "Inkster, MI",
}

SECTION_CFG = {
    "Public Comment": {
        "label": "Public Comment",
        "desc": "an actual PUBLIC COMMENT (a citizen/member of the public addressing council)",
        "gt": "is_public_comment_gt",
    },
    "Public Hearing": {
        "label": "Public Hearing",
        "desc": (
            "an actual PUBLIC REMARK during the hearing (a citizen/member of the public "
            "speaking on the hearing item)"
        ),
        "gt": "is_public_hearing_gt",
    },
}

RULES_V2 = """Answer 1 ONLY for someone who is not part of the city government addressing the body: a resident, business owner, applicant, or other member of the public who has come forward to speak. Such a speaker typically states their name and address, speaks in one longer uninterrupted turn, and petitions or urges the council.

Answer 0 for everyone on the government's side of the dais, including:
- the mayor, chair, or any council member, whether running the meeting or debating the item
- city staff, the city manager, attorney, clerk, police or fire chief, or a consultant presenting or answering questions
- procedural talk: calling the next speaker, thanking a speaker, reading the rules, opening or closing the hearing

The staff presentation and the council's back-and-forth deliberation on the item often fall inside this same section. Those are 0, even when the content sounds conversational or personal."""

# Ablation: the annotators scored a petitioner fielding council questions as 0,
# so "applicant" in the v2 positive list conflicts with the gold standard.
RULES_V2_NO_APPLICANT = """Answer 1 ONLY for a member of the public who has come forward to address the body on their own initiative: a resident, neighbour, or community member who signed up to speak. Such a speaker typically states their name and address, speaks in one longer uninterrupted turn, and petitions or urges the council.

Answer 0 for everyone on the government's side of the dais, and for anyone appearing in an official capacity on an item, including:
- the mayor, chair, or any council member, whether running the meeting or debating the item
- city staff, the city manager, attorney, clerk, police or fire chief, or a consultant presenting or answering questions
- the applicant, developer, petitioner, or their representative when they are presenting their own proposal or answering the council's questions about it
- procedural talk: calling the next speaker, thanking a speaker, reading the rules, opening or closing the hearing

The staff presentation and the council's back-and-forth deliberation on the item often fall inside this same section. Those are 0, even when the content sounds conversational or personal."""

RULESETS = {"v2": RULES_V2, "v2-no-applicant": RULES_V2_NO_APPLICANT}
RULES = RULES_V2       # rebound from --rules in main()

PRICE = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4o-2024-11-20": (2.50, 10.0)}

client = openai.OpenAI()
USAGE = [0, 0, 0]       # prompt tokens, completion tokens, calls
COVERAGE = [0, 0]       # indices asked for, indices the model answered


def call(prompt: str, model: str, max_retries: int = 6) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            u = resp.usage
            USAGE[0] += u.prompt_tokens
            USAGE[1] += u.completion_tokens
            USAGE[2] += 1
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                print(f"    giving up: {exc}", file=sys.stderr)
                return ""
            time.sleep(1.5 * (attempt + 1))
    return ""


def fmt(u: dict) -> str:
    return f"{u.get('speaker','')}: {str(u.get('text') or '').strip()}"


def prompt_c_perline(cfg, utts, label_idx, ctx_idx) -> str:
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


WINDOW_TYPE_PROMPT = """A city council has reached a segment of its meeting where members of the public speak. The agenda heading is shown below, followed by the opening utterances of that segment.

Decide which kind of public-speaking window this is:

A = PUBLIC HEARING — testimony on one or more SPECIFIC items already on tonight's agenda (a rezoning, a special land use, a licence, an ordinance, a budget). The chair usually names the items or case numbers and limits speakers to them.

B = GENERAL PUBLIC COMMENT — an open window where residents may raise any topic, including matters not on the agenda at all.

Cities file these under wildly inconsistent headings: some run a formal hearing under a heading that says "public comment", others call an open-topic window a "hearing". Judge by what the chair actually says is happening in the transcript, not by the heading.

Agenda heading: {name}

Opening utterances:
{lines}

Answer with exactly one letter, A or B, and nothing else."""


def type_window(stage_name: str, utts: list, lo: int, hi: int, model: str) -> str:
    """Return "ph" or "pc" for one public-speaking window, judged semantically.

    Stage names alone cannot separate the two in cities like Lansing, which runs
    its noticed hearings inside a block headed "Public Comment on Legislative
    Matters" and reserves a separately headed block for open-topic comment.
    """
    lines = "\n".join(
        f"[{j}] {fmt(utts[j])[:400]}" for j in range(lo, min(lo + 8, hi + 1))
    )
    out = call(WINDOW_TYPE_PROMPT.format(name=stage_name, lines=lines), model).strip().upper()
    m = re.search(r"\b([AB])\b", out)
    return "ph" if (m and m.group(1) == "A") else "pc"


# A heading that scopes public comment to the items noticed for tonight. A city
# that uses one of these AND a separate general block is running its hearings
# inside a "public comment" heading, so the heading alone cannot tell the two
# windows apart and the content has to be read.
_AGENDA_SCOPED = ("on legislative matter", "on agenda item", "on legislative item",
                  "on items on the agenda", "on the agenda item")
_GENERAL_SCOPED = ("city government", "non-agenda", "non agenda", "general matter",
                   "any other matter", "not on the agenda")


def city_needs_content_typing(structs: dict) -> bool:
    """True when a city splits public comment by agenda scope rather than by heading."""
    scoped = general = False
    for stages in structs.values():
        for s in stages:
            if classify_stage(s.get("stage_name")) != "pc":
                continue
            name = (s.get("stage_name") or "").lower()
            scoped |= any(p in name for p in _AGENDA_SCOPED)
            general |= any(p in name for p in _GENERAL_SCOPED)
    return scoped and general


def typed_section_indices(stages: list, utts: list, model: str, workers: int):
    """public_section_indices, but each window's kind is decided by content."""
    public = []
    for s in stages:
        kind = classify_stage(s.get("stage_name"))
        if kind is None:
            continue
        lo, hi = int(s.get("start_index", 0)), int(s.get("end_index", 0))
        lo, hi = max(0, lo), min(len(utts) - 1, hi)
        if lo <= hi:
            public.append((s.get("stage_name") or "", lo, hi))
    if not public:
        return set(), set()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(public)))) as pool:
        kinds = list(pool.map(lambda w: type_window(w[0], utts, w[1], w[2], model), public))
    pc, ph = set(), set()
    for (_, lo, hi), k in zip(public, kinds):
        (ph if k == "ph" else pc).update(range(lo, hi + 1))
    return pc, ph


def contiguous_runs(idxs) -> list[list[int]]:
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


def chunk_run(run: list[int], size: int) -> list[list[int]]:
    if len(run) <= size:
        return [run]
    return [run[k:k + size] for k in range(0, len(run), size)]


def metrics(y_true, y_pred) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(pr, 4), "recall": round(rc, 4),
        "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "n_pos": tp + fn,
    }


def sanitize(u: dict) -> dict:
    row = dict(u)
    for k in ("text", "speaker"):
        if row.get(k) is not None and not isinstance(row[k], str):
            row[k] = str(row[k])
    return row


def load_city_meetings(city: str) -> dict[str, list]:
    """meeting_key -> utterances, deduped across the train/test/val splits."""
    out: dict[str, list] = {}
    for fp in sorted(SPLITS.glob(f"{city}_*.json")):
        for mk, utts in json.loads(fp.read_text(encoding="utf-8")).items():
            if mk not in out:
                out[mk] = [sanitize(u) for u in utts]
    return out


def load_structures(city: str) -> dict[str, list]:
    sf = STRUCT_CACHE / f"{city}_meeting_structure.json"
    return json.loads(sf.read_text(encoding="utf-8")) if sf.is_file() else {}


def save_structures(city: str, new: dict[str, list]) -> None:
    STRUCT_CACHE.mkdir(parents=True, exist_ok=True)
    sf = STRUCT_CACHE / f"{city}_meeting_structure.json"
    cur = json.loads(sf.read_text(encoding="utf-8")) if sf.is_file() else {}
    cur.update(new)
    sf.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--structure-model", default="gpt-4o-2024-11-20")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--chunk-ctx", type=int, default=5)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--skip-structure", action="store_true",
                    help="score only the meetings that already have a cached structure")
    ap.add_argument("--rules", default="v2", choices=sorted(RULESETS))
    ap.add_argument("--window-typing", default="name", choices=["name", "llm", "auto"],
                    help="'name' maps a stage to PC/PH by its heading; 'llm' reads the "
                         "window's opening turns and decides what kind of window it is; "
                         "'auto' uses 'llm' only for cities whose headings split public "
                         "comment by agenda scope")
    ap.add_argument("--out", default=str(ROOT / "results" / "ic2s2_best_config.json"))
    args = ap.parse_args()

    global RULES
    RULES = RULESETS[args.rules]

    # ---------------------------------------------------------- stage 1
    corpus: dict[str, dict] = {}
    for city in CITIES:
        meetings = load_city_meetings(city)
        structs = load_structures(city)
        missing = [mk for mk in meetings if mk not in structs]
        if missing and not args.skip_structure:
            print(f"[{city}] extracting structure for {len(missing)} meeting(s)...", flush=True)
            from pageindex.meeting_structure import extract_meeting_structure
            new = extract_meeting_structure(
                meeting_data={mk: meetings[mk] for mk in missing},
                model=args.structure_model,
                workers=min(args.workers, len(missing)),
            )
            save_structures(city, new)
            structs.update(new)
        corpus[city] = {
            "meetings": {mk: u for mk, u in meetings.items() if mk in structs},
            "structs": structs,
            "skipped": [mk for mk in meetings if mk not in structs],
        }
        c = corpus[city]
        print(f"[{city}] {len(c['meetings'])} meetings scored, {len(c['skipped'])} without structure")

    # ---------------------------------------------------------- windows + GT
    # Ground truth is scored over every utterance in the meeting, so a window
    # the structure step missed shows up as a false negative, matching the way
    # the pipeline is actually used.
    truth = {c: {s: [] for s in SECTION_CFG} for c in CITIES}
    inwin = {c: {s: [] for s in SECTION_CFG} for c in CITIES}   # GT flag, is-in-window
    slot_of = {}
    tasks = []
    by_content = {
        c: args.window_typing == "llm"
        or (args.window_typing == "auto" and city_needs_content_typing(corpus[c]["structs"]))
        for c in CITIES
    }
    if any(by_content.values()):
        picked = [c for c in CITIES if by_content[c]]
        print(f"typing windows by content for {', '.join(picked)} ({args.model})", flush=True)
    for city in CITIES:
        for mk, utts in corpus[city]["meetings"].items():
            stages = corpus[city]["structs"][mk]
            if by_content[city]:
                pc_idx, ph_idx = typed_section_indices(stages, utts, args.model, args.workers)
            else:
                pc_idx, ph_idx = public_section_indices(stages)
            for section, idxs in (("Public Comment", pc_idx), ("Public Hearing", ph_idx)):
                cfg = SECTION_CFG[section]
                for i, u in enumerate(utts):
                    g = 1 if u.get(cfg["gt"]) in (1, "1", True) else 0
                    slot_of[(city, section, mk, i)] = len(truth[city][section])
                    truth[city][section].append(g)
                    inwin[city][section].append(1 if i in idxs else 0)
                if idxs:
                    tasks.append((city, section, mk, utts, idxs))

    n_window = sum(len(t[4]) for t in tasks)
    print(f"\n{sum(len(corpus[c]['meetings']) for c in CITIES)} meetings, "
          f"{sum(len(truth[c]['Public Comment']) for c in CITIES):,} utterances, "
          f"{n_window:,} inside a predicted window")

    # ---------------------------------------------------------- stage 2
    units = []
    for city, section, mk, utts, idxs in tasks:
        for run in contiguous_runs(idxs):
            for block in chunk_run(run, args.chunk):
                lo, hi = block[0], block[-1]
                ctx = [j for j in range(max(0, lo - args.chunk_ctx),
                                        min(len(utts), hi + args.chunk_ctx + 1))
                       if j not in block]
                units.append((city, section, mk, utts, block, ctx))

    print(f"{len(units)} LLM calls with {args.model} (chunk={args.chunk})", flush=True)

    def run_unit(unit):
        city, section, mk, utts, block, ctx = unit
        out = call(prompt_c_perline(SECTION_CFG[section], utts, block, ctx), args.model)
        got = {int(a): int(b) for a, b in re.findall(r"(\d+)\s*:\s*([01])", out)}
        COVERAGE[0] += len(block)
        COVERAGE[1] += sum(1 for i in block if i in got)
        return [(city, section, mk, i, got.get(i, 0)) for i in block]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_unit, units))
    elapsed = time.time() - t0

    preds = {c: {s: [0] * len(truth[c][s]) for s in SECTION_CFG} for c in CITIES}
    for group in results:
        for city, section, mk, i, val in group:
            preds[city][section][slot_of[(city, section, mk, i)]] = val

    # ---------------------------------------------------------- scoring
    report = {"config": vars(args), "cities": {}, "overall": {}}
    agg = {s: {"t": [], "p": [], "w": []} for s in SECTION_CFG}

    for city in CITIES:
        row = {"name": CITY_NAMES[city],
               "meetings": len(corpus[city]["meetings"]),
               "meetings_without_structure": len(corpus[city]["skipped"])}
        for section in SECTION_CFG:
            t, p, w = truth[city][section], preds[city][section], inwin[city][section]
            agg[section]["t"] += t
            agg[section]["p"] += p
            agg[section]["w"] += w
            npos = sum(t)
            row[section] = {
                "end_to_end": metrics(t, p),
                "within_window": metrics([a for a, f in zip(t, w) if f],
                                         [a for a, f in zip(p, w) if f]),
                "structure_recall": round(
                    sum(1 for a, f in zip(t, w) if a == 1 and f) / npos, 4) if npos else None,
            }
        report["cities"][city] = row

    for section in SECTION_CFG:
        t, p, w = agg[section]["t"], agg[section]["p"], agg[section]["w"]
        npos = sum(t)
        report["overall"][section] = {
            "end_to_end": metrics(t, p),
            "within_window": metrics([a for a, f in zip(t, w) if f],
                                     [a for a, f in zip(p, w) if f]),
            "structure_recall": round(
                sum(1 for a, f in zip(t, w) if a == 1 and f) / npos, 4) if npos else None,
        }

    pin, pout, ncalls = USAGE
    price = PRICE.get(args.model, (0.40, 1.60))
    report["cost"] = {
        "llm_calls": ncalls, "input_tokens": pin, "output_tokens": pout,
        "cost_usd": round(pin / 1e6 * price[0] + pout / 1e6 * price[1], 4),
        "seconds": round(elapsed, 1),
    }
    report["parse_coverage"] = {
        "asked": COVERAGE[0], "answered": COVERAGE[1],
        "rate": round(COVERAGE[1] / COVERAGE[0], 6) if COVERAGE[0] else None,
    }

    # ---------------------------------------------------------- print
    for section in SECTION_CFG:
        print(f"\n=== {section} (end-to-end, whole meeting) ===")
        print(f"{'city':22s} {'mtgs':>5s} {'GT+':>6s} {'P':>7s} {'R':>7s} {'F1':>7s} "
              f"{'sect.R':>7s} {'inwin F1':>9s}")
        for city in CITIES:
            r = report["cities"][city][section]
            e, wv = r["end_to_end"], r["within_window"]
            sr = r["structure_recall"]
            print(f"{CITY_NAMES[city]:22s} {report['cities'][city]['meetings']:5d} "
                  f"{e['n_pos']:6d} {e['precision']:7.3f} {e['recall']:7.3f} {e['f1']:7.3f} "
                  f"{(f'{sr:.3f}' if sr is not None else '    -'):>7s} {wv['f1']:9.3f}")
        o = report["overall"][section]
        print(f"{f'ALL {len(CITIES)} CITIES':22s} {sum(report['cities'][c]['meetings'] for c in CITIES):5d} "
              f"{o['end_to_end']['n_pos']:6d} {o['end_to_end']['precision']:7.3f} "
              f"{o['end_to_end']['recall']:7.3f} {o['end_to_end']['f1']:7.3f} "
              f"{o['structure_recall']:7.3f} {o['within_window']['f1']:9.3f}")

    cv = report["parse_coverage"]
    print(f"\nparse coverage {cv['answered']:,}/{cv['asked']:,} ({cv['rate']:.4%})")
    print(f"{ncalls} calls, {pin:,} in / {pout:,} out, "
          f"${report['cost']['cost_usd']:.3f}, {elapsed:.0f}s")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
