#!/usr/bin/env python3
"""
Resolve localgov transcripts by FIBS + year(s), then run the RAG prediction pipeline.

Transcript path:
  {localgov_root}/{year}/{fibs}/transcripts/

Default LLM: OpenAI GPT (LLM_BACKEND=openai). Set CHATGPT_API_KEY before running.

Examples:
  cd AgendaIndex

  # Single year, numeric FIBS (folder name on disk)
  python run_rag_by_fibs.py --fibs 00674 --year 2023

  # Skip structure infer; reuse exact or combined structure file for that year
  python run_rag_by_fibs.py --fibs 57000 --year 2023 --skip_structure

  # Explicit structure file (filtered to loaded meetings)
  python run_rag_by_fibs.py --fibs 57000 --year 2023 --structure_path results/57000_2021_2026_meeting_structure.json

  # Default range 2020-2026
  python run_rag_by_fibs.py --fibs 00674

  # List files only
  python run_rag_by_fibs.py --fibs 00674 --year_start 2020 --year_end 2026 --list_only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Default to GPT API unless user overrides in the environment.
os.environ.setdefault("LLM_BACKEND", "openai")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_LOCALGOV_ROOT = os.environ.get(
    "LOCALGOV_ROOT",
    "/home/shared/turbo_videos/processed_videos/localgov",
)
DEFAULT_GPT_MODEL = "gpt-4o-2024-11-20"
DEFAULT_YEAR_START = 2020
DEFAULT_YEAR_END = 2026


def _indices_in_stage(stages: list, stage_name: str) -> list:
    """Indices across ALL stages matching stage_name (repeated/aliased public windows)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pageindex.section_taxonomy import indices_for_section

    return indices_for_section(stages, stage_name)


def compute_metrics(y_true: list, y_pred: list) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = len(y_true)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# Sections to extract remarks from, with the matching ground-truth field in the data.
SECTION_SPECS = [
    {"section": "Public Comment", "stage": "Public Comment", "gt_field": "is_public_comment"},
    {"section": "Public Hearing", "stage": "Public Hearing", "gt_field": "is_public_hearing"},
]


def run_predictions(
    meeting_data: dict, structure: dict, model: str, workers: int = 1
) -> tuple[list, dict]:
    """
    Classify member-of-public remarks within both Public Comment and Public Hearing sections.

    One LLM call per utterance; `workers` threads issue those calls concurrently.

    Returns (all_results, metrics_by_section).
    """
    from pageindex.meeting_structure import classify_utterance_public_remark

    jobs = []
    sections_found = set()
    for spec in SECTION_SPECS:
        section = spec["section"]
        gt_field = spec["gt_field"]
        for meeting_date, utterances in meeting_data.items():
            stages = structure.get(meeting_date, [])
            indices = _indices_in_stage(stages, spec["stage"])
            if not indices:
                continue
            sections_found.add(section)
            for idx in indices:
                if idx >= len(utterances):
                    continue
                u = utterances[idx]
                speaker = u.get("speaker", "")
                gt_raw = u.get(gt_field)
                has_gt = gt_raw is not None and gt_raw not in ("",)
                jobs.append(
                    {
                        "meeting_date": meeting_date,
                        "section": section,
                        "utterance_index": idx,
                        "speaker": "" if speaker is None else str(speaker),
                        "text": str(u.get("text") if u.get("text") is not None else "").strip(),
                        "start": u.get("start"),
                        "end": u.get("end"),
                        "ground_truth": 1 if gt_raw in (1, "1", True) else 0 if has_gt else None,
                    }
                )

    for spec in SECTION_SPECS:
        if spec["section"] not in sections_found:
            print(f"  No '{spec['section']}' stage found in any meeting; skipped.")

    def _classify(job: dict) -> dict:
        pred = classify_utterance_public_remark(
            speaker=job["speaker"],
            text=job["text"],
            start=job["start"],
            end=job["end"],
            section=job["section"],
            model=model,
        )
        text = job["text"]
        return {
            **job,
            "text": text[:200] + "..." if len(text) > 200 else text,
            "predicted_is_remark": pred,
        }

    if workers > 1 and jobs:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            all_results = list(pool.map(_classify, jobs))
    else:
        all_results = [_classify(job) for job in jobs]

    # Keep agenda order so downstream diffs stay readable.
    all_results.sort(key=lambda r: (r["section"], r["meeting_date"], r["utterance_index"]))

    metrics_by_section: dict = {}
    for spec in SECTION_SPECS:
        section = spec["section"]
        pairs = [
            (r["ground_truth"], r["predicted_is_remark"])
            for r in all_results
            if r["section"] == section and r["ground_truth"] is not None
        ]
        # transcript_loader fills is_public_comment/is_public_hearing with 0 for
        # unlabelled deployment transcripts. A real annotation of a PC/PH window
        # always contains positives, so an all-zero column is a placeholder and
        # must not be scored as ground truth.
        if not any(t == 1 for t, _ in pairs):
            for r in all_results:
                if r["section"] == section:
                    r["ground_truth"] = None
            pairs = []
        metrics_by_section[section] = (
            compute_metrics([t for t, _ in pairs], [p for _, p in pairs]) if pairs else None
        )

    return all_results, metrics_by_section


def _parse_years(args) -> tuple[list[int], str]:
    """Return (years list, output tag suffix)."""
    if args.year is not None:
        y = int(str(args.year).strip() if len(str(args.year)) == 4 else f"20{args.year}")
        return [y], str(y)

    start = int(args.year_start)
    end = int(args.year_end)
    if end < start:
        raise ValueError(f"year_end ({end}) must be >= year_start ({start})")
    years = list(range(start, end + 1))
    if start == end:
        return years, str(start)
    return years, f"{start}_{end}"


def _is_valid_meeting_structure(structure: object) -> bool:
    """Reject malformed structure JSON (e.g. raw transcript segments saved by mistake)."""
    if not isinstance(structure, dict) or not structure:
        return False
    if "segments" in structure or "word_segments" in structure:
        return False
    return any(isinstance(v, list) for v in structure.values())


def _filter_structure_for_meetings(structure: dict, meeting_keys: set[str]) -> dict:
    return {k: v for k, v in structure.items() if k in meeting_keys}


def _load_structure_file(path: str, meeting_keys: set[str]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        structure = json.load(f)
    if not _is_valid_meeting_structure(structure):
        raise ValueError(f"Invalid meeting structure file (unexpected format): {path}")
    filtered = _filter_structure_for_meetings(structure, meeting_keys)
    return filtered


def _resolve_structure(
    *,
    skip_structure: bool,
    structure_path: str | None,
    fibs_tag: str,
    tag: str,
    meeting_data: dict,
    results_dir: str,
    cache_filtered: bool = True,
) -> tuple[dict | None, str | None]:
    """
    Resolve meeting structure without re-inferring when possible.

    Priority:
      1. --structure_path (explicit)
      2. {tag}_meeting_structure.json (exact year/range tag)
      3. Any {fibs_tag}_*_meeting_structure.json combined file, filtered to loaded meetings
    """
    meeting_keys = set(meeting_data.keys())
    exact_path = os.path.join(results_dir, f"{tag}_meeting_structure.json")

    if structure_path:
        filtered = _load_structure_file(structure_path, meeting_keys)
        print(f"  Loaded structure from --structure_path -> {structure_path}")
        print(f"  Matched {len(filtered)}/{len(meeting_keys)} loaded meeting(s)")
        if len(filtered) < len(meeting_keys):
            print(f"  Warning: {len(meeting_keys) - len(filtered)} meeting(s) missing structure entries")
        return filtered, structure_path

    if not skip_structure:
        return None, None

    if os.path.isfile(exact_path):
        filtered = _load_structure_file(exact_path, meeting_keys)
        print(f"  Reusing structure -> {exact_path}")
        print(f"  Matched {len(filtered)}/{len(meeting_keys)} loaded meeting(s)")
        return filtered, exact_path

    pattern = os.path.join(results_dir, f"{fibs_tag}_*_meeting_structure.json")
    candidates = [
        p for p in sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if os.path.abspath(p) != os.path.abspath(exact_path)
    ]

    best_path: str | None = None
    best_filtered: dict = {}
    for cand in candidates:
        try:
            filtered = _load_structure_file(cand, meeting_keys)
        except (json.JSONDecodeError, ValueError):
            continue
        if len(filtered) > len(best_filtered):
            best_path, best_filtered = cand, filtered

    if best_path and best_filtered:
        print(f"  Reusing structure from combined file -> {best_path}")
        print(f"  Filtered to {len(best_filtered)}/{len(meeting_keys)} loaded meeting(s) for tag {tag}")
        if len(best_filtered) < len(meeting_keys):
            print(f"  Warning: {len(meeting_keys) - len(best_filtered)} meeting(s) missing structure entries")
        if cache_filtered:
            with open(exact_path, "w", encoding="utf-8") as f:
                json.dump(best_filtered, f, indent=2, ensure_ascii=False)
            print(f"  Cached filtered structure -> {exact_path}")
        return best_filtered, best_path

    searched = exact_path
    if candidates:
        searched += f" (also searched {len(candidates)} combined file(s) under {results_dir})"
    raise FileNotFoundError(
        f"--skip_structure set but no usable structure found for tag {tag}. "
        f"Expected {searched}. Run without --skip_structure to infer, or pass --structure_path."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find transcripts by FIBS + year(s) and run RAG section + prediction pipeline"
    )
    parser.add_argument(
        "--fibs",
        required=True,
        help="Numeric localgov FIBS folder id, e.g. 00674 or 674 (maps to .../{year}/{fibs}/transcripts/)",
    )
    year_group = parser.add_mutually_exclusive_group()
    year_group.add_argument("--year", default=None, help="Single year only, e.g. 2023")
    year_group.add_argument(
        "--year_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help=f"Year range inclusive, e.g. 2020 2026 (default when --year omitted: {DEFAULT_YEAR_START} {DEFAULT_YEAR_END})",
    )
    parser.add_argument("--year_start", type=int, default=DEFAULT_YEAR_START, help="Range start if --year not set")
    parser.add_argument("--year_end", type=int, default=DEFAULT_YEAR_END, help="Range end if --year not set")
    parser.add_argument(
        "--localgov_root",
        default=os.getenv("LOCALGOV_ROOT", os.getenv("TRANSCRIPTS_ROOT", DEFAULT_LOCALGOV_ROOT)),
        help="localgov root; transcripts at {root}/{year}/{fibs}/transcripts/",
    )
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_GPT_MODEL),
        help="OpenAI model name (default: gpt-4o-2024-11-20)",
    )
    parser.add_argument("--minutes_dir", default=None)
    parser.add_argument("--eval_path", default=None)
    parser.add_argument("--transcript_files", nargs="*", default=None)
    parser.add_argument("--list_only", action="store_true")
    parser.add_argument(
        "--skip_structure",
        action="store_true",
        help="Reuse existing structure: exact tag file, or filter from a combined {fibs}_*_meeting_structure.json",
    )
    parser.add_argument(
        "--structure_path",
        default=None,
        help="Explicit meeting_structure.json path; filtered to loaded meetings (skips Step 1)",
    )
    parser.add_argument("--skip_classify", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("RAG_WORKERS", "1")),
        help="Concurrent LLM calls (threads) for structure extraction and remark classification",
    )
    args = parser.parse_args()

    if args.structure_path and not os.path.isfile(args.structure_path):
        parser.error(f"--structure_path not found: {args.structure_path}")

    if args.year_range:
        args.year_start, args.year_end = args.year_range

    from pageindex.transcript_loader import (
        discover_transcript_files,
        format_fibs_for_key,
        load_meetings_for_fibs_year,
        load_meetings_for_fibs_years,
        normalize_fibs,
        resolve_transcripts_dir,
    )

    fibs_id = normalize_fibs(args.fibs)
    if not fibs_id.isdigit():
        parser.error(f"--fibs must be a numeric localgov id (e.g. 00674), got {args.fibs!r}")
    fibs_tag = format_fibs_for_key(fibs_id)
    years, year_tag = _parse_years(args)
    tag = f"{fibs_tag}_{year_tag}"

    if args.list_only:
        total = 0
        for year in years:
            try:
                transcripts_dir = resolve_transcripts_dir(args.localgov_root, fibs_id, year)
            except FileNotFoundError:
                print(f"[{year}] directory missing, skip")
                continue
            paths = discover_transcript_files(args.localgov_root, fibs_id, year)
            print(f"[{year}] {transcripts_dir} -> {len(paths)} file(s)")
            for p in paths:
                print(f"  {p}")
            total += len(paths)
        if total == 0:
            print(f"No transcript files for FIBS={fibs_tag} years={years[0]}-{years[-1]}")
            sys.exit(1)
        print(f"Total: {total} file(s)")
        return

    print(f"LLM_BACKEND={os.getenv('LLM_BACKEND')} model={args.model}")

    if len(years) == 1 and not args.transcript_files:
        transcripts_dir = resolve_transcripts_dir(args.localgov_root, fibs_id, years[0])
        print(f"Loading FIBS={fibs_tag} year={years[0]} from {transcripts_dir}")
        meeting_data, sources = load_meetings_for_fibs_year(args.localgov_root, fibs_id, years[0])
        years_loaded = years
    else:
        print(f"Loading FIBS={fibs_tag} years={years[0]}-{years[-1]} from {args.localgov_root}")
        meeting_data, sources, years_loaded = load_meetings_for_fibs_years(
            args.localgov_root, fibs_id, years, file_paths=args.transcript_files
        )
        if not years_loaded:
            raise FileNotFoundError(
                f"No transcript data for FIBS={fibs_tag} in years {years[0]}-{years[-1]}. "
                f"Try --list_only to see which years exist."
            )
        print(f"  Years with data: {years_loaded}")

    print(f"  Loaded {len(meeting_data)} meeting(s) from {len(sources)} file(s)")
    for k, utts in meeting_data.items():
        print(f"    {k}: {len(utts)} utterances")

    os.makedirs(args.results_dir, exist_ok=True)
    meeting_path = os.path.join(args.results_dir, f"{tag}_from_transcripts.json")
    with open(meeting_path, "w", encoding="utf-8") as f:
        json.dump(meeting_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved meeting JSON -> {meeting_path}")

    structure_path = os.path.join(args.results_dir, f"{tag}_meeting_structure.json")
    from pageindex.meeting_structure import extract_public_comment_by_section

    structure, _structure_source = _resolve_structure(
        skip_structure=args.skip_structure,
        structure_path=args.structure_path,
        fibs_tag=fibs_tag,
        tag=tag,
        meeting_data=meeting_data,
        results_dir=args.results_dir,
    )

    if structure is None:
        print("\nStep 1: Extract meeting structure (GPT)...")
        structure, meeting_with_labels, public_comment_extract = extract_public_comment_by_section(
            meeting_path=meeting_path,
            model=args.model,
            minutes_dir=args.minutes_dir,
            workers=args.workers,
        )
        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        print(f"  Saved structure -> {structure_path}")

        labels_path = os.path.join(args.results_dir, f"{tag}_meeting_with_sections.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(meeting_with_labels, f, indent=2, ensure_ascii=False)
        extract_path = os.path.join(args.results_dir, f"{tag}_public_comment_by_section.json")
        with open(extract_path, "w", encoding="utf-8") as f:
            json.dump(public_comment_extract, f, indent=2, ensure_ascii=False)

    print("\nStep 2: Sections per meeting:")
    for meeting_date, stages in structure.items():
        print(f"  {meeting_date}: {len(stages)} sections")

    if args.skip_classify:
        print("\nSkipping classification. Done.")
        return

    predict_data = meeting_data
    if args.eval_path:
        with open(args.eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        predict_data = {k: (eval_data[k] if k in eval_data else v) for k, v in meeting_data.items()}
        print(f"\nEval labels from: {args.eval_path}")

    print("\nStep 3: Within-section remark prediction (Public Comment + Public Hearing, GPT)...")
    predictions, metrics_by_section = run_predictions(
        predict_data, structure, args.model, workers=args.workers
    )

    pred_path = os.path.join(args.results_dir, f"{tag}_within_section_predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "predictions": predictions,
                "metrics_by_section": metrics_by_section,
                "years_loaded": years_loaded,
            },
            f,
            indent=2,
        )
    print(f"  Saved predictions -> {pred_path}")

    any_metrics = False
    for section, metrics in metrics_by_section.items():
        n_pred = sum(1 for p in predictions if p["section"] == section)
        if metrics:
            any_metrics = True
            print(
                f"\n[{section}] acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                f"n={metrics['support']} (utterances scored: {n_pred})"
            )
        else:
            print(f"\n[{section}] predictions only (no ground truth); utterances scored: {n_pred}")
    if not any_metrics:
        print("\n(No ground-truth labels found; predictions only.)")

    print("\nDone.")


if __name__ == "__main__":
    main()
