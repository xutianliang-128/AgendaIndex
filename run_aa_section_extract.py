"""
Pipeline for AA_test.json:
  1. Extract meeting structure (sections: Roll Call, Presentation, Public Comment, etc.) via GPT.
  2. Read out all sections per meeting.
  3. Extract utterances in Public Comment / Public Hearing sections and save.
"""
import argparse
import json
import os

# AA_test.json is at workspace root; script is in IndexRag/
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MEETING_PATH = os.path.join(WORKSPACE_ROOT, "AA_test.json")
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    parser = argparse.ArgumentParser(description="Extract meeting sections then public comment by section (AA_test.json)")
    parser.add_argument("--meeting_path", type=str, default=DEFAULT_MEETING_PATH, help="Path to meeting JSON (default: AA_test.json)")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20", help="Model for structure extraction")
    parser.add_argument(
        "--minutes_dir",
        type=str,
        default=None,
        help="Optional folder of official minutes: <meeting_date>.txt or .md (same keys as JSON). Injected as prompt context for stage detection only.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.meeting_path):
        raise FileNotFoundError(f"Meeting file not found: {args.meeting_path}")

    os.makedirs(args.results_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.meeting_path))[0]

    from pageindex.meeting_structure import extract_public_comment_by_section

    print("Step 1: Extracting meeting structure (sections)...")
    structure, meeting_with_labels, public_comment_extract = extract_public_comment_by_section(
        meeting_path=args.meeting_path,
        model=args.model,
        minutes_dir=args.minutes_dir,
    )

    # Save 1: structure (sections per meeting)
    structure_path = os.path.join(args.results_dir, f"{base}_meeting_structure.json")
    with open(structure_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
    print(f"  Saved structure -> {structure_path}")

    # Read out sections (print + optional save)
    print("\nStep 2: Sections per meeting:")
    for meeting_date, stages in structure.items():
        print(f"  {meeting_date}: {len(stages)} sections")
        for s in stages:
            name = s.get("stage_name", "")
            start = s.get("start_index", 0)
            end = s.get("end_index", start)
            count = end - start + 1
            print(f"    - {name}: utterances [{start}, {end}] ({count} utterances)")

    # Save 2: full meeting with section labels (is_public_comment_section, is_public_hearing_section)
    labels_path = os.path.join(args.results_dir, f"{base}_meeting_with_sections.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(meeting_with_labels, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved meeting with section labels -> {labels_path}")

    # Save 3: only Public Comment / Public Hearing utterances
    extract_path = os.path.join(args.results_dir, f"{base}_public_comment_by_section.json")
    with open(extract_path, "w", encoding="utf-8") as f:
        json.dump(public_comment_extract, f, indent=2, ensure_ascii=False)
    n_total = sum(len(utts) for utts in public_comment_extract.values())
    print(f"  Saved public comment/hearing utterances -> {extract_path} (total {n_total} utterances)")
    print("\nDone.")


if __name__ == "__main__":
    main()
