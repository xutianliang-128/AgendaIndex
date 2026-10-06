"""
AgendaIndex: meeting-only entry. Extract meeting structure or public comment/hearing from meeting JSON.
"""
import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract meeting structure or public comments from meeting JSON (AgendaIndex: meeting only)")
    parser.add_argument("--meeting_path", type=str, required=True, help="Path to meeting JSON (RAG format)")
    parser.add_argument("--task", type=str, default="public_comment", choices=["public_comment", "structure"],
                        help="structure=extract meeting stages; public_comment=extract public comment/hearing labels")
    parser.add_argument("--extract_mode", type=str, default="direct", choices=["direct", "tree"],
                        help="For task=public_comment: direct=classify each utterance; tree=build segment tree then label")
    parser.add_argument("--output", type=str, help="Output path (default: results/<basename>_public_comment.json or _meeting_structure.json)")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20", help="Model to use")
    args = parser.parse_args()

    if not os.path.isfile(args.meeting_path):
        raise ValueError(f"Meeting file not found: {args.meeting_path}")

    base = os.path.splitext(os.path.basename(args.meeting_path))[0]
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    if args.task == "structure":
        from agendaindex.meeting_structure import extract_meeting_structure
        output_path = args.output or os.path.join(output_dir, f"{base}_meeting_structure.json")
        print("Extracting meeting structure (Roll Call, Presentation, Public Comment, etc.)...")
        result = extract_meeting_structure(meeting_path=args.meeting_path, model=args.model)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        for date, stages in result.items():
            print(f"  {date}: {len(stages)} stages -> {[s.get('stage_name') for s in stages]}")
        print(f"Done. Saved to: {output_path}")
    else:
        from agendaindex.extract_public_comment import extract_public_comment
        output_path = args.output or os.path.join(output_dir, f"{base}_public_comment.json")
        print("Extracting public comments...")
        result = extract_public_comment(
            meeting_path=args.meeting_path,
            model=args.model,
            mode=args.extract_mode,
            output_path=output_path,
        )
        n_comment = sum(1 for date_utts in result.values() for u in date_utts if u.get("is_public_comment"))
        n_hearing = sum(1 for date_utts in result.values() for u in date_utts if u.get("is_public_hearing"))
        print(f"Done. Public comments: {n_comment}, Public hearings: {n_hearing}. Saved to: {output_path}")
