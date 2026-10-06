"""
Run full pipeline (section extraction + within-section classification) on all *_test.json
in workspace root, then evaluate and output a summary table (CSV + Markdown).
"""
import argparse
import json
import os
import sys

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


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
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def get_test_files(workspace_root: str):
    """All *_test.json in workspace root only (no subdirs)."""
    out = []
    for name in os.listdir(workspace_root):
        if name.endswith("_test.json") and os.path.isfile(os.path.join(workspace_root, name)):
            out.append(os.path.join(workspace_root, name))
    return sorted(out)


def run_one(meeting_path: str, results_dir: str, model: str, skip_structure: bool = False):
    """Run section extraction (unless skip_structure) then within-section classify; return metrics or None on error."""
    base = os.path.splitext(os.path.basename(meeting_path))[0]
    structure_path = os.path.join(results_dir, f"{base}_meeting_structure.json")

    from pageindex.meeting_structure import extract_public_comment_by_section, classify_utterance_public_comment

    if not skip_structure:
        try:
            structure, _, _ = extract_public_comment_by_section(meeting_path=meeting_path, model=model)
            os.makedirs(results_dir, exist_ok=True)
            with open(structure_path, "w", encoding="utf-8") as f:
                json.dump(structure, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [{base}] Section extraction failed: {e}", file=sys.stderr)
            return None

    if not os.path.isfile(structure_path):
        print(f"  [{base}] Structure not found: {structure_path}", file=sys.stderr)
        return None

    with open(meeting_path, "r", encoding="utf-8") as f:
        meeting_data = json.load(f)
    with open(structure_path, "r", encoding="utf-8") as f:
        structure = json.load(f)

    y_true, y_pred = [], []
    for meeting_date, utterances in meeting_data.items():
        stages = structure.get(meeting_date, [])
        comment_indices = _indices_in_stage(stages, "Public Comment")
        for idx in comment_indices:
            if idx >= len(utterances):
                continue
            u = utterances[idx]
            speaker = u.get("speaker", "")
            text = (u.get("text") or "").strip()
            start = u.get("start")
            end = u.get("end")
            gt = 1 if u.get("is_public_comment", 0) in [1, "1", True] else 0
            pred = classify_utterance_public_comment(speaker=speaker, text=text, start=start, end=end, model=model)
            y_true.append(gt)
            y_pred.append(pred)

    if not y_true:
        print(f"  [{base}] No utterances in Public Comment section.", file=sys.stderr)
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}

    return compute_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Run pipeline on all *_test.json and output evaluation table")
    parser.add_argument("--workspace_root", type=str, default=WORKSPACE_ROOT, help="Directory containing *_test.json")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR, help="Where to save structures and tables")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20", help="Model name")
    parser.add_argument("--skip_structure", action="store_true", help="Skip section extraction; use existing *_meeting_structure.json")
    args = parser.parse_args()

    test_files = get_test_files(args.workspace_root)
    if not test_files:
        print("No *_test.json found in", args.workspace_root, file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(test_files)} test files: {[os.path.basename(p) for p in test_files]}")
    os.makedirs(args.results_dir, exist_ok=True)

    rows = []
    for path in test_files:
        base = os.path.splitext(os.path.basename(path))[0]
        dataset = base.replace("_test", "")
        print(f"Running {dataset}...")
        m = run_one(path, args.results_dir, args.model, skip_structure=args.skip_structure)
        if m is None:
            rows.append({
                "dataset": dataset,
                "accuracy": "",
                "precision": "",
                "recall": "",
                "f1": "",
                "support": "",
                "TP": "", "TN": "", "FP": "", "FN": "",
            })
            continue
        rows.append({
            "dataset": dataset,
            "accuracy": f"{m['accuracy']:.4f}",
            "precision": f"{m['precision']:.4f}",
            "recall": f"{m['recall']:.4f}",
            "f1": f"{m['f1']:.4f}",
            "support": str(m["support"]),
            "TP": str(m["tp"]), "TN": str(m["tn"]), "FP": str(m["fp"]), "FN": str(m["fn"]),
        })

    csv_path = os.path.join(args.results_dir, "all_tests_evaluation.csv")
    cols = ["dataset", "accuracy", "precision", "recall", "f1", "support", "TP", "TN", "FP", "FN"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nSaved CSV -> {csv_path}")

    md_path = os.path.join(args.results_dir, "all_tests_evaluation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
    print(f"Saved Markdown table -> {md_path}")

    print("\n--- Summary table ---")
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    print("\nDone.")


if __name__ == "__main__":
    main()
