"""
Within Public Comment section only: classify each utterance as public comment (1) or not (0)
using ONLY speaker, text, start, end. Compare with ground truth from AA_test.json and output metrics.
"""
import argparse
import json
import sys
import os

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MEETING_PATH = os.path.join(WORKSPACE_ROOT, "AA_test.json")
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _default_structure_path(meeting_path: str) -> str:
    base = os.path.splitext(os.path.basename(meeting_path))[0]
    return os.path.join(DEFAULT_RESULTS_DIR, f"{base}_meeting_structure.json")


def _indices_in_stage(stages: list, stage_name: str) -> list:
    """Indices across ALL stages matching stage_name (repeated/aliased public windows)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from agendaindex.section_taxonomy import indices_for_section

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
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
    }


def main():
    parser = argparse.ArgumentParser(description="Classify within Public Comment section using only speaker, text, start, end")
    parser.add_argument("--meeting_path", type=str, default=DEFAULT_MEETING_PATH, help="Path to meeting JSON (for ground truth)")
    parser.add_argument("--structure_path", type=str, default=None, help="Path to meeting_structure.json (default: results/<basename>_meeting_structure.json)")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20", help="Model for classification")
    args = parser.parse_args()

    if not os.path.isfile(args.meeting_path):
        raise FileNotFoundError(f"Meeting file not found: {args.meeting_path}")
    structure_path = args.structure_path or _default_structure_path(args.meeting_path)
    if not os.path.isfile(structure_path):
        raise FileNotFoundError(f"Structure file not found: {structure_path}. Run run_aa_section_extract.py --meeting_path <same> first.")

    os.makedirs(args.results_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.meeting_path))[0]

    with open(args.meeting_path, "r", encoding="utf-8") as f:
        meeting_data = json.load(f)
    with open(structure_path, "r", encoding="utf-8") as f:
        structure = json.load(f)

    from agendaindex.meeting_structure import classify_utterance_public_comment

    all_results = []
    y_true = []
    y_pred = []

    for meeting_date, utterances in meeting_data.items():
        stages = structure.get(meeting_date, [])
        comment_indices = _indices_in_stage(stages, "Public Comment")
        if not comment_indices:
            continue
        for idx in comment_indices:
            if idx >= len(utterances):
                continue
            u = utterances[idx]
            speaker = u.get("speaker", "")
            text = (u.get("text") or "").strip()
            start = u.get("start")
            end = u.get("end")
            gt = 1 if u.get("is_public_comment", 0) in [1, "1", True] else 0
            pred = classify_utterance_public_comment(speaker=speaker, text=text, start=start, end=end, model=args.model)
            y_true.append(gt)
            y_pred.append(pred)
            all_results.append({
                "meeting_date": meeting_date,
                "utterance_index": idx,
                "speaker": speaker,
                "text": text[:200] + "..." if len(text) > 200 else text,
                "start": start,
                "end": end,
                "ground_truth_is_public_comment": gt,
                "predicted_is_public_comment": pred,
            })

    metrics = compute_metrics(y_true, y_pred)

    out_path = os.path.join(args.results_dir, f"{base}_within_section_predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": all_results, "metrics": metrics}, f, indent=2, ensure_ascii=False)
    print(f"Saved predictions -> {out_path}")

    metrics_path = os.path.join(args.results_dir, f"{base}_within_section_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")

    print("\n--- Metrics (within Public Comment section, predict using only speaker, text, start, end) ---")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Support:   {metrics['support']}")
    print(f"  Confusion: TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
