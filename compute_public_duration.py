"""
Compute average duration of Public Comment and Public Hearing sections.
- pred: section boundaries from meeting_structure.json (GPT).
- gt: section boundaries from "Meeting Section" per utterance (Public Comment / Public Hearing).
"""
import argparse
import json
import os

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _trigger(u: dict, key: str) -> bool:
    return u.get(key, 0) in [1, "1", True]


def _is_public_comment_stage(stage_name: str) -> bool:
    n = (stage_name or "").lower()
    return "public comment" in n and "hearing" not in n


def _is_public_hearing_stage(stage_name: str) -> bool:
    n = (stage_name or "").lower()
    return "public hearing" in n


def section_duration_sec(utterances: list, start_index: int, end_index: int) -> float | None:
    """Duration of section from first utterance start to last utterance end (seconds)."""
    if not utterances or start_index < 0 or end_index >= len(utterances):
        return None
    first = utterances[start_index]
    last = utterances[end_index]
    s = first.get("start")
    e = last.get("end")
    if s is None or e is None:
        return None
    try:
        return float(e) - float(s)
    except (TypeError, ValueError):
        return None


def _in_public_comment_section(ms: str) -> bool:
    """Meeting Section is Public Comment (and not Public Hearing)."""
    if not ms:
        return False
    ms = ms.strip().lower()
    return "public comment" in ms and "hearing" not in ms


def _in_public_hearing_section(ms: str) -> bool:
    """Meeting Section is Public Hearing."""
    if not ms:
        return False
    return "public hearing" in ms.strip().lower()


def collect_durations_by_meeting_section(meeting_data: dict):
    """
    Ground truth: section boundaries from "Meeting Section" per utterance.
    Consecutive utterances with Meeting Section == "Public Comment" (or "Public Hearing")
    form one section. Duration = first.start to last.end in each contiguous block.
    """
    comment_durations = []
    hearing_durations = []

    for meeting_date, utterances in meeting_data.items():
        if not utterances:
            continue
        i = 0
        while i < len(utterances):
            ms = (utterances[i].get("Meeting Section") or "").strip()
            if _in_public_comment_section(ms):
                start_i = i
                while i < len(utterances) and _in_public_comment_section(
                    (utterances[i].get("Meeting Section") or "").strip()
                ):
                    i += 1
                end_i = i - 1
                dur = section_duration_sec(utterances, start_i, end_i)
                if dur is not None and dur >= 0:
                    comment_durations.append((meeting_date, start_i, end_i, dur))
                continue
            if _in_public_hearing_section(ms):
                start_i = i
                while i < len(utterances) and _in_public_hearing_section(
                    (utterances[i].get("Meeting Section") or "").strip()
                ):
                    i += 1
                end_i = i - 1
                dur = section_duration_sec(utterances, start_i, end_i)
                if dur is not None and dur >= 0:
                    hearing_durations.append((meeting_date, start_i, end_i, dur))
                continue
            i += 1

    return comment_durations, hearing_durations


def collect_durations_pred(meeting_data: dict, structure: dict):
    """Pred: section boundaries from meeting_structure.json (GPT)."""
    comment_durations = []
    hearing_durations = []

    for meeting_date, utterances in meeting_data.items():
        stages = structure.get(meeting_date, [])
        for s in stages:
            name = s.get("stage_name", "")
            start_i = s.get("start_index", 0)
            end_i = s.get("end_index", start_i)
            dur = section_duration_sec(utterances, start_i, end_i)
            if dur is None or dur < 0:
                continue
            if _is_public_comment_stage(name):
                comment_durations.append((meeting_date, start_i, end_i, dur))
            elif _is_public_hearing_stage(name):
                hearing_durations.append((meeting_date, start_i, end_i, dur))

    return comment_durations, hearing_durations


def fmt_sec(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:05.2f}"


def main():
    parser = argparse.ArgumentParser(description="Average duration of Public Comment and Public Hearing sections")
    parser.add_argument("--workspace_root", type=str, default=WORKSPACE_ROOT, help="Directory containing *_test.json")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory with *_meeting_structure.json (for pred)")
    parser.add_argument("--use_gt", action="store_true", help="Use ground truth: Meeting Section (Public Comment / Public Hearing) per utterance")
    parser.add_argument("--out_tsv", type=str, default=None, help="Write TSV table to this path (e.g. results/duration_gt_sheet.tsv)")
    args = parser.parse_args()

    use_gt = args.use_gt
    all_comment = []
    all_hearing = []
    by_city = {}  # city -> {"comment": [sec,...], "hearing": [sec,...]}

    for name in sorted(os.listdir(args.workspace_root)):
        if not name.endswith("_test.json"):
            continue
        meeting_path = os.path.join(args.workspace_root, name)
        if not os.path.isfile(meeting_path):
            continue
        base = os.path.splitext(name)[0]  # e.g. AA_test
        city = base.replace("_test", "") if base.endswith("_test") else base
        with open(meeting_path, "r", encoding="utf-8") as f:
            meeting_data = json.load(f)

        if use_gt:
            comment_durs, hearing_durs = collect_durations_by_meeting_section(meeting_data)
        else:
            structure_path = os.path.join(args.results_dir, f"{base}_meeting_structure.json")
            if not os.path.isfile(structure_path):
                continue
            with open(structure_path, "r", encoding="utf-8") as f:
                structure = json.load(f)
            comment_durs, hearing_durs = collect_durations_pred(meeting_data, structure)

        c_list = [d[3] for d in comment_durs]
        h_list = [d[3] for d in hearing_durs]
        all_comment.extend(c_list)
        all_hearing.extend(h_list)
        by_city[city] = {"comment": c_list, "hearing": h_list}

    def stats(label: str, durations: list):
        if not durations:
            print(f"  {label}: no sections found")
            return
        n = len(durations)
        total = sum(durations)
        avg = total / n
        print(f"  {label}:")
        print(f"    sections: {n}")
        print(f"    total:    {fmt_sec(total)} ({total:.1f} sec)")
        print(f"    average: {fmt_sec(avg)} ({avg:.1f} sec)")

    source_label = "Ground Truth (Meeting Section)" if use_gt else "Pred (meeting_structure)"
    print(f"Source: {source_label}\n")
    print("=== Total ===\n")
    print("Average duration of public comments and public hearings (by section):")
    print()
    stats("Public Comment", all_comment)
    print()
    stats("Public Hearing", all_hearing)

    print("\n" + "=" * 88)
    print("=== By City ===\n")
    # Table: city | Public Comment (n, total, avg) | Public Hearing (n, total, avg)
    print(f"{'City':<6} | {'Public Comment':^36} | {'Public Hearing':^36}")
    print(f"       | {'n':>4} {'total':>10} {'avg':>10} | {'n':>4} {'total':>10} {'avg':>10}")
    print("-" * 88)
    for city in sorted(by_city.keys()):
        row = by_city[city]
        c = row["comment"]
        h = row["hearing"]
        c_n, c_tot, c_avg = (len(c), sum(c), sum(c) / len(c)) if c else (0, 0.0, 0.0)
        h_n, h_tot, h_avg = (len(h), sum(h), sum(h) / len(h)) if h else (0, 0.0, 0.0)
        c_str = f"{c_n:>4} {fmt_sec(c_tot):>10} {fmt_sec(c_avg):>10}" if c else "  -       -       -"
        h_str = f"{h_n:>4} {fmt_sec(h_tot):>10} {fmt_sec(h_avg):>10}" if h else "  -       -       -"
        print(f"{city:<6} | {c_str} | {h_str}")

    # Write TSV for Google Sheet
    if args.out_tsv:
        lines = [
            f"(Section boundaries: {source_label})",
            "Total (All Cities)",
            "Stage\t# Sections\tTotal Duration\tAverage Duration",
            f"Public Comment\t{len(all_comment)}\t{fmt_sec(sum(all_comment))}\t{fmt_sec(sum(all_comment) / len(all_comment)) if all_comment else '-'}",
            f"Public Hearing\t{len(all_hearing)}\t{fmt_sec(sum(all_hearing))}\t{fmt_sec(sum(all_hearing) / len(all_hearing)) if all_hearing else '-'}",
            "",
            "By City",
            "City\tPublic Comment #\tPublic Comment Total\tPublic Comment Avg\tPublic Hearing #\tPublic Hearing Total\tPublic Hearing Avg",
        ]
        for city in sorted(by_city.keys()):
            row = by_city[city]
            c, h = row["comment"], row["hearing"]
            c_n, c_tot, c_avg = (len(c), fmt_sec(sum(c)), fmt_sec(sum(c) / len(c))) if c else (0, "", "")
            h_n, h_tot, h_avg = (len(h), fmt_sec(sum(h)), fmt_sec(sum(h) / len(h))) if h else ("", "", "")
            lines.append(f"{city}\t{c_n}\t{c_tot}\t{c_avg}\t{h_n}\t{h_tot}\t{h_avg}")
        os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
        with open(args.out_tsv, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nWrote TSV -> {args.out_tsv}")


if __name__ == "__main__":
    main()
