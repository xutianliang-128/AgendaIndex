"""
Extract public comments / public hearings from meeting JSON.
- direct: classify each utterance with LLM. Contract: only use time, text, and speaker_id as input.
- tree: build segment tree over meeting, ask LLM which nodes are public comment/hearing, map back to utterances.
"""
import json
import asyncio
from typing import Dict, List, Any, Optional

try:
    from .utils import ChatGPT_API, extract_json, structure_to_list
    from .input_meeting import (
        load_meeting_json,
        meeting_to_page_list,
        flatten_meeting_to_utterances,
        unflatten_with_predictions,
    )
except ImportError:
    from utils import ChatGPT_API, extract_json, structure_to_list
    from input_meeting import (
        load_meeting_json,
        meeting_to_page_list,
        flatten_meeting_to_utterances,
        unflatten_with_predictions,
    )


SYSTEM_PROMPT = (
    "You are a binary classifier for meeting transcripts. For each utterance, classify it separately for two categories: "
    "(1) is_public_comment: 1 if it is a public comment (citizen comment during public comment period), 0 otherwise; "
    "(2) is_public_hearing: 1 if it is a public hearing (formal hearing process), 0 otherwise. "
    "An utterance can be BOTH a public comment AND a public hearing. Return ONLY a valid JSON object with these two fields."
)


def _parse_classification(text: str) -> Dict[str, int]:
    out = extract_json(text) if isinstance(text, str) else {}
    return {
        "is_public_comment": 1 if out.get("is_public_comment", 0) in [1, "1", True] else 0,
        "is_public_hearing": 1 if out.get("is_public_hearing", 0) in [1, "1", True] else 0,
    }


def classify_one_utterance(
    text: str,
    model: str = "gpt-4o-2024-11-20",
    speaker_id: str = "",
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Dict[str, int]:
    """Allowed input only: time (start, end), text, speaker_id."""
    if not (text or "").strip():
        return {"is_public_comment": 0, "is_public_hearing": 0}
    prompt = (
        f"Utterance to classify (use only these fields):\n"
        f"- speaker_id: {speaker_id}\n- text: {text}\n- time: start={start} end={end}\n\n"
        f"Return JSON: {{ \"is_public_comment\": 0 or 1, \"is_public_hearing\": 0 or 1 }}"
    )
    response = ChatGPT_API(model=model, prompt=prompt)
    return _parse_classification(response)


def extract_public_comment_direct(
    meeting_data: Dict[str, List[Dict[str, Any]]],
    model: str = "gpt-4o-2024-11-20",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Classify each utterance with LLM. Returns same structure as input with is_public_comment, is_public_hearing set.
    """
    flat = flatten_meeting_to_utterances(meeting_data)
    for meeting_date, utt_idx, item in flat:
        text = item.get("text", "").strip()
        speaker_id = item.get("speaker_id") or item.get("speaker", "")
        start, end = item.get("start"), item.get("end")
        pred = classify_one_utterance(text, model=model, speaker_id=speaker_id, start=start, end=end)
        item["is_public_comment"] = pred.get("is_public_comment", 0)
        item["is_public_hearing"] = pred.get("is_public_hearing", 0)
        item["is_public"] = 1 if (pred.get("is_public_comment") or pred.get("is_public_hearing")) else 0
    return unflatten_with_predictions(flat)


def extract_public_comment_tree(
    meeting_data: Dict[str, List[Dict[str, Any]]],
    model: str = "gpt-4o-2024-11-20",
    opt: Optional[Any] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build tree over meeting segments (process_no_toc), ask LLM which nodes are public comment/hearing,
    then mark all utterances in those segments.
    """
    from .utils import ConfigLoader, write_node_id
    from .page_index import meta_processor, check_title_appearance_in_start_concurrent
    from .utils import post_processing

    if opt is None:
        opt = ConfigLoader().load({"model": model})

    class _NoOpLogger:
        def info(self, msg): pass
        def error(self, msg): pass
    logger = getattr(opt, "logger", None) or _NoOpLogger()

    page_list, _ = meeting_to_page_list(meeting_data, model=model, segment_by="utterance")
    if not page_list:
        return meeting_data
    toc_with_page_number = asyncio.run(
        meta_processor(page_list, mode="process_no_toc", start_index=1, opt=opt, logger=logger)
    )
    toc_with_page_number = asyncio.run(
        check_title_appearance_in_start_concurrent(toc_with_page_number, page_list, model=opt.model, logger=logger)
    )
    valid_toc_items = [item for item in toc_with_page_number if item.get("physical_index") is not None]
    toc_tree = post_processing(valid_toc_items, len(page_list))
    write_node_id(toc_tree)

    nodes_flat = structure_to_list(toc_tree)
    toc_for_prompt = []
    for n in nodes_flat:
        start = n.get("start_index")
        end = n.get("end_index")
        toc_for_prompt.append({
            "node_id": n.get("node_id", ""),
            "title": n.get("title", ""),
            "start_index": start,
            "end_index": end,
        })

    prompt = f"""You are given a meeting structure (segments with titles). Identify which segments are PUBLIC COMMENT or PUBLIC HEARING.
- Public comment: citizen comment during public comment period.
- Public hearing: formal hearing process.

Meeting structure (each segment has start_index and end_index; these are 1-based segment indices):
{json.dumps(toc_for_prompt, indent=2)}

Return a JSON object with two arrays of 1-based segment index ranges that correspond to public comment and public hearing:
{{ "public_comment_ranges": [ [start_index, end_index], ... ], "public_hearing_ranges": [ [start_index, end_index], ... ] }}
If a segment is both, include its range in both arrays. Only include segments that are clearly public comment or public hearing."""

    response = ChatGPT_API(model=opt.model, prompt=prompt)
    out = extract_json(response)
    comment_ranges = out.get("public_comment_ranges") or []
    hearing_ranges = out.get("public_hearing_ranges") or []

    def in_range(seg_idx: int, ranges: List[List[int]]) -> bool:
        for r in ranges:
            if len(r) >= 2 and r[0] <= seg_idx <= r[1]:
                return True
        return False

    flat = flatten_meeting_to_utterances(meeting_data)
    for seg_idx, (_, _, item) in enumerate(flat):
        one_indexed = seg_idx + 1
        item["is_public_comment"] = 1 if in_range(one_indexed, comment_ranges) else 0
        item["is_public_hearing"] = 1 if in_range(one_indexed, hearing_ranges) else 0
        item["is_public"] = 1 if (item["is_public_comment"] or item["is_public_hearing"]) else 0

    return unflatten_with_predictions(flat)


def extract_public_comment(
    meeting_path: str,
    model: str = "gpt-4o-2024-11-20",
    mode: str = "direct",
    output_path: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    meeting_path: path to meeting JSON (RAG format).
    mode: "direct" (classify each utterance) or "tree" (build tree, then label segments).
    output_path: if set, write result JSON here.
    """
    meeting_data = load_meeting_json(meeting_path)
    if mode == "tree":
        result = extract_public_comment_tree(meeting_data, model=model)
    else:
        result = extract_public_comment_direct(meeting_data, model=model)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return result
