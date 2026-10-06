"""
Load meeting JSON (same format as RAG/data_loader) and convert to page_list for IndexRag.
"""
import json
from typing import Dict, List, Any, Tuple

try:
    from .utils import count_tokens
except ImportError:
    from utils import count_tokens


def load_meeting_json(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Same format as RAG: { meeting_date: [ {text, speaker, ...}, ... ] }."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def meeting_to_page_list(
    meeting_data: Dict[str, List[Dict[str, Any]]],
    model: str = "gpt-4o-2024-11-20",
    segment_by: str = "utterance",
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Convert meeting data to (page_list, segment_to_utterance).
    - page_list: [(segment_text, token_count), ...] for use with process_no_toc etc.
    - segment_to_utterance: for each segment index i, (meeting_date, utterance_index_in_that_date).
    segment_by: "utterance" = one segment per utterance; "section" = merge by Meeting Section.
    """
    page_contents = []
    segment_to_utterance: List[Tuple[str, int]] = []

    if segment_by == "utterance":
        for meeting_date, items in meeting_data.items():
            for utt_idx, item in enumerate(items):
                text = item.get("text", "").strip()
                if not text:
                    text = "(empty)"
                speaker = item.get("speaker", "")
                section = item.get("Meeting Section", "")
                line = f"[{meeting_date}] "
                if section:
                    line += f"Section: {section}. "
                if speaker:
                    line += f"Speaker: {speaker}. "
                line += text
                page_contents.append(line)
                segment_to_utterance.append((meeting_date, utt_idx))
    else:
        current_section = None
        current_chunk: List[str] = []
        current_refs: List[Tuple[str, int]] = []
        for meeting_date, items in meeting_data.items():
            for utt_idx, item in enumerate(items):
                text = item.get("text", "").strip()
                section = item.get("Meeting Section", "")
                speaker = item.get("speaker", "")
                if section and section != current_section:
                    if current_chunk:
                        page_contents.append("\n".join(current_chunk))
                        segment_to_utterance.append(current_refs[0] if current_refs else (meeting_date, utt_idx))
                    current_section = section
                    current_chunk = []
                    current_refs = []
                line = f"Speaker: {speaker}. {text}" if speaker else text
                if text:
                    current_chunk.append(line)
                    current_refs.append((meeting_date, utt_idx))
        if current_chunk:
            page_contents.append("\n".join(current_chunk))
            segment_to_utterance.append(current_refs[0] if current_refs else ("", 0))

    token_lengths = [count_tokens(t, model=model) for t in page_contents]
    page_list = [(text, tok) for text, tok in zip(page_contents, token_lengths)]
    return page_list, segment_to_utterance


def flatten_meeting_to_utterances(
    meeting_data: Dict[str, List[Dict[str, Any]]]
) -> List[Tuple[str, int, Dict[str, Any]]]:
    """(meeting_date, utt_index, utterance_dict) for each utterance."""
    out = []
    for meeting_date, items in meeting_data.items():
        for utt_idx, item in enumerate(items):
            out.append((meeting_date, utt_idx, item))
    return out


def unflatten_with_predictions(
    flat: List[Tuple[str, int, Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Rebuild { meeting_date: [ utterances with is_public_comment/hearing ] }."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for meeting_date, utt_idx, item in flat:
        if meeting_date not in result:
            result[meeting_date] = []
        while len(result[meeting_date]) <= utt_idx:
            result[meeting_date].append({})
        result[meeting_date][utt_idx] = item
    return result
