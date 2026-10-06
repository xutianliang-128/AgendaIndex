"""
Extract city council meeting structure: stages like Roll Call, Presentation, Public Comment, Public Hearing, etc.
Returns each stage with start/end utterance indices (0-based).

Contract for stage boundaries: the numbered transcript lines use only time (start/end), text, and speaker_id.
Optional OFFICIAL MINUTES may be appended as global context (hints); indices must still align to the transcript.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional

try:
    from .utils import ChatGPT_API, ChatGPT_API_JSON, extract_json, count_tokens
    from .input_meeting import load_meeting_json
    from .section_taxonomy import indices_for_section, public_section_indices
    from .minutes_context import (
        load_minutes_for_meetings,
        minutes_prompt_block,
        truncate_minutes,
    )
except ImportError:
    from utils import ChatGPT_API, ChatGPT_API_JSON, extract_json, count_tokens
    from input_meeting import load_meeting_json
    from section_taxonomy import indices_for_section, public_section_indices
    from minutes_context import (
        load_minutes_for_meetings,
        minutes_prompt_block,
        truncate_minutes,
    )


COMMON_STAGES = [
    "Roll Call",
    "Pledge of Allegiance",
    "Approval of Agenda",
    "Presentation",
    "Public Comment",
    "Public Hearing",
    "Discussion",
    "Motion / Vote",
    "Adjournment",
]

STAGE_LIST_JSON_SCHEMA = {
    "title": "MeetingStages",
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "stage_name": {"type": "string", "minLength": 1},
            "start_index": {"type": "integer"},
            "end_index": {"type": "integer"},
        },
        "required": ["stage_name", "start_index", "end_index"],
        "additionalProperties": False,
    },
}


def _build_numbered_transcript(utterances: List[Dict[str, Any]], start_index_offset: int = 0) -> str:
    """
    Build transcript text for the LLM.
    Only the allowed fields are used: time (start/end), text, and speaker_id.
    Per-line format: index: [start-end] [speaker_id]: text
    """
    lines = []
    for i, item in enumerate(utterances):
        global_i = i + start_index_offset
        # Some annotated dumps store numeric tokens as int in `text`.
        text = str(item.get("text") if item.get("text") is not None else "").strip()
        if not text:
            text = "(empty)"
        speaker_id = item.get("speaker_id") or item.get("speaker", "")
        start = item.get("start")
        end = item.get("end")
        time_str = ""
        if start is not None and end is not None:
            time_str = f"[{start}-{end}s] "
        elif start is not None:
            time_str = f"[{start}s] "
        parts = [f"{global_i}:", time_str]
        if speaker_id:
            parts.append(f"[{speaker_id}]:")
        parts.append(text)
        lines.append(" ".join(p for p in parts if p))
    return "\n".join(lines)


def _normalize_stage_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


def _is_natural_boundary(item: Dict[str, Any]) -> bool:
    """
    Prefer chunk boundaries near agenda-transition utterances to reduce stage split errors.
    """
    text = ((item.get("text") or "")[:300]).lower()
    cues = [
        "next item",
        "next agenda",
        "agenda item",
        "public comment",
        "public hearing",
        "hearing",
        "motion",
        "second",
        "vote",
        "adjourn",
        "roll call",
        "presentation",
    ]
    return any(c in text for c in cues)


def _chunk_utterance_ranges(
    utterances: List[Dict[str, Any]],
    model: str,
    transcript_token_budget: int,
    overlap_utterances: int,
) -> List[Dict[str, int]]:
    """
    Build chunk ranges by transcript token budget.
    Returns list of dicts: {start, end, emit_from}.
      - start/end: inclusive chunk range sent to LLM
      - emit_from: only keep stage output with end_index >= emit_from (avoid overlap duplication)
    """
    n = len(utterances)
    if n == 0:
        return []

    chunks: List[Dict[str, int]] = []
    cursor = 0
    while cursor < n:
        token_sum = 0
        end = cursor - 1
        last_natural = cursor - 1
        i = cursor
        while i < n:
            line = _build_numbered_transcript([utterances[i]], start_index_offset=i)
            line_tokens = count_tokens(line, model=model)
            if end >= cursor and token_sum + line_tokens > transcript_token_budget:
                break
            token_sum += line_tokens
            end = i
            if _is_natural_boundary(utterances[i]):
                last_natural = i
            i += 1

        if end < cursor:
            end = cursor

        # If possible, cut at a nearby natural boundary to preserve section continuity.
        if last_natural >= cursor and end - last_natural <= 12:
            end = last_natural

        chunk_start = max(0, cursor - overlap_utterances) if cursor > 0 else 0
        chunks.append({"start": chunk_start, "end": end, "emit_from": cursor})
        cursor = end + 1

    return chunks


def _merge_stage_ranges(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not stages:
        return []
    stages_sorted = sorted(stages, key=lambda s: (s.get("start_index", 0), s.get("end_index", 0)))
    merged: List[Dict[str, Any]] = []
    for s in stages_sorted:
        if not isinstance(s, dict):
            continue
        name = (s.get("stage_name") or "").strip() or "Unknown"
        start = int(s.get("start_index", 0))
        end = int(s.get("end_index", start))
        if end < start:
            end = start
        if not merged:
            merged.append({"stage_name": name, "start_index": start, "end_index": end})
            continue
        last = merged[-1]
        same_name = _normalize_stage_name(last.get("stage_name", "")) == _normalize_stage_name(name)
        touches = start <= int(last.get("end_index", 0)) + 1
        if same_name and touches:
            last["end_index"] = max(int(last["end_index"]), end)
        else:
            merged.append({"stage_name": name, "start_index": start, "end_index": end})
    return merged


def _build_stage_memory(stages: List[Dict[str, Any]], tail: int = 8) -> str:
    if not stages:
        return ""
    lines = []
    for s in stages[-tail:]:
        lines.append(
            f'- {s.get("stage_name", "Unknown")}: {s.get("start_index", 0)} to {s.get("end_index", 0)}'
        )
    return "\n".join(lines)


def _make_stage_prompt(
    c_start: int,
    c_end: int,
    transcript: str,
    stages_hint: str,
    rolling_memory: str,
    minutes_block: str = "",
) -> str:
    memory_block = (
        f"Previous chunk stage memory (for continuity only, do not copy blindly):\n{rolling_memory}\n\n"
        if rolling_memory
        else ""
    )
    minutes_part = minutes_block if (minutes_block or "").strip() else ""
    return f"""You are an expert at analyzing city council meeting transcripts. Identify the MEETING STAGES (agenda segments) in the chunk below.

Common stages include (use these names when applicable; you may add others): {stages_hint}.

{minutes_part}{memory_block}For each stage, determine which consecutive utterances belong to it. Utterances in this chunk are numbered from {c_start} to {c_end}. Boundaries must be derived from the transcript lines (time, speaker_id, text). If official minutes are provided above, use them only as supplementary hints for stage names — do NOT assume every section from the minutes exists in the transcript. The transcript may be incomplete, starting mid-meeting or missing sections entirely. Only output stages you can actually identify from the transcript content.

Return a JSON array of objects in chronological order:
[
  {{ "stage_name": "Roll Call", "start_index": {c_start}, "end_index": {min(c_start + 5, c_end)} }},
  {{ "stage_name": "Public Comment", "start_index": {max(c_start, c_end - 20)}, "end_index": {c_end} }}
]

Rules:
- start_index and end_index are 0-based global utterance indices; end_index is INCLUSIVE.
- Indices must stay within this chunk: [{c_start}, {c_end}].
- Stages must be chronological and non-overlapping.
- Prefer keeping stage names consistent with prior memory when the same agenda phase continues.
- A meeting can open and close the SAME kind of stage several times (reserved vs. general public
  commentary, extended time, one hearing per agenda item). Emit each occurrence as its own entry
  instead of merging them or keeping only the first.
- Keep the city's own wording for public-speaking windows ("Public Participation", "Citizens Open
  Forum", "Call to the Public", ...) rather than renaming them to "Public Comment".
- Return only the JSON array, no other text.

Transcript (per line: index, time [start-end s], speaker_id, text):
---
{transcript}
---
"""


def _fit_chunk_end_by_prompt_budget(
    utterances: List[Dict[str, Any]],
    c_start: int,
    c_end: int,
    stages_hint: str,
    rolling_memory: str,
    model: str,
    prompt_token_budget: int,
    minutes_block: str = "",
) -> int:
    """
    Ensure full prompt token count stays under budget by shrinking chunk end.
    """
    low = c_start
    high = c_end
    best = c_start
    while low <= high:
        mid = (low + high) // 2
        transcript = _build_numbered_transcript(utterances[c_start : mid + 1], start_index_offset=c_start)
        prompt = _make_stage_prompt(
            c_start, mid, transcript, stages_hint, rolling_memory, minutes_block=minutes_block
        )
        p_tokens = count_tokens(prompt, model=model)
        if p_tokens <= prompt_token_budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def _extract_stages_for_one_meeting(
    utterances: List[Dict[str, Any]],
    meeting_date: str = "",
    model: str = "gpt-4o-2024-11-20",
    max_utterances: Optional[int] = 800,
    minutes_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Ask GPT to identify meeting stages and return list of { stage_name, start_index, end_index }.
    Indices are 0-based utterance indices. end_index is inclusive.
    """
    if not utterances:
        return []

    if max_utterances and len(utterances) > max_utterances:
        utterances = utterances[:max_utterances]

    n = len(utterances)
    minutes_cap = int(os.getenv("MEETING_MINUTES_MAX_TOKENS", "6000"))
    minutes_raw = (minutes_text or "").strip()
    if minutes_raw:
        minutes_raw = truncate_minutes(minutes_raw, max_tokens=minutes_cap, model=model)
    minutes_block = minutes_prompt_block(minutes_raw, meeting_date=meeting_date) if minutes_raw else ""

    # Conservative defaults for local models; configurable by env for tuning.
    transcript_token_budget = int(os.getenv("MEETING_STAGE_MAX_TRANSCRIPT_TOKENS", "18000"))
    # Full prompt guard to avoid exceeding model context (Qwen3-14B default max_position_embeddings=40960).
    prompt_token_budget = int(os.getenv("MEETING_STAGE_MAX_PROMPT_TOKENS", "32000"))
    overlap_utterances = int(os.getenv("MEETING_STAGE_CHUNK_OVERLAP", "12"))
    chunk_ranges = _chunk_utterance_ranges(
        utterances=utterances,
        model=model,
        transcript_token_budget=transcript_token_budget,
        overlap_utterances=overlap_utterances,
    )

    stages_hint = ", ".join(COMMON_STAGES)
    all_stages: List[Dict[str, Any]] = []
    rolling_memory = ""
    for chunk in chunk_ranges:
        c_start = chunk["start"]
        c_end = chunk["end"]
        emit_from = chunk["emit_from"]
        c_end = _fit_chunk_end_by_prompt_budget(
            utterances=utterances,
            c_start=c_start,
            c_end=c_end,
            stages_hint=stages_hint,
            rolling_memory=rolling_memory,
            model=model,
            prompt_token_budget=prompt_token_budget,
            minutes_block=minutes_block,
        )
        transcript = _build_numbered_transcript(utterances[c_start : c_end + 1], start_index_offset=c_start)
        prompt = _make_stage_prompt(
            c_start, c_end, transcript, stages_hint, rolling_memory, minutes_block=minutes_block
        )
        use_structured_json = os.getenv("QWEN_USE_OUTLINES_JSON", "1").strip().lower() in {"1", "true", "yes", "on"}
        if use_structured_json and os.getenv("LLM_BACKEND", "").strip().lower() in {"qwen", "vllm"}:
            out = ChatGPT_API_JSON(
                model=model,
                prompt=prompt,
                json_schema=STAGE_LIST_JSON_SCHEMA,
            )
        else:
            response = ChatGPT_API(model=model, prompt=prompt)
            out = extract_json(response)
        # Retry once with smaller chunk when model returns non-JSON or empty output.
        invalid_or_empty = (
            not (isinstance(out, list) or (isinstance(out, dict) and "stages" in out))
            or (isinstance(out, list) and len(out) == 0)
            or (isinstance(out, dict) and isinstance(out.get("stages"), list) and len(out.get("stages")) == 0)
        )
        if invalid_or_empty and c_end > emit_from:
            retry_end = max(emit_from, c_start + (c_end - c_start) // 2)
            transcript_retry = _build_numbered_transcript(
                utterances[c_start : retry_end + 1],
                start_index_offset=c_start,
            )
            prompt_retry = _make_stage_prompt(
                c_start,
                retry_end,
                transcript_retry,
                stages_hint,
                rolling_memory,
                minutes_block=minutes_block,
            )
            if use_structured_json and os.getenv("LLM_BACKEND", "").strip().lower() in {"qwen", "vllm"}:
                out = ChatGPT_API_JSON(
                    model=model,
                    prompt=prompt_retry,
                    json_schema=STAGE_LIST_JSON_SCHEMA,
                )
            else:
                response = ChatGPT_API(model=model, prompt=prompt_retry)
                out = extract_json(response)
            c_end = retry_end
        if isinstance(out, list):
            chunk_stages = out
        elif isinstance(out, dict) and "stages" in out:
            chunk_stages = out["stages"]
        else:
            chunk_stages = []

        normalized_chunk: List[Dict[str, Any]] = []
        for s in chunk_stages:
            if not isinstance(s, dict):
                continue
            start = s.get("start_index", c_start)
            end = s.get("end_index", start)
            name = (s.get("stage_name") or "").strip() or "Unknown"
            start = max(c_start, min(int(start), c_end))
            end = max(start, min(int(end), c_end))
            # Keep only new area beyond overlap to avoid duplicate ranges.
            if end < emit_from:
                continue
            start = max(start, emit_from)
            normalized_chunk.append(
                {"stage_name": name, "start_index": start, "end_index": end}
            )

        all_stages.extend(normalized_chunk)
        all_stages = _merge_stage_ranges(all_stages)
        rolling_memory = _build_stage_memory(all_stages)

    for s in all_stages:
        start = s.get("start_index", 0)
        end = s.get("end_index", start)
        s["start_index"] = max(0, min(int(start), n - 1))
        s["end_index"] = max(s["start_index"], min(int(end), n - 1))

    return all_stages


def extract_meeting_structure(
    meeting_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    meeting_path: Optional[str] = None,
    model: str = "gpt-4o-2024-11-20",
    max_utterances: Optional[int] = 800,
    minutes_by_meeting: Optional[Dict[str, str]] = None,
    minutes_dir: Optional[str] = None,
    workers: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract meeting structure (stages) for each meeting in the data.

    Args:
        meeting_data: Dict[meeting_date, list of utterance dicts]. If None, meeting_path must be set.
        meeting_path: Path to meeting JSON (RAG format). Used if meeting_data is None.
        model: Model name for GPT.
        max_utterances: Max utterances per meeting to send (to stay under context). None = no limit.
        minutes_by_meeting: Optional map meeting_date -> raw minutes text (injected into stage prompts).
        minutes_dir: Optional directory of <meeting_date>.txt or .md files (merged into minutes_by_meeting).

    Returns:
        Dict[meeting_date, list of stage dicts]. Each stage dict has:
        - stage_name: str (e.g. "Roll Call", "Public Comment")
        - start_index: int (0-based utterance index, inclusive)
        - end_index: int (0-based utterance index, inclusive)
    """
    if meeting_data is None:
        if not meeting_path:
            raise ValueError("Provide either meeting_data or meeting_path")
        meeting_data = load_meeting_json(meeting_path)

    merged_minutes: Dict[str, str] = {}
    if minutes_dir:
        merged_minutes.update(
            load_minutes_for_meetings(minutes_dir, list(meeting_data.keys()))
        )
    if minutes_by_meeting:
        for k, v in minutes_by_meeting.items():
            if v and str(v).strip():
                merged_minutes[k] = str(v).strip()

    def _one(meeting_date: str) -> List[Dict[str, Any]]:
        return _extract_stages_for_one_meeting(
            meeting_data[meeting_date],
            meeting_date=meeting_date,
            model=model,
            max_utterances=max_utterances,
            minutes_text=merged_minutes.get(meeting_date),
        )

    dates = list(meeting_data.keys())
    if workers > 1 and len(dates) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(zip(dates, pool.map(_one, dates)))
    return {d: _one(d) for d in dates}


def get_utterances_in_stage(
    utterances: List[Dict[str, Any]],
    stages: List[Dict[str, Any]],
    stage_name: str,
) -> List[Dict[str, Any]]:
    """
    Return utterances from every stage that maps to stage_name (e.g. "Public Comment"),
    including repeated windows such as reserved/general time and per-item hearings.
    """
    return [
        utterances[i]
        for i in indices_for_section(stages, stage_name)
        if i < len(utterances)
    ]


def _utterance_indices_in_stage(stages: List[Dict[str, Any]], stage_name: str) -> List[int]:
    """Return sorted 0-based utterance indices across ALL stages matching stage_name."""
    return indices_for_section(stages, stage_name)


def extract_public_comment_by_section(
    meeting_path: Optional[str] = None,
    meeting_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    structure: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    model: str = "gpt-4o-2024-11-20",
    minutes_by_meeting: Optional[Dict[str, str]] = None,
    minutes_dir: Optional[str] = None,
    workers: int = 1,
) -> tuple:
    """
    Step 1: Extract meeting structure (sections) if not provided.
    Step 2: For each meeting, mark utterances in Public Comment / Public Hearing sections and extract them.

    Returns:
        (structure, meeting_with_section_labels, public_comment_extract)
        - structure: Dict[meeting_date, list of {stage_name, start_index, end_index}]
        - meeting_with_section_labels: same as meeting_data but each utterance has is_public_comment_section=0/1, is_public_hearing_section=0/1
        - public_comment_extract: Dict[meeting_date, list of utterances in Public Comment or Public Hearing section]
    """
    if meeting_data is None:
        if not meeting_path:
            raise ValueError("Provide meeting_path or meeting_data")
        meeting_data = load_meeting_json(meeting_path)

    if structure is None:
        structure = extract_meeting_structure(
            meeting_data=meeting_data,
            model=model,
            minutes_by_meeting=minutes_by_meeting,
            minutes_dir=minutes_dir,
            workers=workers,
        )

    meeting_with_labels = {}
    public_comment_extract = {}

    for meeting_date, utterances in meeting_data.items():
        stages = structure.get(meeting_date, [])
        comment_indices, hearing_indices = public_section_indices(stages)
        in_comment_or_hearing = comment_indices | hearing_indices

        labeled = []
        extracted = []
        for i, u in enumerate(utterances):
            u_copy = dict(u)
            u_copy["is_public_comment_section"] = 1 if i in comment_indices else 0
            u_copy["is_public_hearing_section"] = 1 if i in hearing_indices else 0
            labeled.append(u_copy)
            if i in in_comment_or_hearing:
                u_extract = dict(u_copy)
                u_extract["is_public_comment"] = 1 if i in comment_indices else 0
                u_extract["is_public_hearing"] = 1 if i in hearing_indices else 0
                extracted.append(u_extract)
        meeting_with_labels[meeting_date] = labeled
        public_comment_extract[meeting_date] = extracted

    return structure, meeting_with_labels, public_comment_extract


# Section-specific wording for the public-remark classifier.
# Both Public Comment and Public Hearing contain citizen remarks interleaved with
# chair/moderator/staff utterances ("next speaker", "thank you", procedural text).
_REMARK_SECTION_CONFIG = {
    "public comment": {
        "section_label": "Public Comment",
        "remark_desc": "an actual PUBLIC COMMENT (a citizen/member of the public addressing council)",
        "json_key": "is_public_comment",
    },
    "public hearing": {
        "section_label": "Public Hearing",
        "remark_desc": (
            "an actual PUBLIC REMARK during the hearing (a citizen/member of the public "
            "speaking on the hearing item)"
        ),
        "json_key": "is_public_remark",
    },
}


def _resolve_remark_config(section: str) -> dict:
    key = (section or "").strip().lower()
    for cfg_key, cfg in _REMARK_SECTION_CONFIG.items():
        if cfg_key in key:
            return cfg
    # Default to public comment behaviour for unknown sections.
    return _REMARK_SECTION_CONFIG["public comment"]


def classify_utterance_public_remark(
    speaker: str,
    text: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    section: str = "Public Comment",
    model: str = "gpt-4o-2024-11-20",
) -> int:
    """
    Classify one utterance as an actual member-of-the-public remark (1) or not (0),
    within the given section ("Public Comment" or "Public Hearing").

    Allowed input only: time (start, end), text, speaker_id. No other fields.
    """
    cfg = _resolve_remark_config(section)
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    speaker = speaker if isinstance(speaker, str) else ("" if speaker is None else str(speaker))
    prompt = f"""You are a binary classifier. This utterance is from the "{cfg['section_label']}" section of a city council meeting. Decide if it is {cfg['remark_desc']} or not (e.g. chair/moderator/staff saying "next speaker", "thank you", reading rules, or other procedural talk).

Use only the following information (no other fields):
- speaker_id: {speaker}
- text: {text}
- time: start={start} end={end} (seconds)

Return JSON: {{ "{cfg['json_key']}": 0 or 1 }}. 1 = citizen/member-of-public remark, 0 = not (e.g. moderator/chair/staff)."""
    response = ChatGPT_API(model=model, prompt=prompt)
    out = extract_json(response) if isinstance(response, str) else {}
    val = out.get(cfg["json_key"], out.get("is_public_comment", 0))
    return 1 if val in [1, "1", True] else 0


def classify_utterance_public_comment(
    speaker: str,
    text: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    model: str = "gpt-4o-2024-11-20",
) -> int:
    """
    Backward-compatible wrapper: classify a Public Comment utterance (1) or not (0).
    """
    return classify_utterance_public_remark(
        speaker=speaker, text=text, start=start, end=end, section="Public Comment", model=model
    )
