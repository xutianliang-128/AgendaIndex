"""
Optional official minutes (agenda summary / clerk notes) as *global context* for LLM prompts.

Design (aligned with AgendaIndex-style pipelines):
- Transcript utterances remain the authoritative source for start_index/end_index.
- Minutes are hints only: agenda order, section titles, rough timing—may disagree with ASR.

Usage:
  - Pass minutes_by_meeting dict into extract_meeting_structure / extract_public_comment_by_section, or
  - Set MINUTES_DIR and pass meeting keys; load_minutes_for_meetings() finds <meeting_date>.txt/.md
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

try:
    from .utils import count_tokens
except ImportError:
    from utils import count_tokens


def load_minutes_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_minutes_for_meetings(minutes_dir: str, meeting_dates: List[str]) -> Dict[str, str]:
    """Load minutes text per meeting_date if <dir>/<meeting_date>.txt or .md exists."""
    out: Dict[str, str] = {}
    base = os.path.abspath(os.path.expanduser(minutes_dir))
    for d in meeting_dates:
        for ext in (".txt", ".md"):
            p = os.path.join(base, f"{d}{ext}")
            if os.path.isfile(p):
                out[d] = load_minutes_file(p)
                break
    return out


def truncate_minutes(text: str, max_tokens: int, model: str) -> str:
    if max_tokens <= 0 or not (text or "").strip():
        return (text or "").strip()
    t = text.strip()
    if count_tokens(t, model=model) <= max_tokens:
        return t
    # Truncate by growing prefix until token budget (cheap incremental).
    low, high = 0, len(t)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        chunk = t[:mid]
        if count_tokens(chunk, model=model) <= max_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return t[:best].rstrip() + "\n\n[... minutes truncated for token budget ...]"


def minutes_prompt_block(minutes_text: str, meeting_date: str = "") -> str:
    if not (minutes_text or "").strip():
        return ""
    hdr = f"Meeting date key: {meeting_date}\n" if meeting_date else ""
    return f"""OFFICIAL MINUTES (supplementary context only; may be incomplete or differ from audio):
{hdr}
IMPORTANT: The transcript below may be INCOMPLETE — it may start mid-meeting, skip sections,
or end abruptly. Do NOT force every section from the minutes into the transcript.
Only output stages that are actually present and supported by the transcript content.
The numbered transcript is authoritative for utterance indices and boundaries.
Use minutes only to help identify stage names when the transcript content is ambiguous.
---
{minutes_text.strip()}
---


"""
