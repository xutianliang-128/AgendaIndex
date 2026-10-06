"""Shared rules for deciding which meeting stages are public-speaking windows.

Stage names come from the LLM and vary a lot across cities ("Public Comment",
"Public Participation", "Citizens Open Forum", "Call to the Public", ...), and a
single meeting often has several of them (reserved time early, general time
late, one window per hearing item). Every caller must therefore classify by
meaning and keep *all* matching stages, not just the first one.

Recall is favoured over precision here: a stage that slips through only costs a
few extra per-utterance classifier calls, while a stage that is dropped can
never be recovered downstream.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PUBLIC_COMMENT = "Public Comment"
PUBLIC_HEARING = "Public Hearing"

# Stage names that mention a public window but are procedural talk *about* it
# (rules, registration, scheduling, officials responding) rather than the window.
_EXCLUDE = (
    "comment rules",
    "comment protocol",
    "comment instructions",
    "comment registration",
    "speaker registration",
    "speaker rules",
    "speaker guidelines",
    "speaker protocol",
    "registration for public comment",
    "to close public comment",
    "response to public",
    "setting public hearing",
    "set public hearing",
    "call for public hearing",
    "referral of public hearing",
    "hearing scheduling",
    "hearing announcement",
    "hearing setup",
    "hearing introduction",
    "hearing overview",
    "hearing transition",
    "hearing preparation",
    "adjournment of public",
    "show cause",
    "public safety",
    "public works",
    "public service announcement",
    "public meeting act",
    "public meetings act",
    "recognition",
    "citizen of the month",
)

# Windows where members of the public address the body on anything.
_PC_PATTERNS = (
    "public comment",  # also matches "public commentary"
    "public participation",
    "public input",
    "public forum",
    "public speaking",
    "public business from the floor",
    "public communication",
    "communications from the public",
    "matters by public",
    "matters presented by members of the public",
    "matters by the public",
    "call to the public",
    "citizen comment",
    "citizens comment",
    "citizen's comment",
    "citizens' comment",
    "citizen participation",
    "citizens participation",
    "citizen's participation",
    "citizens' participation",
    "citizen communication",
    "citizens communication",
    "citizens' communication",
    "citizen concerns",
    "citizens request",
    "citizens' request",
    "open forum",
    "audience participation",
    "audience comment",
    "audience for visitors",
    "appearance of citizens",
    "non-agenda speakers",
    "non agenda speakers",
    # Ann Arbor style split windows; "extended"/"reserved" time is still public.
    "reserved time",
    "reserve time",
    "general time",
    "extended time",
)

# Windows tied to a noticed hearing item.
_PH_PATTERNS = (
    "public hearing",
    "budget hearing",
)


def classify_stage(stage_name: Optional[str]) -> Optional[str]:
    """Return "pc", "ph", or None for a single stage name."""
    name = (stage_name or "").strip().lower()
    if not name:
        return None
    if any(bad in name for bad in _EXCLUDE):
        return None
    if any(pat in name for pat in _PH_PATTERNS):
        return "ph"
    if any(pat in name for pat in _PC_PATTERNS):
        return "pc"
    return None


def _stage_range(stage: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    try:
        start = int(stage.get("start_index", 0))
        end = int(stage.get("end_index", start))
    except (TypeError, ValueError):
        return None
    if end < start:
        start, end = end, start
    return start, end


def public_section_indices(
    stages: Optional[Iterable[Dict[str, Any]]],
) -> Tuple[Set[int], Set[int]]:
    """Return (public-comment indices, public-hearing indices) over ALL matching stages."""
    pc: Set[int] = set()
    ph: Set[int] = set()
    for stage in stages or []:
        if not isinstance(stage, dict):
            continue
        kind = classify_stage(stage.get("stage_name"))
        if kind is None:
            continue
        span = _stage_range(stage)
        if span is None:
            continue
        target = pc if kind == "pc" else ph
        target.update(range(span[0], span[1] + 1))
    return pc, ph


def indices_for_section(
    stages: Optional[Iterable[Dict[str, Any]]],
    section: str,
) -> List[int]:
    """Sorted indices for one canonical section label ("Public Comment"/"Public Hearing")."""
    pc, ph = public_section_indices(stages)
    return sorted(ph if "hearing" in (section or "").lower() else pc)


def matching_stages(
    stages: Optional[Iterable[Dict[str, Any]]],
    section: str,
) -> List[Dict[str, Any]]:
    """Every stage mapped to the given canonical section, in agenda order."""
    want = "ph" if "hearing" in (section or "").lower() else "pc"
    return [
        s
        for s in (stages or [])
        if isinstance(s, dict) and classify_stage(s.get("stage_name")) == want
    ]
