"""Prompt for the Stage-2 remark classifier fine-tuned on Qwen3-4B.

One target utterance inside a public-speaking window, shown with its
neighbours. The model answers with a single digit so inference can read the
answer off one next-token distribution instead of decoding.
"""
from __future__ import annotations

SECTION_DESC = {
    "Public Comment": "the public-comment portion",
    "Public Hearing": "a public hearing",
}

INSTRUCTIONS = """You label one utterance from {where} of a city council meeting.

Answer 1 if the target utterance is spoken by a member of the public addressing the council: a resident, business owner, or community member who came forward to speak. Such speakers usually give their name or address and speak in one longer turn.

Answer 0 for the mayor, chair, council members, city staff, attorneys, clerks, police or fire officials, consultants, and for procedural talk (calling the next speaker, thanking a speaker, reading rules, opening or closing the item). Staff presentations and council deliberation inside this window are 0.

Agenda heading of this window: {stage}"""


def _clip(text: str, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 3] + "..."


def build_prompt(
    utts: list[dict],
    target: int,
    section: str,
    stage_name: str,
    ctx: int = 5,
    ctx_chars: int = 300,
    target_chars: int = 1200,
) -> str:
    lo, hi = max(0, target - ctx), min(len(utts) - 1, target + ctx)
    lines = []
    for j in range(lo, hi + 1):
        u = utts[j]
        text = _clip(u.get("text"), target_chars if j == target else ctx_chars)
        mark = ">>>" if j == target else "   "
        lines.append(f"{mark} [{j}] {u.get('speaker') or 'UNKNOWN'}: {text}")
    head = INSTRUCTIONS.format(
        where=SECTION_DESC.get(section, "a public-speaking window"),
        stage=_clip(stage_name, 120),
    )
    return (
        f"{head}\n\nTranscript (the target is marked >>>):\n"
        + "\n".join(lines)
        + f"\n\nIs utterance [{target}] spoken by a member of the public? Answer 1 or 0."
    )
