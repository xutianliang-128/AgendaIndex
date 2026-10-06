"""
Locate and load localgov meeting transcripts by numeric FIBS ID and year.

FIBS is the numeric jurisdiction folder name on disk, e.g. 00674, 13276.
Path layout:
  {LOCALGOV_ROOT}/{year}/{fibs}/transcripts/

Meeting keys in output JSON may look like 00674_<video_id>_23 when source
files are named by video id (not AA_08_07_23).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_LOCALGOV_ROOT = __import__("os").environ.get(
    "LOCALGOV_ROOT",
    "/home/shared/turbo_videos/processed_videos/localgov",
)
DEFAULT_TRANSCRIPTS_ROOT = DEFAULT_LOCALGOV_ROOT

MEETING_KEY_RE = re.compile(
    r"^(?P<fibs>[A-Za-z0-9]+)_(?P<mm>\d{2})_(?P<dd>\d{2})_(?P<yy>\d{2})$"
)

FIBS_NUMERIC_WIDTH = 5  # e.g. 00674 on disk


def normalize_fibs(fibs: str | int) -> str:
    """Normalize FIBS for paths/keys. Numeric IDs keep leading zeros when padded."""
    s = str(fibs).strip()
    if not s:
        raise ValueError("FIBS must be non-empty")
    if s.isdigit():
        return s
    if re.fullmatch(r"\d+", s):
        return s
    return s.upper()


def fibs_path_candidates(fibs: str | int) -> List[str]:
    """
    Directory name variants to try under localgov/{year}/.

    Accepts 674, 00674, etc. for 5-digit zero-padded folder names.
    """
    s = normalize_fibs(fibs)
    if not s.isdigit():
        return [s]

    seen: set[str] = set()
    out: List[str] = []
    for candidate in (s, s.zfill(FIBS_NUMERIC_WIDTH), s.lstrip("0") or "0"):
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def format_fibs_for_key(fibs: str) -> str:
    """Use in meeting keys: numeric as-is (zero-padded), legacy alpha uppercased."""
    s = normalize_fibs(fibs)
    if s.isdigit():
        return s.zfill(FIBS_NUMERIC_WIDTH) if len(s) <= FIBS_NUMERIC_WIDTH else s
    return s.upper()


def normalize_year(year: int | str) -> Tuple[str, str]:
    """Return (four_digit, two_digit) year strings."""
    s = str(year).strip()
    if len(s) == 2:
        yy = s
        yyyy = f"20{yy}" if int(yy) <= 50 else f"19{yy}"
    elif len(s) == 4:
        yyyy = s
        yy = yyyy[-2:]
    else:
        raise ValueError(f"Invalid year: {year!r} (use 2023 or 23)")
    return yyyy, yy


def parse_meeting_key(key: str) -> Optional[Dict[str, str]]:
    m = MEETING_KEY_RE.match(key.strip())
    if not m:
        return None
    return m.groupdict()


def meeting_key_from_parts(fibs: str, mm: str, dd: str, yy: str) -> str:
    return f"{format_fibs_for_key(fibs)}_{mm}_{dd}_{yy}"


def resolve_transcripts_dir(
    localgov_root: str,
    fibs: str,
    year: int | str,
) -> Path:
    """
    Resolve .../localgov/{year}/{fibs}/transcripts

    Tries 4-digit year first, then 2-digit. For numeric FIBS, tries zero-padded variants.
    """
    root = Path(localgov_root)
    yyyy, yy = normalize_year(year)

    tried: List[str] = []
    for fibs_token in fibs_path_candidates(fibs):
        for year_token in (yyyy, yy):
            candidate = root / year_token / fibs_token / "transcripts"
            tried.append(str(candidate))
            if candidate.is_dir():
                return candidate

    raise FileNotFoundError(
        "Transcripts directory not found. Tried:\n  " + "\n  ".join(tried)
    )


def _is_transcript_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz"))


def discover_transcript_files(
    localgov_root: str,
    fibs: str,
    year: int | str,
    *,
    recursive: bool = False,
) -> List[Path]:
    """
    List transcript files under localgov/{year}/{fibs}/transcripts/.
    """
    fibs_key = format_fibs_for_key(fibs)
    transcripts_dir = resolve_transcripts_dir(localgov_root, fibs, year)

    if recursive:
        paths = [p for p in transcripts_dir.rglob("*") if p.is_file() and _is_transcript_file(p)]
    else:
        paths = [p for p in transcripts_dir.iterdir() if p.is_file() and _is_transcript_file(p)]

    def sort_key(p: Path) -> Tuple[int, str]:
        stem = p.stem.replace(".json", "")
        exact = 0 if MEETING_KEY_RE.match(stem) and stem.startswith(f"{fibs_key}_") else 1
        return (exact, str(p))

    return sorted(set(paths), key=sort_key)


def _default_utterance() -> Dict[str, Any]:
    return {
        "speaker": "",
        "text": "",
        "start": None,
        "end": None,
        "Transition": '""',
        "Meeting Section": "",
        "Speaker Role": "",
        "is_public_comment": 0,
        "is_public_hearing": 0,
        "is_comment_start_trigger": 0,
        "is_comment_end_trigger": 0,
        "is_hearing_start_trigger": 0,
        "is_hearing_end_trigger": 0,
    }


def _normalize_utterance(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = _default_utterance()
    out["speaker"] = raw.get("speaker") or raw.get("speaker_id") or raw.get("speaker_label") or ""
    out["text"] = (raw.get("text") or raw.get("transcript") or raw.get("content") or "").strip()
    out["start"] = raw.get("start", raw.get("start_time", raw.get("begin")))
    out["end"] = raw.get("end", raw.get("end_time", raw.get("finish")))
    for k in (
        "Transition",
        "Meeting Section",
        "Speaker Role",
        "is_public_comment",
        "is_public_hearing",
        "is_comment_start_trigger",
        "is_comment_end_trigger",
        "is_hearing_start_trigger",
        "is_hearing_end_trigger",
    ):
        if k in raw:
            out[k] = raw[k]
    return out


_TRANSCRIPT_LIST_KEYS = ("segments", "utterances", "results", "items", "transcript")


def _is_meeting_key_dict(payload: Dict[str, Any]) -> bool:
    """True if top-level keys look like meeting ids (AA_08_07_23), not transcript fields."""
    if not payload or not all(isinstance(v, list) for v in payload.values()):
        return False
    return all(parse_meeting_key(k) is not None for k in payload.keys())


def _payload_to_utterances(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [_normalize_utterance(x) for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []

    if _is_meeting_key_dict(payload):
        return []

    for key in _TRANSCRIPT_LIST_KEYS:
        if key in payload and isinstance(payload[key], list):
            items = payload[key]
            if not items:
                continue
            if key == "word_segments" and items and isinstance(items[0], dict) and "word" in items[0]:
                merged: List[Dict[str, Any]] = []
                for x in items:
                    if isinstance(x, dict):
                        merged.append(_normalize_utterance({**x, "text": x.get("word", "")}))
                return merged
            return [_normalize_utterance(x) for x in items if isinstance(x, dict)]

    return []


def _extract_utterance_list(payload: Any) -> List[Dict[str, Any]]:
    return _payload_to_utterances(payload)


def _infer_meeting_key(path: Path, fibs: str, year: int | str) -> str:
    stem = path.name.replace(".json.gz", "").replace(".jsonl", "").replace(".json", "")
    parsed = parse_meeting_key(stem)
    if parsed:
        return meeting_key_from_parts(parsed["fibs"], parsed["mm"], parsed["dd"], parsed["yy"])
    _, yy = normalize_year(year)
    fibs_key = format_fibs_for_key(fibs)
    return f"{fibs_key}_unknown_{yy}_{path.stem}"


def load_transcript_file(path: Path, fibs: str, year: int | str) -> Dict[str, List[Dict[str, Any]]]:
    """Load one transcript file -> { meeting_key: [utterances] }."""
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        if path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz"):
            utterances = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utterances.append(_normalize_utterance(json.loads(line)))
        else:
            payload = json.load(f)

    if path.suffix != ".jsonl" and not path.name.endswith(".jsonl.gz"):
        if isinstance(payload, dict) and _is_meeting_key_dict(payload):
            out: Dict[str, List[Dict[str, Any]]] = {}
            for k, items in payload.items():
                out[k] = [_normalize_utterance(x) for x in items if isinstance(x, dict)]
            return out
        utterances = _payload_to_utterances(payload)
    # jsonl branch already set utterances

    key = _infer_meeting_key(path, fibs, year)
    utterances = [u for u in utterances if u.get("text")]
    if not utterances:
        return {}
    return {key: utterances}


def load_meetings_for_fibs_year(
    localgov_root: str,
    fibs: str,
    year: int | str,
    *,
    file_paths: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """
    Load all meetings for FIBS+year from localgov/{year}/{fibs}/transcripts/.

    Returns (meeting_data, source_files).
    If multiple files map to the same meeting key, later files overwrite earlier ones.
    """
    if file_paths:
        paths = [Path(p) for p in file_paths]
        transcripts_dir = paths[0].parent if paths else None
    else:
        transcripts_dir = resolve_transcripts_dir(localgov_root, fibs, year)
        paths = discover_transcript_files(localgov_root, fibs, year)
    if not paths:
        yyyy, yy = normalize_year(year)
        raise FileNotFoundError(
            f"No transcript files in {transcripts_dir} for FIBS={format_fibs_for_key(fibs)} year={yyyy}/{yy}"
        )

    meeting_data: Dict[str, List[Dict[str, Any]]] = {}
    sources: List[str] = []
    skipped_empty: List[str] = []
    for path in paths:
        chunk = load_transcript_file(path, fibs, year)
        if not chunk:
            skipped_empty.append(str(path))
            continue
        meeting_data.update(chunk)
        sources.append(str(path))
    if skipped_empty:
        print(f"  Skipped {len(skipped_empty)} empty transcript file(s) (no segments/utterances)")
        for p in skipped_empty:
            print(f"    {p}")
    if not meeting_data:
        yyyy, yy = normalize_year(year)
        raise FileNotFoundError(
            f"No usable transcript data in {transcripts_dir} for FIBS={format_fibs_for_key(fibs)} "
            f"year={yyyy}/{yy} ({len(paths)} file(s), all empty or unparseable)"
        )
    return meeting_data, sources


def load_meetings_for_fibs_years(
    localgov_root: str,
    fibs: str,
    years: List[int],
    *,
    file_paths: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str], List[int]]:
    """
    Load meetings across multiple years. Missing year directories are skipped.

    Returns (meeting_data, source_files, years_loaded).
    """
    if file_paths:
        meeting_data, sources = load_meetings_for_fibs_year(
            localgov_root, fibs, years[0], file_paths=file_paths
        )
        return meeting_data, sources, [years[0]]

    meeting_data: Dict[str, List[Dict[str, Any]]] = {}
    sources: List[str] = []
    years_loaded: List[int] = []
    for year in years:
        try:
            chunk, chunk_sources = load_meetings_for_fibs_year(localgov_root, fibs, year)
        except FileNotFoundError:
            continue
        if not chunk:
            continue
        meeting_data.update(chunk)
        sources.extend(chunk_sources)
        years_loaded.append(year)
    return meeting_data, sources, years_loaded
