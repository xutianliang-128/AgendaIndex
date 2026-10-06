#!/usr/bin/env python3
"""Rebuild public-comment JSON from AgendaIndex inference.

is_public_comment / is_public_hearing:
    1 iff within_section_predictions.predicted_is_remark==1 in that section
is_public_comment_section / is_public_hearing_section:
    1 iff meeting_structure places the utterance in a PC / PH window
    (all matching stages, not only the first window)

Drops trigger fields. One JSON per city: {meeting_id: [utterances]}.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
AUDIT = ROOT / "docs" / "audit_summary.json"
META_DIR = ROOT / "datasets"
OUT_DIR = ROOT / "exports" / "inferred_pc_ph_57cities"
ZIP_PATH = ROOT / "exports" / "inferred_pc_ph_57cities.zip"

PAT = re.compile(r"^(\d+)_(\d{4})_(.+)\.json$")
KEEP = (
    "speaker",
    "text",
    "start",
    "end",
    "is_public_comment",
    "is_public_hearing",
    "is_public_comment_section",
    "is_public_hearing_section",
)
# Playlist.state is empty for a few FIPS; confirmed from meeting titles / Census place codes.
STATE_OVERRIDES = {
    "01540": "MI",  # Alma
    "03000": "MI",  # Ann Arbor
    "06020": "MI",  # Bay City
    "46000": "MI",  # Lansing
    "68380": "VA",  # Richmond (Clay / 8th / 9th St. city hall)
}
NAME_OVERRIDES = {
    "00674": ("Albany", "CA"),
    "23515": ("Federal Way", "WA"),
    "27540": ("Hartford", "SD"),
    "36000": ("Jersey City", "NJ"),
    "50000": ("Mobile", "AL"),
    "62140": ("Sahuarita", "AZ"),
}
CROSSWALK = ROOT / "exports" / "fips_city_crosswalk.csv"


sys.path.insert(0, str(ROOT))
from agendaindex.section_taxonomy import public_section_indices  # noqa: E402


def indices_by_type(stages: list) -> tuple[set[int], set[int]]:
    return public_section_indices(stages if isinstance(stages, list) else [])


def as01(v) -> int:
    return 1 if v in (1, "1", True) else 0


def _title_city(name: str) -> str:
    return " ".join(w.capitalize() for w in (name or "").strip().split())


def _put(names: dict[str, tuple[str, str]], fips: str, city: str, state: str, overwrite: bool = False) -> None:
    fips = str(fips or "").strip().zfill(5)
    city = _title_city(city)
    state = (state or "").strip().upper() or STATE_OVERRIDES.get(fips, "")
    if not fips or not city:
        return
    if overwrite or fips not in names or (city and not names[fips][0]):
        prev = names.get(fips, ("", ""))
        names[fips] = (city, state or prev[1])


def load_names_from_db() -> dict[str, tuple[str, str]]:
    names: dict[str, tuple[str, str]] = {}
    host = os.environ.get("LOCALGOV_HOST")
    user = os.environ.get("LOCALGOV_USER")
    password = os.environ.get("LOCALGOV_PASSWORD")
    if not (host and user and password):
        return names
    try:
        import pymysql
    except ImportError:
        return names
    conn = pymysql.connect(
        host=host,
        port=int(os.environ.get("LOCALGOV_PORT", "3306")),
        database=os.environ.get("LOCALGOV_DB", "localgov"),
        user=user,
        password=password,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=30,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT fips, city, state, COUNT(*) n FROM playlist "
                "WHERE city IS NOT NULL AND city <> '' "
                "GROUP BY fips, city, state ORDER BY n DESC"
            )
            for row in cur.fetchall():
                _put(names, row.get("fips"), row.get("city"), row.get("state") or "")
    finally:
        conn.close()
    return names


def load_city_names() -> dict[str, tuple[str, str]]:
    names: dict[str, tuple[str, str]] = {}
    if CROSSWALK.exists():
        with open(CROSSWALK, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                _put(names, row.get("fips"), row.get("city"), row.get("state") or "")
    for fips, pair in load_names_from_db().items():
        _put(names, fips, pair[0], pair[1], overwrite=True)
    for meta in META_DIR.rglob("metadata.csv"):
        try:
            with open(meta, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    _put(names, row.get("fips"), row.get("city"), row.get("state") or "")
        except OSError:
            continue
    for fips, (city, state) in NAME_OVERRIDES.items():
        _put(names, fips, city, state)
    for fips, state in STATE_OVERRIDES.items():
        if fips in names:
            city, st = names[fips]
            names[fips] = (city, st or state)
    return names


def write_crosswalk(names: dict[str, tuple[str, str]], cities: list[str]) -> None:
    CROSSWALK.parent.mkdir(parents=True, exist_ok=True)
    with open(CROSSWALK, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fips", "city", "state"])
        w.writeheader()
        for fips in cities:
            city, state = names.get(fips, ("", ""))
            w.writerow({"fips": fips, "city": city, "state": state})


def index_result_files() -> dict[tuple[str, str], dict[str, Path]]:
    keys: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for fn in os.listdir(RES):
        m = PAT.match(fn)
        if not m:
            continue
        keys[(m.group(1).zfill(5), m.group(2))][m.group(3)] = RES / fn
    return keys


def load_remark_map(path: Path) -> dict[tuple[str, int], dict[str, int]]:
    """(meeting_id, utterance_index) -> {pc: 0/1, ph: 0/1} from predicted_is_remark."""
    out: dict[tuple[str, int], dict[str, int]] = {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    preds = payload.get("predictions") if isinstance(payload, dict) else None
    if not isinstance(preds, list):
        return out
    for p in preds:
        if not isinstance(p, dict):
            continue
        mid = p.get("meeting_date")
        idx = p.get("utterance_index")
        if mid is None or idx is None:
            continue
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        rec = out.setdefault((str(mid), idx), {"pc": 0, "ph": 0})
        if not as01(p.get("predicted_is_remark")):
            continue
        sec = (p.get("section") or "").lower()
        if "hearing" in sec:
            rec["ph"] = 1
        else:
            rec["pc"] = 1
    return out


def make_utt(src: dict, pc_sec: int, ph_sec: int, pc: int, ph: int) -> dict:
    return {
        "speaker": src.get("speaker") or "",
        "text": src.get("text") if src.get("text") is not None else "",
        "start": src.get("start"),
        "end": src.get("end"),
        "is_public_comment": pc,
        "is_public_hearing": ph,
        "is_public_comment_section": pc_sec,
        "is_public_hearing_section": ph_sec,
    }


def process_city_year(paths: dict[str, Path]) -> tuple[dict[str, list], dict]:
    stats = {
        "n_meetings": 0,
        "n_meetings_with_public_section": 0,
        "n_section_utts": 0,
        "n_pc_remarks": 0,
        "n_ph_remarks": 0,
        "n_pred_unmatched": 0,
    }
    src_path = paths.get("from_transcripts") or paths.get("meeting_with_sections")
    struct_path = paths.get("meeting_structure")
    if not src_path or not struct_path:
        return {}, stats

    try:
        transcripts = json.loads(src_path.read_text(encoding="utf-8"))
        structure = json.loads(struct_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, stats
    if not isinstance(transcripts, dict) or not isinstance(structure, dict):
        return {}, stats

    remarks = load_remark_map(paths["within_section_predictions"]) if "within_section_predictions" in paths else {}
    used_pred: set[tuple[str, int]] = set()
    meetings: dict[str, list] = {}

    for mid, utts in transcripts.items():
        if not isinstance(utts, list):
            continue
        stats["n_meetings"] += 1
        pc_idx, ph_idx = indices_by_type(structure.get(mid, []))
        section_idx = pc_idx | ph_idx
        if not section_idx:
            continue
        stats["n_meetings_with_public_section"] += 1
        rows = []
        for i in sorted(section_idx):
            if i < 0 or i >= len(utts) or not isinstance(utts[i], dict):
                continue
            pc_sec = 1 if i in pc_idx else 0
            ph_sec = 1 if i in ph_idx else 0
            rec = remarks.get((mid, i), {"pc": 0, "ph": 0})
            if (mid, i) in remarks:
                used_pred.add((mid, i))
            # Remark only if that utterance is actually in the corresponding section.
            pc = rec["pc"] if pc_sec else 0
            ph = rec["ph"] if ph_sec else 0
            rows.append(make_utt(utts[i], pc_sec, ph_sec, pc, ph))
            stats["n_section_utts"] += 1
            stats["n_pc_remarks"] += pc
            stats["n_ph_remarks"] += ph
        if rows:
            meetings[mid] = rows

    stats["n_pred_unmatched"] = sum(1 for k in remarks if k not in used_pred)
    return meetings, stats


def add_stats(total: dict, part: dict) -> None:
    for k, v in part.items():
        total[k] = total.get(k, 0) + v


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    cities = [str(c).zfill(5) for c in audit["cities"]]
    names = load_city_names()
    n_named = sum(1 for f in cities if names.get(f, ("", ""))[0])
    n_stated = sum(1 for f in cities if names.get(f, ("", ""))[1])
    print(f"city names: {n_named}/{len(cities)}  with state: {n_stated}/{len(cities)}", flush=True)
    missing = [f for f in cities if not names.get(f, ("", ""))[0]]
    if missing:
        raise SystemExit(f"missing city names for {missing}")
    files = index_result_files()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_crosswalk(names, cities)
    (OUT_DIR / "fips_city_crosswalk.csv").write_text(CROSSWALK.read_text(encoding="utf-8"), encoding="utf-8")
    city_stats = {}
    grand = defaultdict(int)

    for i, fips in enumerate(cities, 1):
        years = sorted(y for (f, y) in files if f == fips)
        city_meetings: dict[str, list] = {}
        st = defaultdict(int)
        for y in years:
            meetings, part = process_city_year(files[(fips, y)])
            city_meetings.update(meetings)
            add_stats(st, part)
            print(f"[{i}/{len(cities)}] {fips} {y}: meetings={part['n_meetings_with_public_section']} "
                  f"utts={part['n_section_utts']} pc={part['n_pc_remarks']} ph={part['n_ph_remarks']}",
                  flush=True)
        city, state = names.get(fips, ("", ""))
        out_obj = {
            "fips": fips,
            "city": city,
            "state": state,
            "n_meetings": len(city_meetings),
            "meetings": city_meetings,
        }
        out_path = OUT_DIR / f"{fips}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        rec = {
            "fips": fips,
            "city": city,
            "state": state,
            "years": years,
            "n_meetings_in_file": len(city_meetings),
            **dict(st),
            "file": f"{fips}.json",
        }
        city_stats[fips] = rec
        add_stats(grand, st)
        grand["n_cities"] += 1

    readme = {
        "description": (
            "Inferred public-comment / public-hearing utterances for the IC2S2 57-city "
            "deployment corpus. Section flags come from meeting_structure; remark flags "
            "come from within_section_predictions.predicted_is_remark."
        ),
        "fields": {
            "is_public_comment": "1 = inferred citizen remark inside a Public Comment window",
            "is_public_hearing": "1 = inferred citizen remark inside a Public Hearing window",
            "is_public_comment_section": "1 = utterance falls in a recovered Public Comment stage",
            "is_public_hearing_section": "1 = utterance falls in a recovered Public Hearing stage",
        },
        "notes": [
            "Only utterances inside a recovered PC or PH window are included.",
            "Trigger fields (is_*_start_trigger / is_*_end_trigger) are omitted.",
            "Section windows include every matching stage in meeting_structure, not only the first.",
            "predicted_is_remark was originally run on the first matching PC/PH window per meeting; later windows may have section=1 and remark=0.",
            "city/state come from localgov.playlist (FIPS crosswalk), not from the Whisper transcripts.",
        ],
        "totals": dict(grand),
        "cities": city_stats,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(readme, ensure_ascii=False, indent=2), encoding="utf-8")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(OUT_DIR.iterdir()):
            if p.is_file():
                zf.write(p, arcname=p.name)
    print("wrote", ZIP_PATH, "bytes", ZIP_PATH.stat().st_size, flush=True)
    print("totals", dict(grand), flush=True)


if __name__ == "__main__":
    main()
