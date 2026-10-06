#!/usr/bin/env python3
"""Re-run section extraction + within-section inference for every city in a year.

Resumable: a city is skipped when its predictions file already exists and was
written after this script's taxonomy cutoff, so the run can be interrupted and
restarted. Each city runs in its own `run_rag_by_fibs.py` subprocess; `--workers`
controls LLM concurrency inside a city, `--city-parallel` how many cities run at
once.

Usage:
  python scripts/run_year_all_cities.py --year 2019 --workers 24 --city-parallel 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCALGOV = Path(os.environ.get(
    "LOCALGOV_ROOT",
    "/home/shared/turbo_videos/processed_videos/localgov",
))
ENV_RUNNER = ROOT / "scripts" / "_run_with_env.py"


def discover_cities(year: int) -> list[tuple[str, int]]:
    """Return [(fibs, n_transcripts)] for every numeric city dir with data that year."""
    base = LOCALGOV / str(year)
    out = []
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        if not d.name.isdigit():
            continue
        tdir = d / "transcripts"
        if not tdir.is_dir():
            continue
        n = len([p for p in tdir.iterdir() if p.suffix == ".json"])
        if n:
            out.append((d.name, n))
    return out


def already_done(results_dir: Path, fibs: str, year: int) -> bool:
    path = results_dir / f"{fibs}_{year}_within_section_predictions.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return isinstance(payload, dict) and "predictions" in payload


def run_city(fibs: str, year: int, args, log_dir: Path) -> dict:
    log_path = log_dir / f"{fibs}_{year}.log"
    cmd = [
        sys.executable,
        str(ENV_RUNNER),
        str(ROOT / "run_rag_by_fibs.py"),
        "--fibs", fibs,
        "--year", str(year),
        "--results_dir", args.results_dir,
        "--workers", str(args.workers),
        "--model", args.model,
    ]
    t0 = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return {
        "fibs": fibs,
        "rc": proc.returncode,
        "secs": round(time.time() - t0, 1),
        "log": str(log_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--results_dir", default=str(ROOT / "results"))
    ap.add_argument("--model", default="gpt-4o-2024-11-20")
    ap.add_argument("--workers", type=int, default=24, help="LLM threads within one city")
    ap.add_argument("--city-parallel", type=int, default=3, help="Cities processed at once")
    ap.add_argument("--only", nargs="*", default=None, help="Restrict to these FIBS ids")
    ap.add_argument("--force", action="store_true", help="Re-run cities that already have output")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / f"run_logs_{args.year}"
    log_dir.mkdir(exist_ok=True)

    cities = discover_cities(args.year)
    if args.only:
        wanted = set(args.only)
        cities = [c for c in cities if c[0] in wanted]

    pending, skipped = [], []
    for fibs, n in cities:
        if not args.force and already_done(results_dir, fibs, args.year):
            skipped.append(fibs)
        else:
            pending.append((fibs, n))

    total_meetings = sum(n for _, n in pending)
    print(f"year={args.year}  cities with data={len(cities)}  "
          f"already done={len(skipped)}  to run={len(pending)}  meetings={total_meetings}")
    if args.list_only:
        for fibs, n in pending:
            print(f"  {fibs}  {n} transcripts")
        return

    # Largest cities first so the tail of the run is cheap stragglers.
    pending.sort(key=lambda x: -x[1])

    done = 0
    failures = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.city_parallel) as pool:
        futs = {pool.submit(run_city, fibs, args.year, args, log_dir): (fibs, n)
                for fibs, n in pending}
        for fut in as_completed(futs):
            fibs, n = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((fibs, repr(exc)))
                res = {"rc": -1, "secs": 0}
            done += 1
            if res["rc"] != 0:
                failures.append((fibs, f"rc={res['rc']} see {res.get('log')}"))
            mins = (time.time() - t_start) / 60
            print(f"[{done}/{len(pending)}] {fibs} ({n} mtgs) rc={res['rc']} "
                  f"{res['secs']}s | elapsed {mins:.1f}m", flush=True)

    print(f"\nFinished in {(time.time() - t_start)/60:.1f} min. failures={len(failures)}")
    for fibs, why in failures:
        print(f"  FAIL {fibs}: {why}")


if __name__ == "__main__":
    main()
