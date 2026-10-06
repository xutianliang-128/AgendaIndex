# AgendaIndex

**Structure-first public-remark extraction** from long, heterogeneous city-council transcripts.

AgendaIndex first recovers a TOC-like agenda tree over a diarized meeting, then runs localized remark inference only inside public-participation windows (Public Comment / Public Hearing and city-specific aliases). This repository contains the **core inference components** used for section extraction, within-window classification, and city-year batch runs.

> Paper / talk: *AgendaIndex: Structure-First Public-Remark Extraction* (IC2S2 2026).

---

## Pipeline

```
meeting JSON  ──►  Stage 1: extract_meeting_structure
                         │
                         ▼
                  agenda stages with start_index / end_index
                         │
                         ▼
                  section_taxonomy  (alias + exclude rules;
                                    all matching windows, not just the first)
                         │
                         ▼
                  Stage 2: classify utterances inside PC / PH windows
                         │
                         ▼
                  is_public_comment / is_public_hearing predictions
```

**Stage 1** asks an LLM to segment the transcript into agenda stages (Roll Call, Public Comment, Public Hearing, …), keeping each city’s own wording and emitting repeated windows separately (e.g. reserved vs general time, one hearing per item).

**Stage 2** labels which utterances inside those windows are actual member-of-the-public remarks (vs chair/clerk management, staff presentation, or council deliberation).

Recommended classification setup from our bake-off (see `scripts/ab_context_strategies.py` and `scripts/infer_ic2s2_best.py`):

| Setting | Value |
|--------|--------|
| Model | `gpt-4.1-mini` |
| Prompt | v2 (explicit negatives for council / staff deliberation) |
| Context | whole-window batch labeling |
| Output | per-line `index: 0/1` (chunk size 100) |
| Window typing | `auto` — content-based PC/PH typing only for cities that split public comment by agenda scope (e.g. Lansing) |

Production entry points still expose the simpler per-utterance classifier in `agendaindex/meeting_structure.py`; the batch strategy lives in `scripts/infer_ic2s2_best.py` / `scripts/ab_context_strategies.py`.

---

## Repository layout

```
AgendaIndex/
├── agendaindex/                      # Core library
│   ├── meeting_structure.py        # Stage extraction + within-section classify
│   ├── section_taxonomy.py         # PC/PH aliases, excludes, multi-window indices
│   ├── transcript_loader.py        # Meeting JSON / WhisperX loaders
│   ├── input_meeting.py            # Meeting I/O helpers
│   ├── minutes_context.py          # Optional official-minutes hints
│   ├── utils.py / llm_backend.py   # OpenAI / local Qwen routing
│   ├── extract_public_comment.py   # Alternate direct / tree extract path
│   └── page_index.py               # Legacy tree utilities (PDF-oriented)
├── run_agendaindex.py                # Single-meeting CLI (structure or labels)
├── run_rag_by_fibs.py              # City-year batch over localgov tree
├── run_aa_section_extract.py       # Structure extract helper
├── run_aa_classify_within_section.py
├── run_all_tests_evaluate.py       # Evaluate on annotated *_test.json files
├── compute_public_duration.py      # Duration stats from predictions or GT
├── scripts/
│   ├── infer_ic2s2_best.py         # Best-config inference + IC2S2 scoring
│   ├── ab_context_strategies.py    # Context A/B/C bake-off
│   ├── ab_classifier_models.py     # Model × prompt A/B
│   ├── run_year_all_cities.py      # Resumable multi-city year driver
│   └── export_inferred_pc_ph.py    # Export predicted PC/PH utterances
├── docs/section_name_by_city.md    # Observed PC/PH stage-name aliases
└── requirements.txt
```

**Not included** (kept private / too large for this repo): full annotated all_splits corpus, multi-year localgov transcripts, result caches, and deployment DB helpers.

---

## Setup

```bash
git clone git@github.com:xutianliang-128/AgendaIndex.git
cd AgendaIndex
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set CHATGPT_API_KEY
```

Environment:

| Variable | Meaning |
|----------|---------|
| `LLM_BACKEND` | `openai` (default) or `qwen` |
| `CHATGPT_API_KEY` | OpenAI API key (also accepts `OPENAI_API_KEY`) |
| `QWEN_MODEL` | Path to local Qwen weights when `LLM_BACKEND=qwen` |

---

## Quick start

### 1. Extract agenda structure for one meeting

Meeting JSON shape: `{ "MEETING_KEY": [ {speaker, text, start, end, ...}, ... ] }`.

```bash
export CHATGPT_API_KEY=sk-...
export LLM_BACKEND=openai

python run_agendaindex.py \
  --meeting_path /path/to/meeting.json \
  --task structure \
  --model gpt-4o-2024-11-20
# → results/<basename>_meeting_structure.json
```

### 2. Classify public remarks (simple path)

```bash
python run_agendaindex.py \
  --meeting_path /path/to/meeting.json \
  --task public_comment \
  --extract_mode direct \
  --model gpt-4o-2024-11-20
```

Or structure → within-section classify:

```bash
python run_aa_section_extract.py \
  --meeting_path /path/to/meeting.json \
  --results_dir results

python run_aa_classify_within_section.py \
  --meeting_path /path/to/meeting.json \
  --structure_path results/<basename>_meeting_structure.json
```

### 3. City-year batch (localgov layout)

Expects transcripts at `{localgov_root}/{year}/{fibs}/transcripts/`.

```bash
python run_rag_by_fibs.py \
  --fibs 00674 \
  --year 2023 \
  --localgov_root /path/to/localgov \
  --results_dir results \
  --workers 8
```

### 4. Best-config IC2S2-style scoring

`scripts/infer_ic2s2_best.py` runs the winning batch classifier (C + per-line + chunk 100 + `gpt-4.1-mini` + v2 rules) and optional content-based window typing (`--window-typing auto`). Point it at your annotated split directory and structure cache (see script docstring).

---

## Section taxonomy

`agendaindex/section_taxonomy.py` is the single source of truth for which stage names count as Public Comment vs Public Hearing. It:

- matches **all** windows in a meeting (not only the first hit);
- covers aliases such as *Public Participation*, *Citizens Communication*, *Call to the Public*, *Reserved / General / Extended Time*;
- excludes procedural headings (*Public Comment Rules*, *Setting Public Hearings*, *Speaker Registration*, …).

Observed names across cities: [`docs/section_name_by_city.md`](docs/section_name_by_city.md).

---

## Input format

Minimal utterance fields for inference:

```json
{
  "AP_08_07_23": [
    {
      "speaker": "SPEAKER_00",
      "text": "Good evening, members of council.",
      "start": 12.4,
      "end": 15.1
    }
  ]
}
```

Optional ground-truth fields for evaluation: `is_public_comment`, `is_public_hearing` (or `*_gt` variants used by some scripts).

---

## Citation

If you use AgendaIndex in research, please cite the IC2S2 2026 extended abstract / presentation:

```
Xu et al. AgendaIndex: Structure-First Public-Remark Extraction. IC2S2 2026.
```

---

## License

Research code. Contact the authors before redistribution of any annotated evaluation data (not shipped in this repository).
