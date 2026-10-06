# AgendaIndex

**Structure-first public-remark extraction** from long, heterogeneous city-council transcripts.

AgendaIndex first recovers a TOC-like agenda tree over a diarized meeting, then runs localized remark inference only inside public-participation windows (Public Comment / Public Hearing and city-specific aliases). This repository contains the **core inference components** used for section extraction, within-window classification, and city-year batch runs. Parts of the code were adopted from: https://github.com/VectifyAI/PageIndex

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

## Results (IC2S2 ten-city gold set)

Binary public-comment F1 per city, on the annotated 2023 test meetings.

| Model | SEA | OAK | RCH | AA | LS | RO | JS | AP | PE | IN | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4o (whole transcript) | 0.638 | 0.117 | 0.857 | 0.515 | 0.000 | 0.629 | 0.348 | 0.333 | 0.000 | 0.254 | 0.369 |
| RoBERTa (fine-tuned) | 0.929 | 0.717 | 0.000 | 0.642 | 0.587 | 0.811 | 0.723 | 0.222 | 0.000 | 0.247 | 0.488 |
| PublicSpeak | 0.929 | 0.785 | **1.000** | 0.894 | 0.612 | 0.811 | 0.870 | 0.222 | 0.247 | 0.000 | 0.637 |
| AgendaIndex (IC2S2 paper) | 0.920 | **0.866** | 0.857 | **0.978** | **0.731** | 0.911 | 0.941 | 0.889 | **0.862** | 0.345 | 0.830 |
| AgendaIndex (`gpt-4.1-mini` Stage 2) † | 0.967 | 0.759 | **1.000** | 0.918 | 0.647 | 0.912 | **0.987** | 0.842 | 0.840 | 0.510 | 0.838 |
| **AgendaIndex-SFT (Qwen3-4B LoRA Stage 2)** † | **1.000** | 0.853 | **1.000** | 0.930 | 0.643 | **0.957** | **0.987** | **1.000** | 0.763 | **0.535** | **0.867** |

The first four rows are from the IC2S2 extended abstract (Table 1). Rows marked † are produced by this repository on the test split (34 meetings, 2–4 per city), scored end-to-end over the whole meeting so utterances outside a predicted window count as misses. Both † rows use the same Stage 1 structures and heading-based windows and differ only in the Stage 2 classifier; per-city numbers rest on a handful of meetings and move by a few points between runs.

Pooled over the ten cities (precision / recall / F1, threshold 0.5):

| Stage 2 classifier | Public Comment | Public Hearing |
|---|---|---|
| Qwen3-4B zero-shot | 0.627 / 0.727 / 0.673 | 0.623 / 0.656 / 0.639 |
| `gpt-4.1-mini`, per-line batch | 0.754 / 0.816 / 0.784 | 0.831 / 0.674 / 0.745 |
| **Qwen3-4B + LoRA (SFT)** | 0.784 / 0.846 / **0.814** | 0.932 / 0.646 / **0.763** |

Fine-tuning mainly buys precision: the model learns to label staff presentations and council deliberation inside a hearing as 0. Hearing recall is capped by Stage 1, and Lansing hearing F1 is 0 in both † rows because Lansing holds its hearings under a "Public Comment" heading; `--window-typing auto` in `scripts/infer_ic2s2_best.py` fixes that and is not yet combined with the SFT classifier. The SFT model is trained on meetings from the same 37 cities as the test split, so these numbers do not measure transfer to unseen cities.

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
├── sft/                            # Stage 2 as a fine-tuned Qwen3-4B (LoRA)
│   ├── build_windows.py            # Per-utterance records inside predicted windows
│   ├── prompt.py                   # Classifier prompt (answer 1 / 0)
│   ├── train_lora.py               # LoRA training (torchrun, multi-GPU)
│   ├── score.py                    # p("1") scoring, base or +adapter
│   ├── gpt_baseline.py             # gpt-4.1-mini on the same windows
│   ├── evaluate.py                 # End-to-end / within-window, per city
│   └── run_scoring.sh
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

### 5. Stage 2 with a fine-tuned Qwen3-4B

Replaces the API classifier with a local LoRA model. Training labels are "spoken by a member of the public" (PC or PH gold) for every utterance inside a predicted window; the window decides PC vs PH.

```bash
python sft/build_windows.py --splits-dir DATA/all_splits \
  --structure-dir CACHE/structures --out-dir results/sft_data

torchrun --nproc_per_node 3 sft/train_lora.py \
  --data-dir results/sft_data --output-dir results/sft_qwen4b

torchrun --nproc_per_node 3 sft/score.py --data results/sft_data/test.jsonl \
  --adapter results/sft_qwen4b/adapter --out results/sft_eval/qwen_sft_test.json

python sft/evaluate.py --data-dir results/sft_data \
  --system qwen_sft=results/sft_eval/qwen_sft_test.json --out results/sft_eval/report.json
```

Defaults: LoRA r=16, alpha=32 on all attention and MLP projections, 2 epochs, lr 2e-4, effective batch 24, about an hour on 3× A6000. Scoring the test split takes about two minutes.

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
