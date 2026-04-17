# Project History

Chronological log of completed tasks for the Romanized Telugu Language Model project.

---

## Week 1 — Baseline Evaluation

### Task 1.1 — Project Scaffolding
**What was done:**
- Created the project directory structure: `data/`, `tokenizers/`, `models/`, `notebooks/`, `scripts/`, `report/`, `demo/`, `configs/`
- Wrote `requirements.txt` with core dependencies: `torch`, `transformers`, `tokenizers`, `datasets`, `accelerate`, `peft`, `safetensors`, `huggingface_hub`, `numpy`, `pandas`, `tqdm`, `jupyter`, `ipykernel`
- Wrote `README.md` documenting the project goal, folder layout, and setup instructions
- Created `.gitignore` excluding `.venv/`, model checkpoints, tokenizer artifacts, raw data, and macOS junk

**Key files:**
- `README.md`
- `requirements.txt`
- `.gitignore`

---

### Task 1.2 — Colab Smoke Test
**What was done:**
- Created `notebooks/colab_gpt2_smoke_test.ipynb` to verify GPU access, auto-detect the project root by walking up from the kernel CWD until `README.md` is found, load GPT-2, run a forward pass, and confirm the folder structure exists
- Saved smoke test result to `report/gpt2_smoke_test_result.json` (device: mps, logits shape confirmed)

**Key files:**
- `notebooks/colab_gpt2_smoke_test.ipynb`
- `report/gpt2_smoke_test_result.json`

---

### Task 1.3 — GPT-2 Baseline Evaluation
**What was done:**
- Wrote `scripts/baseline_eval.py` to evaluate stock GPT-2 (117M) on 15 Romanized Telugu sentences and 5 English sentences
- Measured three things per sentence: tokenization (subword fragmentation), top-5 next-token predictions from final-position logits, and perplexity via causal LM cross-entropy loss
- Saved results to `report/results.csv`

**Key findings:**
- Average Telugu perplexity: ~4,600 vs English ~128 — a **~36× gap**
- Telugu avg tokens/sentence: ~8.1 vs English ~5.4 — **~1.5× fragmentation overhead**
- Worst case: `"nenu maatladatam nerchukuntunna"` → 12 tokens for a 6-word sentence
- Top-5 predictions for Telugu were all punctuation or single ASCII chars — model is doing byte-guessing, not word prediction
- Established the failure baseline and quantitative target for fine-tuning (goal: close the PPL gap to <2×)

**Key files:**
- `scripts/baseline_eval.py`
- `report/results.csv`

---

### Task 1.4 — mGPT Baseline Evaluation + Comparison
**What was done:**
- Refactored `baseline_eval.py` to guard all execution code under `if __name__ == "__main__":` so sentence lists can be imported without triggering GPT-2 load
- Also fixed the CSV output path to write to `report/results.csv` relative to project root instead of cwd
- Wrote `scripts/mgpt_eval.py` — mirrors `baseline_eval.py` exactly but runs on `ai-forever/mGPT`, with automatic fallback to `bigscience/bloom-560m` if mGPT fails to load
  - Uses `device_map="cpu"` and `torch_dtype=torch.float16` to keep RAM usage manageable (~1GB)
  - Imports sentence lists directly from `baseline_eval` — identical sentences for a valid comparison
  - Saves to `report/mgpt_results.csv` with identical column schema
- Wrote `scripts/compare.py` — loads both CSVs and prints:
  - Per-sentence table with GPT-2 vs mGPT token counts and perplexity, flagging sentences where mGPT wins (`fewer-tokens`, `lower-ppl`)
  - Summary: mean PPL and tokens/sentence per model per language
  - Token fertility (tokens/word) on the Telugu set for both models
  - Win counts: how many sentences mGPT beats GPT-2 on each metric

**Key files:**
- `scripts/baseline_eval.py` (modified)
- `scripts/mgpt_eval.py`
- `scripts/compare.py`

---

## Week 2 — Data Pipeline

### Task 2.1 — WhatsApp Data Cleaning Pipeline
**What was done:**
- Audited all 298 chat files to determine actual format: `[YYYY-MM-DD HH:MM] Sender: message` (~99% of lines), with a small number of legacy `DD/MM/YY, HH:MM AM/PM - Sender: message` lines
- Discovered the corpus contains ~50K multi-line continuation messages (message bodies that wrap across multiple lines with no timestamp prefix) — implemented joining logic to reassemble them before processing
- Wrote `scripts/clean_data.py` with full argparse CLI (`--input`, `--output`, `--report`); accepts a single file or a directory
- Encoding: tries `utf-8-sig` first, falls back to `latin-1` with a warning
- Filtering: removes URL-only messages, emoji-only messages (via `emoji` library), system messages (`<Media omitted>`, deleted messages, encryption notices), and messages under 3 words after normalisation
- Anonymisation: collects all unique senders in order of first appearance, maps to `USER_A`, `USER_B`, ...; replaces Indian mobile numbers (`(\+91[\s-]?)?[6-9]\d{9}`) with `[PHONE]`
- Normalisation: lowercase, strip punctuation (preserving apostrophes and intra-word hyphens), collapse multiple spaces
- Outputs `data/processed/cleaned_data.txt` (one sentence per line) and `report/cleaning_report.txt`
- Added `emoji` to `requirements.txt`; added `data/processed/` to `.gitignore`

**Corpus stats (full 298-file run):**
- Total messages parsed: 325,478
- After filtering: 296,903 (91.2% kept)
- Filtered breakdown: 25,178 too short | 2,948 emoji-only | 449 URL-only | 0 system
- Unique senders: 14,853
- Output size: 373 MB, 296,903 lines

**Key files:**
- `scripts/clean_data.py`
- `report/cleaning_report.txt`
- `.gitignore` (updated — excludes `data/processed/`)
- `requirements.txt` (updated — added `emoji`)

---

### Task 1.5 — Git Initialization & History Setup
**What was done:**
- Initialized git repository (`git init`)
- Added `.claude/` to `.gitignore`
- Created initial commit with all Week 1 work
- Created this `History.md` file
- Established workflow: after every task, commit code and update `History.md`

**Key files:**
- `.gitignore` (updated)
- `History.md`
