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

## Week 2 — Data Pipeline & Tokenizer Training

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

### Task 2.3 — Tokenizer Fertility Analysis
**What was done:**
- Wrote `scripts/fertility_analysis.py` with argparse CLI (`--input`, `--sample`, `--codemixed-dir`, `--romanized-dir`, `--outdir`)
- Loads all 4 tokenizers: GPT-2 (HF), mGPT (HF, graceful skip on failure), codemixed BPE, romanized_only BPE
- Samples 1000 sentences from cleaned corpus at seed=42 for reproducible comparison
- Computes two metrics per tokenizer over the sample:
  - **Fertility**: avg tokens-per-word (sentence token count ÷ word count)
  - **Continued-word %**: fraction of words (encoded independently) that split into >1 token
- Unified `compute_metrics()` function handles both HF (`add_special_tokens=False`) and BPE (`.encode().ids`) APIs
- Builds 25-word spotlight table showing exact token splits per tokenizer
- Saves `report/fertility_results.csv`, `report/spotlight_splits.csv`, `report/fertility_report.txt` (pretty-printed with `tabulate`)
- Added `tabulate` to `requirements.txt`

**Results (1000 sentences, seed=42):**

| Tokenizer      | Avg Fertility | CWP (%) |
|----------------|--------------|---------|
| gpt2           | 2.990        | 83.3%   |
| mgpt           | 2.715        | 85.2%   |
| codemixed      | **1.633**    | 85.0%   |
| romanized_only | **1.633**    | 85.0%   |

Key spotlight observations:
- `unnaru` → 1 token (custom) vs 2 (GPT-2/mGPT)
- `ledu`, `meeru`, `nenu` → 1 token each (custom) vs 2–3 (baselines)
- `cheppukovadam` → 2 tokens (custom) vs 6 (GPT-2)
- `okkaసారి` → mGPT correctly clusters Telugu Unicode bytes into 4 tokens; GPT-2 and custom BPE byte-explode into 14/13 pieces respectively
- codemixed and romanized_only produce near-identical splits — the Telugu filter had little effect at 12k vocab with this corpus size

**Key files:**
- `scripts/fertility_analysis.py`
- `report/fertility_results.csv`
- `report/spotlight_splits.csv`
- `report/fertility_report.txt`
- `requirements.txt` (updated — added `tabulate`)

---

### Task 2.2 — Train Custom BPE Tokenizers
**What was done:**
- Wrote `scripts/train_tokenizer.py` with argparse CLI (`--input`, `--outdir`)
- Loads NLTK `words` corpus (234k words) at startup; auto-downloads if missing
- English-ratio filter: keeps sentences where <70% of words appear in the NLTK English dictionary — identifies Telugu-dominated sentences for the second tokenizer
- Trains `ByteLevelBPETokenizer` (vocab_size=12000, min_frequency=2, 4 special tokens) on two corpora:
  - **tokenizer_codemixed**: full 296,903-sentence corpus
  - **tokenizer_romanized_only**: 274,035-sentence Telugu-dominated subset (92.3%)
- Writes filtered sentences to a `tempfile`, trains, then deletes temp file
- Reload check: reloads both tokenizers from disk via `from_file()` and asserts token output matches pre-save
- Verification block: prints token splits for 5 test sentences across both tokenizers
- Added `nltk` to `requirements.txt`; updated `.gitignore` comment

**Training results:**
- Both tokenizers reached full 12,000 vocab target
- Key improvement over GPT-2 baseline: `"meeru ela unnaru"` → 3 tokens (vs 6 with GPT-2); `"nenu ikkadiki vastunnanu"` → 5 tokens (vs 10 with GPT-2)
- `"naku Telugu vastundi"` still splits `Telugu` due to capital T not seen in lowercased training data — expected

**Key files:**
- `scripts/train_tokenizer.py`
- `tokenizers/tokenizer_codemixed/vocab.json` + `merges.txt`
- `tokenizers/tokenizer_romanized_only/vocab.json` + `merges.txt`
- `requirements.txt` (updated — added `nltk`)

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
