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

## Week 3 — Fine-Tuning

### Task 3.2 — LoRA Fine-Tune GPT-2
**What was done:**
- Wrote `scripts/finetune_gpt2_lora.py` with argparse CLI (`--dataset`, `--output`, `--epochs`, `--rank`, `--batch_size`, `--lr`)
- MPS-first device detection: MPS → CUDA → CPU, with `use_mps_device=True` and `no_cuda=True` passed to `TrainingArguments` for Apple Silicon compatibility
- LoRA config: `r=8`, `lora_alpha=16`, `target_modules=["c_attn"]`, `lora_dropout=0.1` → 294,912 trainable params (0.24% of 124M)
- Fixed a `transformers 4.57` incompatibility: `DataCollatorForLanguageModeling` fails when the dataset already stores a variable-length `labels` column (calls `tokenizer.pad()` on it). Replaced with a custom `CLMCollator` that strips pre-stored labels, pads `input_ids`/`attention_mask`, then recreates `labels` with padding positions masked to -100
- Fixed `evaluation_strategy` → `eval_strategy` rename in `TrainingArguments` (transformers ≥4.50)
- `PerplexityCallback`: prints `exp(eval_loss)` after each epoch; records `eval_perplexity` in trainer_state.json
- Post-training evaluation: loads fresh pretrained GPT-2 (no LoRA) and the saved LoRA adapter separately, runs all 15 Week-1 Telugu sentences, saves `report/finetune_comparison.csv`
- `extract_perplexity_curve()`: parses `trainer_state.json` log_history → `report/perplexity_curve.csv`
- MPS troubleshooting block documented in script header (OOM → reduce batch/rank; op errors → fall back to CPU)
- Training: 5 epochs, lr=2e-4 cosine, warmup=100 steps, `load_best_model_at_end=True`
- Model saved to `models/gpt2_lora_finetuned/` (adapter weights only — ~1.2MB)

- Local MPS training killed after ~23 min at 3% (ETA ~55 hours) — shifted to Colab T4
- Created `notebooks/colab_lora_finetune.ipynb`: mounts Drive, clones repo, installs deps, runs `prepare_dataset.py` + `finetune_gpt2_lora.py` on T4, saves model + CSVs back to Drive
- `fp16` auto-enabled on CUDA (T4), disabled on MPS/CPU
- **Colab timing fix (Task 3.2 follow-up):** Original Colab run at `batch_size=16` on full 237k training set was taking 5-8 hours (T4 real throughput ~10-15 it/s → 74k steps). Fixed by:
  - Added `--max_train_samples` flag to `finetune_gpt2_lora.py` to cap dataset size
  - Added `dataloader_num_workers=4` and `dataloader_pin_memory` to `TrainingArguments`
  - Updated Colab notebook cell to `--batch_size 64 --max_train_samples 50000`
  - Expected training time with fixes: **~20 minutes** on T4 (50k samples → 3,906 steps/epoch at batch=64)

**Training results (Colab T4, 50k samples, 4 epochs):**

| Epoch | Train Loss | Val Loss | Val PPL |
|-------|-----------|----------|---------|
| 1     | 4.9085    | 4.7241   | 112.63  |
| 2     | 4.7648    | 4.5626   | 95.83   |
| 3     | 4.6977    | 4.4878   | 88.93   |
| 4     | 4.6607    | 4.4566   | **86.20** |

**Before/After PPL on 15 Telugu sentences:**
- All 15 sentences improved (lower PPL after fine-tuning)
- Avg pretrained GPT-2 PPL: ~4,244
- Avg finetuned PPL: ~481
- Best improvement: `"adi evaru chepparu"` 9,634 → 344 (96.4% drop)
- PPL still high in absolute terms — expected given tokenizer mismatch and domain gap. Improvement trend is the key result.

**Key files:**
- `scripts/finetune_gpt2_lora.py`
- `notebooks/colab_lora_finetune.ipynb`
- `models/gpt2_lora_finetuned/` — adapter weights (~1.2 MB, `.safetensors`)
- `report/finetune_comparison.csv`
- `report/perplexity_curve.csv`

---

### Task 3.1 — Prepare CLM Training Dataset
**What was done:**
- Wrote `scripts/prepare_dataset.py` with argparse CLI (`--input`, `--output`, `--tokenizer`, `--max_length`)
- `--tokenizer` accepts both HF model IDs (e.g. `gpt2`) and local BPE directories (e.g. `tokenizers/tokenizer_codemixed`) — detected by checking for `vocab.json` presence
- Custom BPE directories are wrapped as `PreTrainedTokenizerFast` to get HF dataset API compatibility
- Worked around a `tokenizers==0.22.2` / `transformers` version mismatch: `PreTrainedTokenizerFast` tries to call `enable_truncation(direction=...)` which 0.22.x doesn't support — fixed by calling without truncation args and slicing `input_ids[:max_length]` manually in the BPE tokenize function
- Tokenizes for CLM: `labels = input_ids.copy()` (shift handled by `DataCollatorForLanguageModeling` at training time)
- 80/10/10 split with `seed=42` for reproducibility
- Verification block decodes 3 train examples and warns on high UNK rate
- Stats block reports avg token length, max length, truncation rate; warns if >30%
- Saves HuggingFace `DatasetDict` to disk via `save_to_disk()`
- Writes `report/dataset_preview_{name}.txt` with first 20 decoded train examples
- Added `data/clm_dataset_*/` to `.gitignore`

**Results:**

| Dataset | Tokenizer | Train | Val | Test | Truncation |
|---|---|---|---|---|---|
| `clm_dataset_gpt2` | GPT-2 | 237,522 | 29,690 | 29,691 | 56.1% ⚠ |
| `clm_dataset_codemixed` | custom BPE | 237,522 | 29,690 | 29,691 | 45.4% ⚠ |

56% truncation rate for GPT-2 vs 45% for custom BPE — consistent with custom tokenizer's lower fertility (1.63 vs 2.99 tokens/word). The high truncation rate is due to multi-sentence scraped blobs in the corpus; a pack_sequences strategy would recover that context.

**Key files:**
- `scripts/prepare_dataset.py`
- `report/dataset_preview_clm_dataset_gpt2.txt`
- `report/dataset_preview_clm_dataset_codemixed.txt`
- `.gitignore` (updated — excludes `data/clm_dataset_*/`)

---

### Task 3.3 — LoRA Fine-Tune GPT-2 with Custom BPE Tokenizer
**What was done:**
- Wrote `scripts/finetune_gpt2_custom_tok.py` — mirrors `finetune_gpt2_lora.py` but swaps in the custom 12k BPE tokenizer
- Key differences from Task 3.2:
  - Loads `tokenizers/tokenizer_codemixed` (fertility 1.63, identical to romanized_only) wrapped as `PreTrainedTokenizerFast`
  - Re-tokenizes `cleaned_data.txt` from scratch with manual truncation (tokenizers 0.22.x `enable_truncation` bug workaround)
  - Resizes GPT-2 embedding layer 50,257 → 12,000; new rows randomly initialised
  - Same LoRA config: r=8, lora_alpha=16, target_modules=["c_attn"]
  - Same training fixes: `CLMCollator`, `eval_strategy`, no MPS flags, fp16 auto-enabled on CUDA
- Produces **three-way comparison** across: pretrained GPT-2, finetuned orig-tok, finetuned custom-tok
  - Saves `report/custom_tok_comparison.csv` and `report/perplexity_curve_custom_tok.csv`
  - PPL values noted as non-comparable across tokenizers (custom tok has lower fertility → fewer tokens → mechanically lower per-token loss)
- Created `notebooks/colab_custom_tok_finetune.ipynb` for T4 Colab run:
  - Extra cell copies previous `gpt2_lora_finetuned` from Drive for 3-way comparison
  - Runs with `--batch_size 64 --max_train_samples 50000` (~20 min expected on T4)
- **Mac M3 vs Colab:** M3 MPS estimated ~110 min; Colab T4 estimated ~20 min. Colab chosen.

**Training results (Colab T4, 50k samples, 4 epochs):**

| Epoch | Train Loss | Val Loss | Val PPL |
|-------|-----------|----------|---------|
| 1     | 8.1524    | 8.1060   | 3314.19 |
| 2     | 8.1603    | 8.0878   | 3254.37 |
| 3     | 8.1280    | 8.0754   | 3214.55 |
| 4     | 8.1081    | 8.0683   | **3191.65** |

Val PPL is much higher than the orig-tok run (86.2) — expected. Two reasons:
1. Embedding layer was randomly initialised (12k vocab, no pretrained signal)
2. PPL is not directly comparable across tokenizers (custom tok = fewer tokens per sentence = different loss scale)

**Three-way comparison highlights:**
- Custom-tok top-5 predictions now include actual Telugu words: `mariyu`, `telugu`, `yokka`, `ala`, `alu` — the model has learned the domain vocabulary
- Orig-tok finetuned still wins on raw PPL (same tokenizer as pretrained baseline, direct comparison valid)
- Custom-tok adapter is 74MB (vs 1.2MB for orig-tok) — expected: resizing embeddings 50,257→12,000 saves the full new embedding matrix as trainable params alongside the LoRA weights

**Key files:**
- `scripts/finetune_gpt2_custom_tok.py`
- `notebooks/colab_custom_tok_finetune.ipynb`
- `models/gpt2_lora_custom_tok/` — adapter + resized embeddings (~74MB, gitignored)
- `report/custom_tok_comparison.csv`
- `report/perplexity_curve_custom_tok.csv`

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
