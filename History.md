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
