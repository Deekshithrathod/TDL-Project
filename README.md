# Romanized Telugu Language Model

Language modeling for Romanized Telugu — Telugu written in Latin script, heavily code-mixed with English and Hindi, as used in WhatsApp.

**Key result:** A multilingual pretrained model (mGPT) outperforms all fine-tuned GPT-2 variants, achieving 15% top-1 next-word accuracy vs 4% for GPT-2 fine-tuned with LoRA. Fine-tuning reduces perplexity 4× (362 → 92) but yields marginal accuracy gains due to GPT-2's English BPE tokenizer fragmenting Telugu words into subword stubs. A custom 12k BPE tokenizer achieves 45% lower fertility (1.63 vs 2.99 tokens/word) but its fine-tuned model collapses due to embedding misalignment from vocabulary resizing.

Full write-up: [`report/final_report.pdf`](report/final_report.pdf)

---

## Results Summary

| Model | Test PPL | Top-1 | Top-3 | Top-5 |
|-------|----------|-------|-------|-------|
| GPT-2 pretrained | 362.9 | 2.0% | 5.0% | 6.0% |
| mGPT pretrained | 213.3 | **15.0%** | **22.0%** | **25.0%** |
| GPT-2 fine-tuned (orig tok) | **92.3** | 4.0% | 8.0% | 13.0% |
| GPT-2 fine-tuned (custom tok) | 3,624.8 † | 1.0% | 1.0% | 1.0% |

*† Custom tok PPL uses a different vocabulary and is not directly comparable.*

---

## Repository Structure

```
04-01-TDL-Project/
├── data/
│   ├── whatsapp_chats/          # 298 raw WhatsApp chat exports (gitignored)
│   ├── processed/               # cleaned_data.txt — 296,903 lines, 373 MB (gitignored)
│   ├── clm_dataset_gpt2/        # HuggingFace DatasetDict, GPT-2 tokenized (gitignored)
│   └── clm_dataset_codemixed/   # HuggingFace DatasetDict, custom BPE tokenized (gitignored)
│
├── tokenizers/
│   ├── tokenizer_codemixed/     # 12k BPE vocab trained on full corpus
│   └── tokenizer_romanized_only/# 12k BPE vocab trained on Telugu-dominant subset
│
├── models/
│   ├── gpt2_lora_finetuned/     # LoRA adapter, original GPT-2 tokenizer (1.2 MB)
│   └── gpt2_lora_custom_tok/    # LoRA adapter + resized embeddings, custom tok (74 MB, gitignored)
│
├── scripts/
│   ├── baseline_eval.py         # Week 1: GPT-2 baseline on Telugu vs English
│   ├── mgpt_eval.py             # Week 1: mGPT baseline evaluation
│   ├── compare.py               # Week 1: compare GPT-2 vs mGPT results
│   ├── clean_data.py            # Week 2: WhatsApp chat cleaning pipeline
│   ├── train_tokenizer.py       # Week 2: train custom 12k BPE tokenizers
│   ├── fertility_analysis.py    # Week 2: tokenizer fertility comparison
│   ├── prepare_dataset.py       # Week 3: create CLM datasets for fine-tuning
│   ├── finetune_gpt2_lora.py    # Week 3: LoRA fine-tune, original tokenizer
│   └── finetune_gpt2_custom_tok.py  # Week 3: LoRA fine-tune, custom tokenizer
│
├── eval/
│   ├── eval_set.json            # 100 prefix/next_word pairs from test split
│   ├── perplexity.py            # Corpus-level PPL per model
│   ├── topk_accuracy.py         # Top-1/3/5 next-word accuracy
│   ├── run_eval.py              # Orchestrates full evaluation, writes results.json
│   └── results.json             # Final numeric results, all 4 models
│
├── analysis/
│   ├── extract_failures.py      # Run all models, record top-5 predictions + PPL
│   ├── predictions.json         # Model predictions for all 100 eval examples
│   ├── failures.json            # 30 failure cases with error categories
│   └── error_analysis.md        # Full written error analysis
│
├── notebooks/
│   ├── colab_gpt2_smoke_test.ipynb
│   ├── colab_lora_finetune.ipynb        # T4 Colab training, orig tokenizer
│   └── colab_custom_tok_finetune.ipynb  # T4 Colab training, custom tokenizer
│
├── report/
│   ├── final_report.md          # Report source (Markdown)
│   ├── final_report.pdf         # Submission-ready PDF
│   ├── finetune_comparison.csv  # Before/after PPL, 15 Telugu sentences
│   ├── fertility_results.csv    # Tokenizer fertility comparison
│   ├── perplexity_curve.csv     # Epoch-by-epoch training PPL (orig tok)
│   └── perplexity_curve_custom_tok.csv
│
├── History.md                   # Chronological task log (all 4 weeks)
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducing Results

**Run the full evaluation pipeline** (requires model weights in `models/`):
```bash
python eval/run_eval.py
# writes eval/results.json and prints the results table
```

**Run error analysis** (requires model weights):
```bash
python analysis/extract_failures.py
# writes analysis/predictions.json and analysis/failures.json
```

**Tokenizer fertility comparison** (no model weights needed):
```bash
python scripts/fertility_analysis.py \
  --input data/processed/cleaned_data.txt \
  --codemixed-dir tokenizers/tokenizer_codemixed \
  --romanized-dir tokenizers/tokenizer_romanized_only
# writes report/fertility_results.csv and report/fertility_report.txt
```

**GPT-2 baseline evaluation** (no training needed):
```bash
python scripts/baseline_eval.py   # report/results.csv
python scripts/mgpt_eval.py       # report/mgpt_results.csv
python scripts/compare.py         # prints comparison table
```

---

## Training from Scratch

**Step 1 — Clean data:**
```bash
python scripts/clean_data.py \
  --input data/whatsapp_chats/ \
  --output data/processed/cleaned_data.txt \
  --report report/cleaning_report.txt
```

**Step 2 — Train tokenizers:**
```bash
python scripts/train_tokenizer.py \
  --input data/processed/cleaned_data.txt \
  --outdir tokenizers/
```

**Step 3 — Prepare CLM datasets:**
```bash
python scripts/prepare_dataset.py \
  --input data/processed/cleaned_data.txt \
  --output data/clm_dataset_gpt2 \
  --tokenizer gpt2

python scripts/prepare_dataset.py \
  --input data/processed/cleaned_data.txt \
  --output data/clm_dataset_codemixed \
  --tokenizer tokenizers/tokenizer_codemixed
```

**Step 4 — Fine-tune** (recommended on Colab T4; ~20 min per run at batch=64, 50k samples):
```bash
# Original tokenizer
python scripts/finetune_gpt2_lora.py \
  --dataset data/clm_dataset_gpt2 \
  --output models/gpt2_lora_finetuned \
  --epochs 4 --batch_size 64 --max_train_samples 50000

# Custom tokenizer
python scripts/finetune_gpt2_custom_tok.py \
  --output models/gpt2_lora_custom_tok \
  --epochs 4 --batch_size 64 --max_train_samples 50000
```

Use the Colab notebooks in `notebooks/` for GPU-accelerated training on Google Colab (T4 recommended).

---

## Key Findings

- **mGPT is the right baseline** for code-mixed South Asian text. Its 60-language pretraining covers Hindi, transliterated registers, and informal code-mixing that monolingual GPT-2 fine-tuning cannot replicate.
- **Fine-tuning reduces perplexity significantly** (4× on the held-out set) but yields only marginal top-k accuracy gains because GPT-2's English BPE tokenizer fragments Telugu words into single-consonant subtokens — the model learns to predict stubs, not words.
- **Custom tokenizer fine-tuning requires more than LoRA.** Resizing GPT-2's 50k embedding table to 12k and updating only attention weights (LoRA) leaves the embeddings geometrically incoherent. The model collapses to a single high-frequency prediction. Fix: initialize custom embeddings from subword-averaged GPT-2 embeddings before LoRA fine-tuning, or retrain embeddings fully.
- **Code-switching at phrase boundaries defeats all models.** When the correct next word is in a different language than the preceding context, every model in this study fails with near-certainty.
