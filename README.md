# TDL Project — Romanized Telugu Language Model

Building a language model that handles Romanized Telugu (Telugu typed in Latin script, as used in WhatsApp). GPT-2 baseline shows extreme fragmentation and perplexity (500–12,000+) on Telugu vs English (16–290). The project fine-tunes GPT-2 with a custom BPE tokenizer trained on a WhatsApp chat corpus.

## Folder Structure

```
04-01-TDL-Project/
├── data/
│   ├── whatsapp_chats/     # 298 raw WhatsApp chat exports (training corpus)
│   ├── raw/                # other raw inputs
│   └── processed/          # cleaned / tokenized data outputs
├── tokenizers/             # trained tokenizer artifacts
├── models/                 # model checkpoints
├── notebooks/              # Jupyter / Colab notebooks
│   └── colab_gpt2_smoke_test.ipynb
├── scripts/                # standalone Python scripts
│   └── baseline_eval.py    # Week 1: GPT-2 baseline evaluation
├── report/                 # results, logs, summaries
│   ├── results.csv
│   └── gpt2_smoke_test_result.json
├── configs/                # training configs / hyperparameters
├── demo/                   # demo assets and example outputs
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Week 1 — Baseline

`scripts/baseline_eval.py` evaluates GPT-2 on Romanized Telugu vs English.
Measures tokenization fragmentation, top-5 predictions, and perplexity.

```bash
python scripts/baseline_eval.py
# outputs to report/results.csv
```

## Colab Smoke Test

`notebooks/colab_gpt2_smoke_test.ipynb` verifies GPU access, creates the folder
structure, loads GPT-2, and runs a forward pass. Works locally (Jupyter) or in
Google Colab — the project root is detected automatically by walking up from the
kernel's CWD until `README.md` is found.
