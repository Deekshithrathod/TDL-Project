"""
mgpt_eval.py
============
Week 1 second baseline: evaluate mGPT (ai-forever/mGPT) on the same Romanized
Telugu and English sentences used in baseline_eval.py.  Falls back to
bigscience/bloom-560m if mGPT fails to load.

Usage:
    python scripts/mgpt_eval.py

Outputs:
    - Formatted table to stdout
    - report/mgpt_results.csv  (language | sentence | tokenization | top5_predictions | perplexity)
"""

import math
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reuse sentence lists from baseline_eval — keep comparison valid
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from baseline_eval import TELUGU_SENTENCES, ENGLISH_SENTENCES  # noqa: E402

# ---------------------------------------------------------------------------
# Model loading (mGPT → bloom-560m fallback)
# ---------------------------------------------------------------------------

PRIMARY_MODEL = "ai-forever/mGPT"
FALLBACK_MODEL = "bigscience/bloom-560m"

def load_model_and_tokenizer(model_name: str):
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


try:
    model, tokenizer = load_model_and_tokenizer(PRIMARY_MODEL)
    MODEL_USED = PRIMARY_MODEL
except Exception as e:
    print(f"mGPT failed ({e}). Falling back to {FALLBACK_MODEL} ...")
    model, tokenizer = load_model_and_tokenizer(FALLBACK_MODEL)
    MODEL_USED = FALLBACK_MODEL

print(f"Model loaded: {MODEL_USED}\n")

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def get_tokenization(sentence: str) -> list[str]:
    return tokenizer.tokenize(sentence)


def get_top5_predictions(sentence: str) -> list[str]:
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]
    top5_ids = torch.topk(last_logits, k=5).indices
    decoded = []
    for idx in top5_ids:
        token = tokenizer.decode([idx.item()]).strip()
        decoded.append(token if token else repr(tokenizer.convert_ids_to_tokens(idx.item())))
    return decoded


def compute_perplexity(sentence: str) -> float:
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return math.exp(loss.item())


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

records = []

def evaluate_batch(sentences: list[str], lang: str):
    for sentence in sentences:
        tokens = get_tokenization(sentence)
        top5 = get_top5_predictions(sentence)
        ppl = compute_perplexity(sentence)
        records.append({
            "language": lang,
            "sentence": sentence,
            "num_tokens": len(tokens),
            "tokenization": " | ".join(tokens),
            "top5_predictions": ", ".join(top5),
            "perplexity": round(ppl, 2),
        })
        print(
            f"[{lang}] {sentence!r}\n"
            f"  tokens ({len(tokens)}): {tokens}\n"
            f"  top-5 : {top5}\n"
            f"  PPL   : {ppl:.2f}\n"
        )


print("=" * 70)
print("ROMANIZED TELUGU SENTENCES")
print("=" * 70)
evaluate_batch(TELUGU_SENTENCES, "Telugu")

print("=" * 70)
print("ENGLISH SENTENCES")
print("=" * 70)
evaluate_batch(ENGLISH_SENTENCES, "English")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

df = pd.DataFrame(records)

report_dir = Path(__file__).parent.parent / "report"
report_dir.mkdir(exist_ok=True)
csv_path = report_dir / "mgpt_results.csv"

df[["language", "sentence", "tokenization", "top5_predictions", "perplexity"]].to_csv(
    csv_path, index=False
)
print(f"\nResults saved to {csv_path}\n")

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

telugu_df = df[df["language"] == "Telugu"]
english_df = df[df["language"] == "English"]

avg_ppl_telugu = telugu_df["perplexity"].mean()
avg_ppl_english = english_df["perplexity"].mean()
avg_tok_telugu = telugu_df["num_tokens"].mean()
avg_tok_english = english_df["num_tokens"].mean()

telugu_word_counts = [len(s.split()) for s in TELUGU_SENTENCES]
tokens_per_word = telugu_df["num_tokens"].values / telugu_word_counts

print("=" * 70)
print(f"Model used: {MODEL_USED}")
print("=" * 70)
print(f"Avg perplexity  (Telugu) : {avg_ppl_telugu:.2f}")
print(f"Avg perplexity  (English): {avg_ppl_english:.2f}")
print(f"PPL ratio (Telugu/English): {avg_ppl_telugu / avg_ppl_english:.2f}x")
print()
print(f"Avg tokens/sentence (Telugu) : {avg_tok_telugu:.1f}")
print(f"Avg tokens/sentence (English): {avg_tok_english:.1f}")
print(f"Avg tokens/word     (Telugu) : {tokens_per_word.mean():.2f}")
print()
worst = telugu_df.sort_values("num_tokens", ascending=False).head(3)[["sentence", "num_tokens"]].values.tolist()
print("Worst tokenized Telugu sentences (most fragments):")
for sent, ntok in worst:
    print(f"  [{ntok} tokens] {sent!r}")
