"""
fertility_analysis.py
=====================
Week 2 — Tokenizer Fertility Analysis.

Compares four tokenizers on Romanized Telugu:
  1. GPT-2      (English-only BPE, 50k vocab)
  2. mGPT       (multilingual BPE, 100k vocab)
  3. codemixed  (custom BPE trained on full WhatsApp corpus, 12k vocab)
  4. romanized  (custom BPE trained on Telugu-dominated subset, 12k vocab)

Metrics
-------
  Fertility            : avg tokens-per-word across the sample.
                         Lower = more efficient tokenisation.
  Continued-word %     : fraction of words that are split into >1 token.
                         Lower = fewer broken words.

Outputs
-------
  report/fertility_results.csv
  report/spotlight_splits.csv
  report/fertility_report.txt

Usage
-----
    python scripts/fertility_analysis.py \\
        --input  data/processed/cleaned_data.txt \\
        --sample 1000
"""

import argparse
import random
import sys
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd
from tabulate import tabulate
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Spotlight words — 25 Romanized Telugu words + one Unicode code-mix case
# ---------------------------------------------------------------------------

SPOTLIGHT_WORDS = [
    "vastunnanu", "cheppadu",   "andukey",       "ikkadiki",    "unnaru",
    "chestunnanu","ledu",       "meeru",          "nenu",        "kavali",
    "chudandi",   "pampinchu",  "thelusukuntanu", "marchipoyanu","cheppukovadam",
    "okkaసారి",   "bayataki",   "pampinchanu",    "tirigi",      "vellipoindi",
    "chestam",    "undadu",     "kashtanga",      "nerchukunna", "matladataniki",
]

# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def load_tokenizers(codemixed_dir: str, romanized_dir: str):
    """
    Returns a dict  name → (tokenizer_object, type_str).
    type_str is 'hf' for HuggingFace AutoTokenizer, 'bpe' for
    ByteLevelBPETokenizer — used by compute_metrics to pick the right API.
    mGPT is attempted from HuggingFace Hub; skipped gracefully on failure.
    """
    toks: dict[str, tuple] = {}

    print("Loading GPT-2 tokenizer...")
    toks["gpt2"] = (AutoTokenizer.from_pretrained("gpt2"), "hf")

    print("Loading mGPT tokenizer...")
    try:
        toks["mgpt"] = (AutoTokenizer.from_pretrained("ai-forever/mGPT"), "hf")
    except Exception as e:
        print(f"  WARNING: mGPT failed ({e}) — skipping.", file=sys.stderr)

    print("Loading custom codemixed tokenizer...")
    toks["codemixed"] = (
        ByteLevelBPETokenizer.from_file(
            str(Path(codemixed_dir) / "vocab.json"),
            str(Path(codemixed_dir) / "merges.txt"),
        ),
        "bpe",
    )

    print("Loading custom romanized_only tokenizer...")
    toks["romanized_only"] = (
        ByteLevelBPETokenizer.from_file(
            str(Path(romanized_dir) / "vocab.json"),
            str(Path(romanized_dir) / "merges.txt"),
        ),
        "bpe",
    )
    return toks


# ---------------------------------------------------------------------------
# Unified encode helpers
# ---------------------------------------------------------------------------

def _sentence_token_count(tok, tok_type: str, sentence: str) -> int:
    """Return number of tokens for a whole sentence."""
    if tok_type == "hf":
        return len(tok.encode(sentence, add_special_tokens=False))
    else:  # bpe
        return len(tok.encode(sentence).ids)


def _word_tokens(tok, tok_type: str, word: str) -> list[str]:
    """Return token strings for a single word."""
    if tok_type == "hf":
        return tok.tokenize(word)
    else:  # bpe
        return tok.encode(word).tokens


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    tok, tok_type: str, sentences: list[str]
) -> dict[str, float]:
    """
    Compute avg fertility and continued-word % over a list of sentences.

    Fertility per sentence  = n_tokens / n_words
    Continued-word %        = fraction of words (encoded independently) that
                              produce more than one token.
    """
    total_tokens = 0
    total_words  = 0
    split_words  = 0

    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        n_tok = _sentence_token_count(tok, tok_type, sent)
        total_tokens += n_tok
        total_words  += len(words)

        for w in words:
            wt = _word_tokens(tok, tok_type, w)
            if len(wt) > 1:
                split_words += 1

    avg_fertility   = total_tokens / total_words if total_words else 0.0
    continued_word_pct = 100.0 * split_words / total_words if total_words else 0.0
    return {
        "avg_fertility":       round(avg_fertility,       4),
        "continued_word_pct":  round(continued_word_pct,  2),
        "total_words_seen":    total_words,
    }


# ---------------------------------------------------------------------------
# Spotlight table
# ---------------------------------------------------------------------------

def build_spotlight(
    toks: dict[str, tuple]
) -> list[dict]:
    """
    For each spotlight word, collect token split for each tokenizer.
    Returns list of dicts suitable for a DataFrame.
    """
    rows = []
    for word in SPOTLIGHT_WORDS:
        row = {"word": word}
        for name, (tok, tok_type) in toks.items():
            tokens = _word_tokens(tok, tok_type, word)
            row[name] = str(tokens)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fertility & continued-word analysis across 4 tokenizers."
    )
    parser.add_argument(
        "--input",
        default="data/processed/cleaned_data.txt",
        help="Path to cleaned_data.txt",
    )
    parser.add_argument(
        "--sample", type=int, default=1000,
        help="Number of sentences to sample (default: 1000; 0 = all)",
    )
    parser.add_argument(
        "--codemixed-dir",
        default="tokenizers/tokenizer_codemixed",
        help="Directory containing codemixed vocab.json + merges.txt",
    )
    parser.add_argument(
        "--romanized-dir",
        default="tokenizers/tokenizer_romanized_only",
        help="Directory containing romanized_only vocab.json + merges.txt",
    )
    parser.add_argument(
        "--outdir",
        default="report",
        help="Directory to write CSV and report (default: report/)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load tokenizers
    # ------------------------------------------------------------------ #
    toks = load_tokenizers(args.codemixed_dir, args.romanized_dir)

    # ------------------------------------------------------------------ #
    # Load + sample corpus
    # ------------------------------------------------------------------ #
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        all_sentences = [line.rstrip("\n") for line in f if line.strip()]

    random.seed(42)
    n = len(all_sentences)
    if args.sample and args.sample < n:
        sentences = random.sample(all_sentences, args.sample)
    else:
        sentences = all_sentences
    print(f"\nSampled {len(sentences):,} / {n:,} sentences (seed=42).")

    # ------------------------------------------------------------------ #
    # Compute metrics
    # ------------------------------------------------------------------ #
    print("\nComputing fertility & CWP...")
    results = []
    for name, (tok, tok_type) in toks.items():
        print(f"  {name} ...", end=" ", flush=True)
        m = compute_metrics(tok, tok_type, sentences)
        results.append({
            "tokenizer":          name,
            "avg_fertility":      m["avg_fertility"],
            "continued_word_pct": m["continued_word_pct"],
            "sample_size":        len(sentences),
        })
        print(f"fertility={m['avg_fertility']:.3f}  CWP={m['continued_word_pct']:.1f}%")

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------ #
    # Spotlight splits
    # ------------------------------------------------------------------ #
    print("\nBuilding spotlight table...")
    spotlight_rows = build_spotlight(toks)
    spotlight_df = pd.DataFrame(spotlight_rows)

    # ------------------------------------------------------------------ #
    # Save CSVs
    # ------------------------------------------------------------------ #
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results_csv  = outdir / "fertility_results.csv"
    spotlight_csv = outdir / "spotlight_splits.csv"

    results_df.to_csv(results_csv, index=False)
    spotlight_df.to_csv(spotlight_csv, index=False)
    print(f"\nSaved: {results_csv}")
    print(f"Saved: {spotlight_csv}")

    # ------------------------------------------------------------------ #
    # Pretty-print tables
    # ------------------------------------------------------------------ #
    fertility_table = tabulate(
        results_df,
        headers="keys",
        tablefmt="github",
        floatfmt=".3f",
        showindex=False,
    )

    # Spotlight: truncate token strings to 40 chars so the table fits
    spotlight_display = spotlight_df.copy()
    for col in spotlight_display.columns:
        if col != "word":
            spotlight_display[col] = spotlight_display[col].str[:45]
    spotlight_table = tabulate(
        spotlight_display,
        headers="keys",
        tablefmt="github",
        showindex=False,
    )

    report = f"""\
=======================================================================
  Tokenizer Fertility & Continued-Word Analysis
  Sample: {len(sentences):,} sentences  |  seed=42
=======================================================================

### Fertility & CWP Summary

{fertility_table}

Columns:
  avg_fertility      — tokens per word (lower = more efficient)
  continued_word_pct — % of words split into >1 token (lower = better)

-----------------------------------------------------------------------

### Word-Level Spotlight  ({len(SPOTLIGHT_WORDS)} Romanized Telugu words)

{spotlight_table}

-----------------------------------------------------------------------

## Analysis (fill in after reviewing results)

# TODO: Which tokenizer has the lowest fertility on Telugu words?
# TODO: Does romanized_only outperform codemixed on Telugu-heavy words?
# TODO: Any surprising splits in the spotlight words?
# TODO: How does okkaసారి split differently across byte-level vs mGPT?
# TODO: Which tokenizer would you recommend as the base for fine-tuning?
"""

    print("\n" + report)

    report_path = outdir / "fertility_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
