"""
train_tokenizer.py
==================
Week 2: Train two ByteLevelBPE tokenizers on the cleaned WhatsApp corpus.

  tokenizer 1 — codemixed:         full cleaned_data.txt (12 k vocab)
  tokenizer 2 — romanized_only:    sentences where ≤70 % of words are English

Both are saved as HuggingFace-compatible vocab.json + merges.txt pairs so they
can be loaded directly into a GPT-2 config for fine-tuning.

Usage:
    python scripts/train_tokenizer.py \\
        --input  data/processed/cleaned_data.txt \\
        --outdir tokenizers/
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import nltk
from tokenizers import ByteLevelBPETokenizer

# ---------------------------------------------------------------------------
# NLTK English wordlist — loaded once
# ---------------------------------------------------------------------------

def load_english_set() -> set[str]:
    try:
        from nltk.corpus import words as nltk_words
        nltk_words.words()          # test it's available
    except LookupError:
        print("Downloading NLTK 'words' corpus...")
        nltk.download("words", quiet=True)
        from nltk.corpus import words as nltk_words
    return set(w.lower() for w in nltk_words.words())


# ---------------------------------------------------------------------------
# Telugu-dominated sentence filter
# ---------------------------------------------------------------------------

def is_romanized_telugu_dominated(sentence: str, english_set: set[str],
                                   threshold: float = 0.70) -> bool:
    """
    Return True if < threshold fraction of words appear in the English dictionary.
    Empty sentences are excluded (return False).
    """
    words = sentence.split()
    if not words:
        return False
    english_count = sum(1 for w in words if w in english_set)
    return (english_count / len(words)) < threshold


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

VOCAB_SIZE    = 12_000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]

def train_bpe(files: list[str], save_dir: str, label: str) -> ByteLevelBPETokenizer:
    """Train a ByteLevelBPE tokenizer and save vocab.json + merges.txt."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nTraining {label} tokenizer → {save_dir}")

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=files,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )
    tok.save_model(save_dir)
    print(f"  Saved: {save_dir}/vocab.json + merges.txt")
    return tok


# ---------------------------------------------------------------------------
# Reload check
# ---------------------------------------------------------------------------

def reload_check(save_dir: str, reference_tok: ByteLevelBPETokenizer,
                 probe: str = "nenu ikkadiki vastunnanu") -> None:
    """Reload tokenizer from disk and assert output matches the trained one."""
    vocab  = os.path.join(save_dir, "vocab.json")
    merges = os.path.join(save_dir, "merges.txt")
    reloaded = ByteLevelBPETokenizer.from_file(vocab, merges)

    orig_tokens     = reference_tok.encode(probe).tokens
    reloaded_tokens = reloaded.encode(probe).tokens

    if orig_tokens != reloaded_tokens:
        raise RuntimeError(
            f"Reload mismatch for {save_dir}!\n"
            f"  original : {orig_tokens}\n"
            f"  reloaded : {reloaded_tokens}"
        )
    print(f"  Reload check passed ✓  ({save_dir})")


# ---------------------------------------------------------------------------
# Verification block
# ---------------------------------------------------------------------------

TEST_SENTENCES = [
    "nenu ikkadiki vastunnanu",
    "meeru ela unnaru",
    "okka nimisham agu",
    "I am coming there",
    "naku Telugu vastundi",
]

def run_verification(tok1: ByteLevelBPETokenizer,
                     tok2: ByteLevelBPETokenizer) -> None:
    print("\n" + "=" * 70)
    print("VERIFICATION — token splits on test sentences")
    print("=" * 70)
    for sent in TEST_SENTENCES:
        for name, tok in [("codemixed    ", tok1), ("romanized_only", tok2)]:
            tokens = tok.encode(sent).tokens
            print(f"  [{name}] '{sent}'\n"
                  f"            → {tokens}  (n={len(tokens)})")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train two ByteLevelBPE tokenizers on the cleaned WhatsApp corpus."
    )
    parser.add_argument(
        "--input",
        default="data/processed/cleaned_data.txt",
        help="Path to cleaned_data.txt (one sentence per line)",
    )
    parser.add_argument(
        "--outdir",
        default="tokenizers",
        help="Parent directory for saved tokenizer subdirs (default: tokenizers/)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    outdir = Path(args.outdir)
    codemixed_dir    = str(outdir / "tokenizer_codemixed")
    romanized_dir    = str(outdir / "tokenizer_romanized_only")

    # ------------------------------------------------------------------
    # Load NLTK English set
    # ------------------------------------------------------------------
    print("Loading NLTK English wordlist...")
    english_set = load_english_set()
    print(f"  {len(english_set):,} English words loaded.")

    # ------------------------------------------------------------------
    # Read corpus + filter
    # ------------------------------------------------------------------
    print(f"\nReading corpus: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        all_sentences = [line.rstrip("\n") for line in f if line.strip()]

    total = len(all_sentences)
    print(f"  {total:,} sentences loaded.")

    print("Filtering for Romanized-Telugu-dominated sentences (<70 % English words)...")
    telugu_sentences = [
        s for s in all_sentences
        if is_romanized_telugu_dominated(s, english_set)
    ]
    n_telugu = len(telugu_sentences)
    pct = 100 * n_telugu / total if total else 0
    print(f"  {n_telugu:,} sentences pass the filter ({pct:.1f}%).")

    # ------------------------------------------------------------------
    # Tokenizer 1 — code-mixed (full corpus)
    # ------------------------------------------------------------------
    tok1 = train_bpe(
        files=[str(input_path)],
        save_dir=codemixed_dir,
        label="codemixed (full corpus)",
    )

    # ------------------------------------------------------------------
    # Tokenizer 2 — Romanized Telugu only (temp file, deleted after)
    # ------------------------------------------------------------------
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="romanized_only_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_f:
            tmp_f.write("\n".join(telugu_sentences))

        tok2 = train_bpe(
            files=[tmp_path],
            save_dir=romanized_dir,
            label="romanized_only (Telugu-dominated subset)",
        )
    finally:
        os.unlink(tmp_path)
        print(f"  Temp file removed: {tmp_path}")

    # ------------------------------------------------------------------
    # Reload checks
    # ------------------------------------------------------------------
    print("\nRunning reload checks...")
    reload_check(codemixed_dir, tok1)
    reload_check(romanized_dir, tok2)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    run_verification(tok1, tok2)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    vocab1 = tok1.get_vocab_size()
    vocab2 = tok2.get_vocab_size()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total sentences (full corpus):         {total:>10,}")
    print(f"Sentences passing Telugu filter:       {n_telugu:>10,}  ({pct:.1f}%)")
    print(f"Tokenizer 1 vocab size (actual):       {vocab1:>10,}")
    print(f"Tokenizer 2 vocab size (actual):       {vocab2:>10,}")
    print(f"\nSaved to:")
    print(f"  {codemixed_dir}/")
    print(f"  {romanized_dir}/")


if __name__ == "__main__":
    main()
