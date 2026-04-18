"""
prepare_dataset.py
==================
Week 3 — Prepare tokenized CLM dataset for fine-tuning.

Reads cleaned_data.txt, tokenizes for causal language modelling,
splits 80/10/10, and saves a HuggingFace DatasetDict to disk.

Supports two tokenizer modes:
  --tokenizer gpt2                             → AutoTokenizer (HF)
  --tokenizer ai-forever/mGPT                  → AutoTokenizer (HF)
  --tokenizer tokenizers/tokenizer_codemixed   → custom BPE (vocab.json + merges.txt)

Note on custom BPE: tokenizers==0.22.x and transformers have a version
mismatch where PreTrainedTokenizerFast cannot pass truncation=True (the
wrapper tries to call enable_truncation(direction=...) which 0.22.x
doesn't support). We work around this by calling the tokenizer without
truncation flags and slicing input_ids manually in the tokenize function.

Usage:
    python scripts/prepare_dataset.py \\
        --input   data/processed/cleaned_data.txt \\
        --output  data/clm_dataset_gpt2 \\
        --tokenizer gpt2 \\
        --max_length 128
"""

import argparse
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def is_custom_bpe(tokenizer_name: str) -> bool:
    """Return True if the tokenizer arg points to a local BPE directory."""
    p = Path(tokenizer_name)
    return p.is_dir() and (p / "vocab.json").exists() and (p / "merges.txt").exists()


def load_tokenizer(tokenizer_name: str):
    """
    Load tokenizer and return (tokenizer, mode).
    mode is 'hf' or 'bpe' — used to pick the right tokenize strategy.
    """
    if is_custom_bpe(tokenizer_name):
        from tokenizers import ByteLevelBPETokenizer
        raw = ByteLevelBPETokenizer.from_file(
            str(Path(tokenizer_name) / "vocab.json"),
            str(Path(tokenizer_name) / "merges.txt"),
        )
        tok = PreTrainedTokenizerFast(
            tokenizer_object=raw,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
        # For CLM, use </s> as pad so the model can still predict EOS
        tok.pad_token = tok.eos_token
        print(f"  Loaded custom BPE tokenizer from: {tokenizer_name}")
        print(f"  Vocab size: {tok.vocab_size:,}")
        return tok, "bpe"
    else:
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        # GPT-2 has no pad token; set to EOS so DataCollator doesn't crash
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"  Loaded HF tokenizer: {tokenizer_name}")
        print(f"  Vocab size: {tok.vocab_size:,}")
        return tok, "hf"


# ---------------------------------------------------------------------------
# Tokenisation — two strategies
# ---------------------------------------------------------------------------

def make_tokenize_fn(tokenizer, mode: str, max_length: int):
    """
    Return a batched tokenize function for dataset.map().

    HF mode:  use standard truncation=True — works fine.
    BPE mode: call without truncation args (avoids enable_truncation/direction
              bug in tokenizers 0.22.x) and slice manually.
    """
    if mode == "hf":
        def tokenize_hf(batch):
            out = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            out["labels"] = out["input_ids"].copy()
            return out
        return tokenize_hf

    else:  # bpe — manual truncation
        def tokenize_bpe(batch):
            out = tokenizer(batch["text"])
            truncated_ids = [ids[:max_length] for ids in out["input_ids"]]
            masks         = [m[:max_length]   for m   in out["attention_mask"]]
            return {
                "input_ids":      truncated_ids,
                "attention_mask": masks,
                "labels":         truncated_ids.copy(),
            }
        return tokenize_bpe


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_stats(dataset_split, max_length: int) -> dict:
    lengths = [len(x["input_ids"]) for x in dataset_split]
    n = len(lengths)
    avg_len   = sum(lengths) / n if n else 0
    max_len   = max(lengths)  if n else 0
    truncated = sum(1 for l in lengths if l == max_length)
    return {
        "n":         n,
        "avg_len":   avg_len,
        "max_len":   max_len,
        "truncated": truncated,
        "trunc_pct": 100 * truncated / n if n else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare tokenized CLM dataset for fine-tuning."
    )
    parser.add_argument(
        "--input",
        default="data/processed/cleaned_data.txt",
        help="Path to cleaned_data.txt",
    )
    parser.add_argument(
        "--output",
        default="data/clm_dataset_gpt2",
        help="Directory to save HuggingFace DatasetDict",
    )
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help=(
            "HF model name (e.g. 'gpt2', 'ai-forever/mGPT') "
            "or path to a local BPE directory with vocab.json + merges.txt"
        ),
    )
    parser.add_argument(
        "--max_length", type=int, default=128,
        help="Max token length per example (default: 128)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer, mode = load_tokenizer(args.tokenizer)

    # ------------------------------------------------------------------
    # Load raw text
    # ------------------------------------------------------------------
    print(f"\nReading corpus: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    total_sentences = len(lines)
    print(f"  {total_sentences:,} sentences loaded.")

    dataset = Dataset.from_dict({"text": lines})

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------
    print(f"\nTokenizing (max_length={args.max_length}, mode={mode})...")
    tokenize_fn = make_tokenize_fn(tokenizer, mode, args.max_length)
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenising",
    )
    print(f"  {len(tokenized):,} examples after tokenisation.")

    # ------------------------------------------------------------------
    # 80 / 10 / 10 split  (seed=42)
    # ------------------------------------------------------------------
    print("\nSplitting 80/10/10 (seed=42)...")
    split1 = tokenized.train_test_split(test_size=0.2, seed=42)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=42)
    final = DatasetDict({
        "train":      split1["train"],
        "validation": split2["train"],
        "test":       split2["test"],
    })
    sizes = {k: len(v) for k, v in final.items()}
    print(f"  Sizes: {sizes}")

    # ------------------------------------------------------------------
    # Verification block
    # ------------------------------------------------------------------
    print("\n--- Decoded examples (train) ---")
    for i in range(3):
        ids    = final["train"][i]["input_ids"]
        decoded = tokenizer.decode(ids)
        tokens  = tokenizer.convert_ids_to_tokens(ids[:15])
        unk_count = decoded.count(tokenizer.unk_token or "<unk>")
        print(f"[{i}] {decoded[:120]}")
        print(f"     tokens : {tokens}...")
        print(f"     length : {len(ids)}")
        if unk_count > len(ids) * 0.3:
            print(f"     WARNING: high UNK rate ({unk_count}/{len(ids)}) — check cleaned_data.txt")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    print("\nComputing stats (iterating train split)...")
    stats = compute_stats(final["train"], args.max_length)

    print("\n" + "=" * 60)
    print("DATASET STATS")
    print("=" * 60)
    print(f"Total sentences loaded:               {total_sentences:>10,}")
    print(f"After tokenisation:                   {len(tokenized):>10,}")
    print(f"Train / Val / Test sizes:             "
          f"{sizes['train']:,} / {sizes['validation']:,} / {sizes['test']:,}")
    print(f"Avg token length per sentence:        {stats['avg_len']:>10.1f}")
    print(f"Max token length:                     {stats['max_len']:>10}")
    print(f"Sentences truncated at {args.max_length}:           "
          f"{stats['truncated']:>6,}  ({stats['trunc_pct']:.1f}%)")

    if stats["trunc_pct"] > 30:
        print(
            "\n  ⚠ OBSERVATION: truncation rate is >30%. Consider increasing "
            "--max_length or concatenating sentences with EOS tokens between them "
            "(pack_sequences strategy) to avoid losing tail context during training."
        )

    # ------------------------------------------------------------------
    # Save dataset
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving dataset to {out_path} ...")
    final.save_to_disk(str(out_path))
    print("  Done.")

    # ------------------------------------------------------------------
    # Preview file — first 20 decoded train examples
    # ------------------------------------------------------------------
    report_dir = Path("report")
    report_dir.mkdir(exist_ok=True)
    preview_stem = Path(args.output).name  # e.g. clm_dataset_gpt2
    preview_path = report_dir / f"dataset_preview_{preview_stem}.txt"

    with open(preview_path, "w", encoding="utf-8") as pf:
        pf.write(f"Dataset preview — {preview_stem}\n")
        pf.write(f"Tokenizer: {args.tokenizer} | max_length: {args.max_length}\n")
        pf.write("=" * 70 + "\n\n")
        for i in range(min(20, len(final["train"]))):
            ids = final["train"][i]["input_ids"]
            pf.write(f"[{i:02d}] ({len(ids)} tokens)\n")
            pf.write(tokenizer.decode(ids) + "\n\n")

    print(f"Preview saved to {preview_path}")
    print(f"\nAll done. Dataset at: {out_path}/")


if __name__ == "__main__":
    main()
