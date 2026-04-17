"""
compare.py
==========
Side-by-side comparison of GPT-2 vs mGPT baseline results.

Usage:
    python scripts/compare.py

Reads:
    report/results.csv       (GPT-2 baseline)
    report/mgpt_results.csv  (mGPT / bloom-560m baseline)

Prints:
    - Per-sentence table: sentence | gpt2_tokens | mgpt_tokens | gpt2_ppl | mgpt_ppl | flags
    - Summary: mean perplexity and token fertility per model, broken down by language
"""

from pathlib import Path

import pandas as pd

REPORT_DIR = Path(__file__).parent.parent / "report"
GPT2_CSV = REPORT_DIR / "results.csv"
MGPT_CSV = REPORT_DIR / "mgpt_results.csv"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

gpt2_df = pd.read_csv(GPT2_CSV)
mgpt_df = pd.read_csv(MGPT_CSV)

# Derive token counts from the pipe-separated tokenization column
gpt2_df["num_tokens"] = gpt2_df["tokenization"].str.split(" | ").str.len()
mgpt_df["num_tokens"] = mgpt_df["tokenization"].str.split(" | ").str.len()

merged = gpt2_df[["language", "sentence", "num_tokens", "perplexity"]].merge(
    mgpt_df[["sentence", "num_tokens", "perplexity"]],
    on="sentence",
    suffixes=("_gpt2", "_mgpt"),
)

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def flag_row(row) -> str:
    flags = []
    if row["num_tokens_mgpt"] < row["num_tokens_gpt2"]:
        flags.append("fewer-tokens")
    if row["perplexity_mgpt"] < row["perplexity_gpt2"]:
        flags.append("lower-ppl")
    return ", ".join(flags) if flags else "-"


merged["flags"] = merged.apply(flag_row, axis=1)

# ---------------------------------------------------------------------------
# Per-sentence table
# ---------------------------------------------------------------------------

COL_W = 38

def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


header = (
    f"{'sentence':<{COL_W}} {'lang':<8} "
    f"{'tok_gpt2':>8} {'tok_mgpt':>8} "
    f"{'ppl_gpt2':>10} {'ppl_mgpt':>10}  flags"
)
sep = "-" * len(header)

print(sep)
print(header)
print(sep)

for _, row in merged.iterrows():
    print(
        f"{truncate(row['sentence'], COL_W):<{COL_W}} "
        f"{row['language']:<8} "
        f"{int(row['num_tokens_gpt2']):>8} {int(row['num_tokens_mgpt']):>8} "
        f"{row['perplexity_gpt2']:>10.1f} {row['perplexity_mgpt']:>10.1f}  "
        f"{row['flags']}"
    )

print(sep)

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

for lang in ["Telugu", "English"]:
    sub = merged[merged["language"] == lang]
    print(f"\n{lang}:")
    print(f"  GPT-2  — mean PPL: {sub['perplexity_gpt2'].mean():>9.1f}   mean tokens/sent: {sub['num_tokens_gpt2'].mean():.1f}")
    print(f"  mGPT   — mean PPL: {sub['perplexity_mgpt'].mean():>9.1f}   mean tokens/sent: {sub['num_tokens_mgpt'].mean():.1f}")

# Token fertility (tokens per word) on Telugu set
telugu = merged[merged["language"] == "Telugu"].copy()
word_counts = telugu["sentence"].str.split().str.len()
fertility_gpt2 = (telugu["num_tokens_gpt2"] / word_counts).mean()
fertility_mgpt = (telugu["num_tokens_mgpt"] / word_counts).mean()
print(f"\nTelugu token fertility (tokens/word):")
print(f"  GPT-2 : {fertility_gpt2:.2f}")
print(f"  mGPT  : {fertility_mgpt:.2f}")

# Count sentences where mGPT wins on each metric
fewer_tok = (merged["num_tokens_mgpt"] < merged["num_tokens_gpt2"]).sum()
lower_ppl = (merged["perplexity_mgpt"] < merged["perplexity_gpt2"]).sum()
total = len(merged)
print(f"\nmGPT wins (fewer tokens)  : {fewer_tok}/{total} sentences")
print(f"mGPT wins (lower ppl)     : {lower_ppl}/{total} sentences")
