"""
run_eval.py
===========
Orchestrates the full evaluation pipeline across all four model variants,
writes eval/results.json, and prints a formatted comparison table.

Usage:
    python eval/run_eval.py                        # standard run
    python eval/run_eval.py --max_ppl_samples 200  # faster, fewer PPL sentences
    python eval/run_eval.py --skip_build           # reuse existing eval_set.json

Models evaluated
----------------
  gpt2                  – GPT-2 pretrained, no fine-tuning
  mgpt                  – mGPT pretrained, no fine-tuning
  gpt2-finetuned-orig   – GPT-2 + LoRA, original GPT-2 tokenizer
  gpt2-finetuned-custom – GPT-2 + LoRA, custom 12k BPE tokenizer

Output
------
  eval/eval_set.json   – 100 prefix/next_word pairs from Week-3 test split
  eval/results.json    – full numeric results for all models
  stdout               – formatted table

Note on PPL comparability
--------------------------
Perplexity is computed per-tokenizer, so values for gpt2-finetuned-custom
(12k vocab) are not directly comparable to the GPT-2 / mGPT numbers.
Top-k accuracy is word-level and comparable across all models.
"""

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset

ROOT = Path(__file__).parent.parent
EVAL_DIR = ROOT / "eval"
EVAL_SET_PATH = EVAL_DIR / "eval_set.json"
RESULTS_PATH = EVAL_DIR / "results.json"

MODEL_KEYS = [
    "gpt2",
    "mgpt",
    "gpt2-finetuned-orig",
    "gpt2-finetuned-custom",
]

MODEL_LABELS = {
    "gpt2":                  "GPT-2 pretrained",
    "mgpt":                  "mGPT pretrained",
    "gpt2-finetuned-orig":   "GPT-2 fine-tuned (orig tok)",
    "gpt2-finetuned-custom": "GPT-2 fine-tuned (custom tok)",
}


# ---------------------------------------------------------------------------
# Eval-set construction
# ---------------------------------------------------------------------------

def build_eval_set(n: int = 100) -> list[dict]:
    """Sample n prefix/next_word pairs from the Week-3 test split."""
    corpus = ROOT / "data/processed/cleaned_data.txt"
    if not corpus.exists():
        sys.exit(f"Corpus not found: {corpus}")

    print("Building eval set from cleaned_data.txt …")
    with open(corpus, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    ds = Dataset.from_dict({"text": lines})
    split1 = ds.train_test_split(test_size=0.2, seed=42)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=42)
    test_lines = split2["test"]["text"]
    print(f"  Test split: {len(test_lines):,} sentences")

    rng = random.Random(42)
    candidates = rng.sample(test_lines, min(n * 5, len(test_lines)))

    eval_set = []
    for sent in candidates:
        words = sent.split()
        if len(words) < 3:
            continue
        next_word = words[-1]
        if len(next_word) < 2:
            continue
        prefix = " ".join(words[:-1])
        eval_set.append({"prefix": prefix, "next_word": next_word})
        if len(eval_set) >= n:
            break

    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(eval_set)} examples → {EVAL_SET_PATH}")
    return eval_set


# ---------------------------------------------------------------------------
# Lazy imports from sibling modules
# ---------------------------------------------------------------------------

def _import_eval_modules():
    import importlib.util, sys as _sys

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    ppl_mod = _load("perplexity", EVAL_DIR / "perplexity.py")
    topk_mod = _load("topk_accuracy", EVAL_DIR / "topk_accuracy.py")
    return ppl_mod, topk_mod


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run full model evaluation.")
    parser.add_argument(
        "--max_ppl_samples",
        type=int,
        default=500,
        help="Max test sentences for perplexity (default: 500). Set 0 for all.",
    )
    parser.add_argument(
        "--skip_build",
        action="store_true",
        help="Skip building eval_set.json (reuse existing file).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS,
        default=MODEL_KEYS,
        help="Which models to evaluate (default: all four).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1 – Build eval set
    # ------------------------------------------------------------------
    if args.skip_build and EVAL_SET_PATH.exists():
        with open(EVAL_SET_PATH, encoding="utf-8") as f:
            eval_set = json.load(f)
        print(f"Reusing existing eval set: {len(eval_set)} examples.")
    else:
        eval_set = build_eval_set(n=100)

    # ------------------------------------------------------------------
    # Step 2 – Import eval modules
    # ------------------------------------------------------------------
    ppl_mod, topk_mod = _import_eval_modules()
    device = ppl_mod.get_device()
    print(f"\nDevice: {device}\n")

    # ------------------------------------------------------------------
    # Step 3 – Evaluate each model
    # ------------------------------------------------------------------
    print("Loading test sentences for perplexity …")
    test_sents = ppl_mod.get_test_sentences(args.max_ppl_samples)
    print(f"  Using {len(test_sents):,} sentences.\n")

    results = {}

    for model_key in args.models:
        label = MODEL_LABELS[model_key]
        print(f"{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        try:
            model, tokenizer = ppl_mod.load_model_and_tokenizer(model_key)

            # Perplexity
            print("  → Perplexity …")
            ppl = ppl_mod.compute_perplexity(model, tokenizer, test_sents, device)
            print(f"     PPL = {ppl:.2f}")

            # Top-k accuracy
            print("  → Top-k accuracy …")
            acc = topk_mod.topk_accuracy(model, tokenizer, eval_set, device)
            print(f"     Top-1={acc['top1']:.1%}  Top-3={acc['top3']:.1%}  Top-5={acc['top5']:.1%}  (n={acc['n']})")

            results[model_key] = {
                "label": label,
                "ppl": round(ppl, 2),
                "top1": round(acc["top1"], 4),
                "top3": round(acc["top3"], 4),
                "top5": round(acc["top5"], 4),
                "n_topk": acc["n"],
                "n_ppl": len(test_sents),
            }

        except Exception as exc:
            print(f"  ERROR evaluating {model_key}: {exc}")
            results[model_key] = {
                "label": label,
                "ppl": None,
                "top1": None,
                "top3": None,
                "top5": None,
                "error": str(exc),
            }

        finally:
            # Free GPU/MPS memory between models
            try:
                del model
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print()

    # ------------------------------------------------------------------
    # Step 4 – Write results.json
    # ------------------------------------------------------------------
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {RESULTS_PATH}\n")

    # ------------------------------------------------------------------
    # Step 5 – Print formatted table
    # ------------------------------------------------------------------
    def fmt(v, pct=False):
        if v is None:
            return "  N/A  "
        if pct:
            return f"{v:.1%}"
        return f"{v:.2f}"

    col_w = 29
    print(f"{'Model':<{col_w}} | {'Test PPL':>8} | {'Top-1':>7} | {'Top-3':>7} | {'Top-5':>7}")
    print("-" * col_w + "-+-" + "-" * 8 + "-+-" + "-" * 7 + "-+-" + "-" * 7 + "-+-" + "-" * 7)

    for key in MODEL_KEYS:
        if key not in results:
            continue
        r = results[key]
        lbl = MODEL_LABELS[key]
        ppl_s = fmt(r.get("ppl"))
        t1 = fmt(r.get("top1"), pct=True)
        t3 = fmt(r.get("top3"), pct=True)
        t5 = fmt(r.get("top5"), pct=True)
        print(f"{lbl:<{col_w}} | {ppl_s:>8} | {t1:>7} | {t3:>7} | {t5:>7}")

    print()
    print("Note: PPL for gpt2-finetuned-custom uses a 12k-vocab tokenizer")
    print("      and is not directly comparable to the other three models.")


if __name__ == "__main__":
    main()
