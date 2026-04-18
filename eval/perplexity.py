"""
perplexity.py
=============
Compute per-token perplexity for a model on the Week-3 held-out test split.

Usage:
    python eval/perplexity.py --model gpt2
    python eval/perplexity.py --model mgpt
    python eval/perplexity.py --model gpt2-finetuned-orig
    python eval/perplexity.py --model gpt2-finetuned-custom
    python eval/perplexity.py --model gpt2 --max_samples 200

Supported model keys:
    gpt2                  – pretrained GPT-2, no fine-tuning
    mgpt                  – pretrained mGPT (ai-forever/mGPT)
    gpt2-finetuned-orig   – GPT-2 + LoRA, original GPT-2 tokenizer
    gpt2-finetuned-custom – GPT-2 + LoRA, custom 12k BPE tokenizer

PPL = exp(total_NLL / total_tokens) — corpus-level, weighted by sentence length.
Note: PPL values are not directly comparable across tokenizers (different
      token granularity), but are still meaningful within-tokenizer diagnostics.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent

MODELS = {
    "gpt2":                  "gpt2",
    "mgpt":                  "ai-forever/mGPT",
    "gpt2-finetuned-orig":   str(ROOT / "models/gpt2_lora_finetuned"),
    "gpt2-finetuned-custom": str(ROOT / "models/gpt2_lora_custom_tok"),
}

MAX_LEN = 128


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _restore_custom_embeddings(model, adapter_path: Path):
    """
    After merge_and_unload(), PEFT doesn't copy wte/lm_head back from the
    adapter safetensors.  Load them explicitly so the trained 12k embeddings
    are actually used.
    """
    from safetensors.torch import load_file
    state = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")
    wte_key = "base_model.model.transformer.wte.weight"
    lm_head_key = "base_model.model.lm_head.weight"
    if wte_key in state:
        model.transformer.wte.weight.data.copy_(state[wte_key])
    if lm_head_key in state:
        # Only copy if not tied to wte (avoid double-write when tied)
        if model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr():
            model.lm_head.weight.data.copy_(state[lm_head_key])


def load_model_and_tokenizer(model_key: str):
    """Return (model, tokenizer) for the given model key."""
    from peft import PeftModel

    if model_key == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2")

    elif model_key == "mgpt":
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT")

    elif model_key == "gpt2-finetuned-orig":
        adapter_path = ROOT / "models/gpt2_lora_finetuned"
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained("gpt2")
        model = PeftModel.from_pretrained(base, str(adapter_path))
        model = model.merge_and_unload()

    elif model_key == "gpt2-finetuned-custom":
        adapter_path = ROOT / "models/gpt2_lora_custom_tok"
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained("gpt2")
        base.resize_token_embeddings(len(tokenizer))
        peft_model = PeftModel.from_pretrained(base, str(adapter_path))
        model = peft_model.merge_and_unload()
        # PEFT doesn't restore wte/lm_head from adapter_model.safetensors after
        # merge_and_unload; load them manually so the custom embeddings are used.
        _restore_custom_embeddings(model, adapter_path)

    else:
        sys.exit(f"Unknown model key: {model_key!r}. Choose from: {list(MODELS)}")

    return model, tokenizer


def get_test_sentences(max_samples: int) -> list[str]:
    """Reproduce the Week-3 held-out test split from cleaned_data.txt."""
    corpus = ROOT / "data/processed/cleaned_data.txt"
    if not corpus.exists():
        sys.exit(f"Corpus not found: {corpus}")

    with open(corpus, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    ds = Dataset.from_dict({"text": lines})
    split1 = ds.train_test_split(test_size=0.2, seed=42)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=42)
    test_lines = split2["test"]["text"]

    if max_samples and max_samples < len(test_lines):
        # Fixed seed for reproducibility
        import random
        rng = random.Random(42)
        test_lines = rng.sample(test_lines, max_samples)

    return test_lines


def compute_perplexity(model, tokenizer, sentences: list[str], device) -> float:
    model.eval()
    model.to(device)

    total_nll = 0.0
    total_tokens = 0
    skipped = 0

    for sent in tqdm(sentences, desc="Computing PPL", leave=False):
        enc = tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            skipped += 1
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)

        loss = outputs.loss.item()
        if math.isnan(loss) or math.isinf(loss):
            skipped += 1
            continue

        n_tokens = input_ids.shape[1]
        total_nll += loss * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    ppl = math.exp(total_nll / total_tokens)
    if skipped:
        print(f"  (skipped {skipped} sentences with invalid loss)")
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Compute perplexity for a model.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS),
        help="Model variant to evaluate.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Max test sentences to use (default: 500). Use 0 for all ~29k.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    print(f"  Vocab size: {len(tokenizer):,}")

    print("Loading test sentences...")
    sentences = get_test_sentences(args.max_samples)
    print(f"  Using {len(sentences):,} test sentences.")

    ppl = compute_perplexity(model, tokenizer, sentences, device)
    print(f"\nPerplexity [{args.model}]: {ppl:.2f}")
    return ppl


if __name__ == "__main__":
    main()
