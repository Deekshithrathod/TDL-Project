"""
topk_accuracy.py
================
Compute top-1, top-3, top-5 next-word prediction accuracy on eval_set.json.

Usage:
    python eval/topk_accuracy.py --model gpt2
    python eval/topk_accuracy.py --model mgpt
    python eval/topk_accuracy.py --model gpt2-finetuned-orig
    python eval/topk_accuracy.py --model gpt2-finetuned-custom

For each prefix in eval_set.json the script:
  1. Tokenizes the prefix.
  2. Runs a forward pass to get the next-token logits.
  3. Takes the top-5 predicted token IDs.
  4. Checks whether the correct next_word appears in top-1 / top-3 / top-5.

Subword handling
----------------
Many words are split across multiple tokens by GPT-2 / mGPT.  A prediction
counts as correct when ANY of the following conditions holds:

  (a) The decoded token exactly matches next_word (single-token word).
  (b) The decoded token exactly matches the first subword of next_word
      (i.e. tokenizer.tokenize(next_word)[0] decoded matches the prediction).

This is equivalent to asking: "does the model predict the correct first token
of the next word?" — the standard metric used in causal LM next-word evals.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
EVAL_SET_PATH = ROOT / "eval/eval_set.json"
MAX_LEN = 128

MODELS = [
    "gpt2",
    "mgpt",
    "gpt2-finetuned-orig",
    "gpt2-finetuned-custom",
]


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
        if model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr():
            model.lm_head.weight.data.copy_(state[lm_head_key])


def load_model_and_tokenizer(model_key: str):
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
        _restore_custom_embeddings(model, adapter_path)

    else:
        sys.exit(f"Unknown model key: {model_key!r}. Choose from: {MODELS}")

    return model, tokenizer


def get_correct_first_token_id(tokenizer, next_word: str) -> int | None:
    """
    Return the token ID that the model should predict as the first token of
    next_word.  Handles leading-space conventions (GPT-2 BPE prepends Ġ).
    Returns None if tokenization yields no tokens.
    """
    # Try with a space prefix (GPT-2 style) first, then without
    for candidate in [" " + next_word, next_word]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            return ids[0]
    return None


def topk_accuracy(model, tokenizer, eval_set: list[dict], device, k: int = 5) -> dict:
    model.eval()
    model.to(device)

    correct_at = {1: 0, 3: 0, 5: 0}
    total = 0

    for item in tqdm(eval_set, desc="Top-k accuracy", leave=False):
        prefix = item["prefix"]
        next_word = item["next_word"]

        target_id = get_correct_first_token_id(tokenizer, next_word)
        if target_id is None:
            continue

        enc = tokenizer(
            prefix,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN - 1,  # Leave room for the next token
        )
        input_ids = enc["input_ids"].to(device)

        if input_ids.shape[1] == 0:
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        # Logits for the position AFTER the last prefix token
        next_logits = outputs.logits[0, -1, :]  # (vocab_size,)
        top_k_ids = torch.topk(next_logits, k=min(k, next_logits.shape[0])).indices.tolist()

        total += 1
        for cutoff in [1, 3, 5]:
            if target_id in top_k_ids[:cutoff]:
                correct_at[cutoff] += 1

    if total == 0:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "n": 0}

    return {
        "top1": correct_at[1] / total,
        "top3": correct_at[3] / total,
        "top5": correct_at[5] / total,
        "n": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute top-k next-word accuracy.")
    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS,
        help="Model variant to evaluate.",
    )
    parser.add_argument(
        "--eval_set",
        default=str(EVAL_SET_PATH),
        help="Path to eval_set.json (default: eval/eval_set.json).",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        sys.exit(f"eval_set.json not found at {eval_path}. Run eval/run_eval.py first.")

    with open(eval_path, encoding="utf-8") as f:
        eval_set = json.load(f)
    print(f"Loaded {len(eval_set)} evaluation examples.")

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    print(f"  Vocab size: {len(tokenizer):,}")

    results = topk_accuracy(model, tokenizer, eval_set, device)
    print(f"\nTop-k accuracy [{args.model}]  (n={results['n']})")
    print(f"  Top-1: {results['top1']:.1%}")
    print(f"  Top-3: {results['top3']:.1%}")
    print(f"  Top-5: {results['top5']:.1%}")
    return results


if __name__ == "__main__":
    main()
