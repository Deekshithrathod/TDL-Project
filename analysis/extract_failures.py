"""
extract_failures.py
===================
For every example in eval/eval_set.json, run all four models and record:
  - top-5 predicted tokens (decoded)
  - whether top-1 / top-3 / top-5 matched the correct next word
  - per-example perplexity of the full sentence (prefix + next_word)

Saves analysis/predictions.json and analysis/failures.json.
Run from repo root:
    python3 analysis/extract_failures.py
"""

import gc
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from perplexity import _restore_custom_embeddings  # noqa: E402

EVAL_SET_PATH = ROOT / "eval/eval_set.json"
PRED_OUT = ROOT / "analysis/predictions.json"
FAIL_OUT = ROOT / "analysis/failures.json"

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
MAX_LEN = 128


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(model_key):
    from peft import PeftModel

    if model_key == "gpt2":
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained("gpt2")

    elif model_key == "mgpt":
        tok = AutoTokenizer.from_pretrained("ai-forever/mGPT")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT")

    elif model_key == "gpt2-finetuned-orig":
        ap = ROOT / "models/gpt2_lora_finetuned"
        tok = AutoTokenizer.from_pretrained(str(ap))
        tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained("gpt2")
        mdl = PeftModel.from_pretrained(base, str(ap)).merge_and_unload()

    elif model_key == "gpt2-finetuned-custom":
        ap = ROOT / "models/gpt2_lora_custom_tok"
        tok = AutoTokenizer.from_pretrained(str(ap))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained("gpt2")
        base.resize_token_embeddings(len(tok))
        mdl = PeftModel.from_pretrained(base, str(ap)).merge_and_unload()
        _restore_custom_embeddings(mdl, ap)
    else:
        sys.exit(f"Unknown model: {model_key}")
    return mdl, tok


def first_token_id(tokenizer, word):
    """Return token ID that should be predicted for `word`."""
    for candidate in [" " + word, word]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            return ids[0]
    return None


def run_model_on_eval(model, tokenizer, eval_set, device):
    model.eval()
    model.to(device)
    results = []

    for item in eval_set:
        prefix = item["prefix"]
        next_word = item["next_word"]

        # ---- top-k predictions ----
        enc = tokenizer(prefix, return_tensors="pt",
                        truncation=True, max_length=MAX_LEN - 1)
        input_ids = enc["input_ids"].to(device)

        target_id = first_token_id(tokenizer, next_word)

        top5_decoded = []
        in_top1 = in_top3 = in_top5 = False

        if input_ids.shape[1] > 0 and target_id is not None:
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[0, -1, :]
            top5_ids = torch.topk(logits, k=5).indices.tolist()
            top5_decoded = [
                tokenizer.decode([tid]).strip() for tid in top5_ids
            ]
            in_top1 = target_id in top5_ids[:1]
            in_top3 = target_id in top5_ids[:3]
            in_top5 = target_id in top5_ids[:5]

        # ---- per-example perplexity of the full sentence ----
        full = prefix + " " + next_word
        full_enc = tokenizer(full, return_tensors="pt",
                             truncation=True, max_length=MAX_LEN)
        full_ids = full_enc["input_ids"].to(device)
        ppl = None
        if full_ids.shape[1] >= 2:
            with torch.no_grad():
                loss = model(input_ids=full_ids, labels=full_ids).loss.item()
            if not (math.isnan(loss) or math.isinf(loss)):
                ppl = round(math.exp(loss), 2)

        results.append({
            "prefix":       prefix,
            "next_word":    next_word,
            "top1":         top5_decoded[0] if top5_decoded else "",
            "top5_decoded": top5_decoded,
            "in_top1":      in_top1,
            "in_top3":      in_top3,
            "in_top5":      in_top5,
            "ppl":          ppl,
        })

    return results


def main():
    device = get_device()
    print(f"Device: {device}")

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)
    print(f"Loaded {len(eval_set)} eval examples.")

    all_preds = {}   # model_key -> list of per-example dicts

    for key in MODEL_KEYS:
        print(f"\n--- {MODEL_LABELS[key]} ---")
        model, tokenizer = load_model_and_tokenizer(key)
        preds = run_model_on_eval(model, tokenizer, eval_set, device)
        all_preds[key] = preds
        print(f"  top-1 acc: {sum(p['in_top1'] for p in preds)/len(preds):.1%}")
        del model
        gc.collect()

    # Save all predictions
    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_OUT, "w", encoding="utf-8") as f:
        json.dump(all_preds, f, indent=2, ensure_ascii=False)
    print(f"\nSaved predictions → {PRED_OUT}")

    # Build failure list: examples wrong for at least 3 of 4 models
    # Sort by number of models failing, then by GPT-2 fine-tuned PPL descending
    failures = []
    for i, item in enumerate(eval_set):
        models_wrong = {k for k in MODEL_KEYS if not all_preds[k][i]["in_top1"]}
        if len(models_wrong) < 3:
            continue

        # PPL from fine-tuned orig tok as the "worst-case" signal
        ft_ppl = all_preds["gpt2-finetuned-orig"][i]["ppl"] or 9999

        failures.append({
            "idx":           i,
            "prefix":        item["prefix"],
            "next_word":     item["next_word"],
            "predictions": {
                k: {
                    "top1":     all_preds[k][i]["top1"],
                    "top5":     all_preds[k][i]["top5_decoded"],
                    "in_top1":  all_preds[k][i]["in_top1"],
                    "in_top3":  all_preds[k][i]["in_top3"],
                    "in_top5":  all_preds[k][i]["in_top5"],
                    "ppl":      all_preds[k][i]["ppl"],
                }
                for k in MODEL_KEYS
            },
            "n_models_wrong": len(models_wrong),
            "ft_ppl":         ft_ppl,
            "error_categories": [],   # filled by hand / downstream
        })

    # Sort: most models wrong first, then highest PPL
    failures.sort(key=lambda x: (-x["n_models_wrong"], -x["ft_ppl"]))
    failures = failures[:30]

    with open(FAIL_OUT, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(failures)} failures → {FAIL_OUT}")


if __name__ == "__main__":
    main()
