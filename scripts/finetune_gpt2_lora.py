"""
finetune_gpt2_lora.py
=====================
Week 3 — LoRA fine-tune GPT-2 on the cleaned Romanized Telugu WhatsApp corpus.

Wraps GPT-2 with LoRA adapters (r=8, target c_attn), trains with HuggingFace
Trainer on MPS / CUDA / CPU, then runs a before/after comparison on the 15
Week-1 test sentences.

# MPS TROUBLESHOOTING:
# If you get "RuntimeError: MPS backend out of memory":
#   - Drop --batch_size to 4
#   - Add gradient_checkpointing=True to TrainingArguments
# If you get "op not implemented for MPS" errors on a specific op:
#   - Set device = torch.device("cpu") as a temporary workaround
#   - This is rare for GPT-2 + LoRA but can happen on older macOS/PyTorch versions
# Recommended: PyTorch >= 2.0, macOS >= 13.0 for stable MPS support

Usage
-----
    python scripts/finetune_gpt2_lora.py \\
        --dataset data/clm_dataset_gpt2 \\
        --output  models/gpt2_lora_finetuned \\
        --epochs  5 \\
        --rank    8
"""

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Device detection — MPS first, then CUDA, then CPU
# ---------------------------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("WARNING: No GPU detected. Training on CPU will be slow (~2-3x per epoch).")
    print("Consider: reduce --epochs to 3, --batch_size to 4.")

# ---------------------------------------------------------------------------
# Imports (after device is resolved)
# ---------------------------------------------------------------------------

from datasets import load_from_disk  # noqa: E402
from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Week-1 test sentences — identical to baseline_eval.py, do not change
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
try:
    from baseline_eval import TELUGU_SENTENCES
except ImportError:
    TELUGU_SENTENCES = [
        "nenu ikkadiki vastunnanu", "meeru ela unnaru", "nuvvu ela unnav",
        "okka nimisham agu", "naku Telugu vastundi", "idi chala kashtanga undi",
        "mee peru enti", "nenu intiki veltunna", "manaku chala pani undi",
        "ikkade kurchoni matladamu", "mee number ivvagalara", "adi evaru chepparu",
        "reyyi chala baagundi", "nenu maatladatam nerchukuntunna",
        "mee inti address cheppandi",
    ]

# ---------------------------------------------------------------------------
# Custom collator
#
# transformers 4.50+ DataCollatorForLanguageModeling fails when the dataset
# already has a 'labels' column with variable-length sequences — it passes
# the labels field to tokenizer.pad() which doesn't know how to handle it.
# Fix: strip labels, pad input_ids/attention_mask, then recreate labels with
# padding positions masked to -100 so they're excluded from the loss.
# ---------------------------------------------------------------------------

@dataclass
class CLMCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # Drop pre-stored labels — we recreate from padded input_ids
        stripped = [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
            for f in features
        ]
        batch = self.tokenizer.pad(
            stripped,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        # Mask padding tokens so they don't contribute to the loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Perplexity callback
# ---------------------------------------------------------------------------

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"])
            print(f"  Epoch {state.epoch:.1f} — "
                  f"Val Loss: {metrics['eval_loss']:.4f} | Val PPL: {ppl:.2f}")
            metrics["eval_perplexity"] = ppl


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, sentence: str) -> float:
    ids = tokenizer.encode(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return round(math.exp(loss.item()), 2)


def get_top5(model, tokenizer, sentence: str) -> list[str]:
    ids = tokenizer.encode(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :]
    top5_ids = torch.topk(logits, k=5).indices
    return [tokenizer.decode(i.item()).strip() for i in top5_ids]


# ---------------------------------------------------------------------------
# Perplexity curve from trainer_state.json
# ---------------------------------------------------------------------------

def extract_perplexity_curve(checkpoint_dir: str) -> pd.DataFrame:
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.exists():
        print(f"  trainer_state.json not found at {state_path}")
        return pd.DataFrame()

    with open(state_path) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    # Latest train loss per epoch
    train_by_epoch: dict[float, float] = {}
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            ep = round(entry.get("epoch", 0), 1)
            train_by_epoch[ep] = entry["loss"]

    rows = []
    for entry in log_history:
        if "eval_loss" in entry:
            ep  = round(entry.get("epoch", 0), 1)
            evl = entry["eval_loss"]
            ppl = round(math.exp(evl), 2)
            trn = next(
                (v for k, v in sorted(train_by_epoch.items(), reverse=True) if k <= ep),
                None,
            )
            rows.append({
                "epoch":          ep,
                "train_loss":     round(trn, 4) if trn is not None else None,
                "val_loss":       round(evl, 4),
                "val_perplexity": ppl,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune GPT-2 on cleaned Romanized Telugu corpus."
    )
    parser.add_argument("--dataset",    default="data/clm_dataset_gpt2")
    parser.add_argument("--output",     default="models/gpt2_lora_finetuned")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--rank",       type=int,   default=8)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--report_dir", default="report")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        alt = Path("data") / args.dataset
        if alt.exists():
            dataset_path = alt
        else:
            sys.exit(f"Dataset not found at {dataset_path} or {alt}")

    output_dir = Path(args.output)
    report_dir = Path(args.report_dir)
    ckpt_dir   = output_dir.parent / (output_dir.name + "_checkpoints")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer + dataset
    # ------------------------------------------------------------------
    print("\nLoading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(str(dataset_path))
    print(f"  Train: {len(dataset['train']):,}  "
          f"Val: {len(dataset['validation']):,}  "
          f"Test: {len(dataset['test']):,}")

    # ------------------------------------------------------------------
    # Base model → LoRA
    # ------------------------------------------------------------------
    print("\nLoading GPT-2 and wrapping with LoRA...")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    base_model.config.pad_token_id = tokenizer.eos_token_id
    base_model = base_model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # TrainingArguments — let Trainer auto-detect MPS/CUDA/CPU
    #
    # In transformers ≥4.50, Trainer auto-detects MPS when available.
    # Passing use_mps_device=True or no_cuda=True causes conflicts on
    # macOS 26+, so we leave device selection to the framework default.
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        fp16=False,   # MPS does not support fp16; CUDA users can set True
        bf16=False,
        seed=42,
        report_to="none",
    )

    collator = CLMCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        callbacks=[PerplexityCallback()],
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nStarting training — {args.epochs} epochs | lr={args.lr} | "
          f"rank={args.rank} | batch={args.batch_size} | device={device}")
    print("=" * 60)
    trainer.train()

    # ------------------------------------------------------------------
    # Save best model
    # ------------------------------------------------------------------
    print(f"\nSaving final (best) model → {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("  Done.")

    # ------------------------------------------------------------------
    # Perplexity curve
    # ------------------------------------------------------------------
    print("\nExtracting perplexity curve...")
    curve_df = extract_perplexity_curve(str(ckpt_dir))

    if not curve_df.empty:
        print("\n" + curve_df.to_string(index=False))
        curve_csv = report_dir / "perplexity_curve.csv"
        curve_df.to_csv(curve_csv, index=False)
        print(f"\nSaved: {curve_csv}")
    else:
        print("  No eval entries found in trainer_state.json.")

    # ------------------------------------------------------------------
    # Post-training comparison — Week 1 sentences
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION — Week 1 test sentences")
    print("=" * 60)

    print("Loading pretrained GPT-2 (no LoRA)...")
    pretrained = AutoModelForCausalLM.from_pretrained("gpt2")
    pretrained.config.pad_token_id = tokenizer.eos_token_id
    pretrained.eval()
    pretrained.to(device)

    print(f"Loading finetuned LoRA model from {output_dir}...")
    ft_base = AutoModelForCausalLM.from_pretrained("gpt2")
    ft_base.config.pad_token_id = tokenizer.eos_token_id
    finetuned = PeftModel.from_pretrained(ft_base, str(output_dir))
    finetuned.eval()
    finetuned.to(device)

    records = []
    for sent in TELUGU_SENTENCES:
        pre_ppl  = compute_perplexity(pretrained, tokenizer, sent)
        ft_ppl   = compute_perplexity(finetuned,  tokenizer, sent)
        pre_top5 = get_top5(pretrained, tokenizer, sent)
        ft_top5  = get_top5(finetuned,  tokenizer, sent)
        pct      = 100 * (pre_ppl - ft_ppl) / pre_ppl if pre_ppl else 0
        arrow    = "↓" if ft_ppl < pre_ppl else "↑"

        records.append({
            "sentence":        sent,
            "pretrained_ppl":  pre_ppl,
            "finetuned_ppl":   ft_ppl,
            "ppl_delta":       round(pre_ppl - ft_ppl, 2),
            "pretrained_top5": ", ".join(pre_top5),
            "finetuned_top5":  ", ".join(ft_top5),
        })
        print(f"\n  [{sent}]")
        print(f"    pretrained PPL: {pre_ppl:>10.1f}  top5: {pre_top5}")
        print(f"    finetuned  PPL: {ft_ppl:>10.1f}  top5: {ft_top5}")
        print(f"    change: {arrow} {abs(pct):.1f}%")

    comp_df  = pd.DataFrame(records)
    comp_csv = report_dir / "finetune_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nComparison saved: {comp_csv}")

    # Summary
    improved = (comp_df["finetuned_ppl"] < comp_df["pretrained_ppl"]).sum()
    avg_pre  = comp_df["pretrained_ppl"].mean()
    avg_ft   = comp_df["finetuned_ppl"].mean()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sentences improved (lower PPL): {improved}/{len(comp_df)}")
    print(f"Avg pretrained PPL (Telugu):    {avg_pre:.1f}")
    print(f"Avg finetuned  PPL (Telugu):    {avg_ft:.1f}")
    if avg_ft > 500:
        print("\n  NOTE: Val PPL still >500 after fine-tuning. This is expected "
              "given the domain gap and tokenizer mismatch. The improvement "
              "trend matters more than the absolute number for this experiment.")


if __name__ == "__main__":
    main()
