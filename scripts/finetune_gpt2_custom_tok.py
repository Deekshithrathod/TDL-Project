"""
finetune_gpt2_custom_tok.py
===========================
Week 3 — LoRA fine-tune GPT-2 with the custom BPE tokenizer on the
cleaned Romanized Telugu WhatsApp corpus.

Key difference from finetune_gpt2_lora.py:
  - Uses a custom 12k ByteLevelBPE tokenizer (fertility 1.63 vs GPT-2's 2.99)
  - GPT-2's embedding layer is resized from 50,257 → 12,000 tokens
  - New embedding rows are randomly initialised — expect slower convergence
  - Raw perplexity numbers are NOT comparable with the original-tokenizer run
    (perplexity is tokenizer-dependent; fewer tokens per sentence lowers loss)

# MPS TROUBLESHOOTING:
# If "RuntimeError: MPS backend out of memory":
#   - Drop --batch_size to 4
#   - Add gradient_checkpointing=True to TrainingArguments
# If "op not implemented for MPS":
#   - Set device = torch.device("cpu") as a temporary workaround
# Recommended: PyTorch >= 2.0, macOS >= 13.0

Usage
-----
    python scripts/finetune_gpt2_custom_tok.py \\
        --tokenizer_path tokenizers/tokenizer_codemixed \\
        --cleaned_data   data/processed/cleaned_data.txt \\
        --output         models/gpt2_lora_custom_tok \\
        --epochs         5 \\
        --rank           8
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
    print("WARNING: No GPU detected. Training on CPU will be slow.")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from datasets import Dataset, DatasetDict  # noqa: E402
from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: E402
from tokenizers import ByteLevelBPETokenizer  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Week-1 test sentences
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
# already has a 'labels' column with variable-length sequences. Fix: strip
# labels, pad input_ids/attention_mask, recreate labels with -100 on padding.
# ---------------------------------------------------------------------------

@dataclass
class CLMCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
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
        for p in Path(checkpoint_dir).glob("**/trainer_state.json"):
            state_path = p
            break
    if not state_path.exists():
        print(f"  trainer_state.json not found under {checkpoint_dir}")
        return pd.DataFrame()

    with open(state_path) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
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
        description="LoRA fine-tune GPT-2 with custom BPE tokenizer."
    )
    parser.add_argument(
        "--tokenizer_path", required=True,
        help="Path to custom tokenizer dir (e.g. tokenizers/tokenizer_codemixed)",
    )
    parser.add_argument("--cleaned_data",      default="data/processed/cleaned_data.txt")
    parser.add_argument("--output",            default="models/gpt2_lora_custom_tok")
    parser.add_argument("--prev_model_dir",    default="models/gpt2_lora_finetuned",
                        help="Path to previous LoRA model (GPT-2 tokenizer) for 3-way comparison")
    parser.add_argument("--epochs",            type=int,   default=5)
    parser.add_argument("--rank",              type=int,   default=8)
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--lr",                type=float, default=2e-4)
    parser.add_argument("--report_dir",        default="report")
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Cap training set size (e.g. 50000). Full 237k set takes hours on T4.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    tokenizer_path    = Path(args.tokenizer_path)
    cleaned_data_path = Path(args.cleaned_data)

    if not tokenizer_path.exists():
        sys.exit(f"Tokenizer path not found: {tokenizer_path}")
    if not cleaned_data_path.exists():
        sys.exit(f"Cleaned data not found: {cleaned_data_path}")

    output_dir = Path(args.output)
    report_dir = Path(args.report_dir)
    ckpt_dir   = output_dir.parent / (output_dir.name + "_checkpoints")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load custom tokenizer
    # ------------------------------------------------------------------
    print(f"\nLoading custom tokenizer from {tokenizer_path}...")
    raw_tok = ByteLevelBPETokenizer.from_file(
        str(tokenizer_path / "vocab.json"),
        str(tokenizer_path / "merges.txt"),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    # For CLM, pad with eos so padding positions are masked in the loss
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {len(tokenizer):,}")

    # ------------------------------------------------------------------
    # Re-tokenize cleaned_data.txt
    #
    # Workaround: tokenizers 0.22.x doesn't support enable_truncation(direction=...)
    # which transformers passes internally when truncation=True is set on a
    # PreTrainedTokenizerFast wrapping a custom BPE. We call without truncation
    # flags and slice manually instead.
    # ------------------------------------------------------------------
    print(f"\nLoading corpus: {cleaned_data_path}")
    with open(cleaned_data_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    print(f"  {len(lines):,} sentences loaded.")

    raw_ds  = Dataset.from_dict({"text": lines})
    MAX_LEN = 128

    def tokenize(batch):
        out   = tokenizer(batch["text"], truncation=False, padding=False)
        ids   = [seq[:MAX_LEN] for seq in out["input_ids"]]
        masks = [seq[:MAX_LEN] for seq in out["attention_mask"]]
        return {"input_ids": ids, "attention_mask": masks, "labels": [i.copy() for i in ids]}

    print(f"Tokenizing (max_length={MAX_LEN}, manual truncation)...")
    tokenized = raw_ds.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenising")

    split1  = tokenized.train_test_split(test_size=0.2, seed=42)
    split2  = split1["test"].train_test_split(test_size=0.5, seed=42)
    dataset = DatasetDict({
        "train":      split1["train"],
        "validation": split2["train"],
        "test":       split2["test"],
    })
    print(f"  Split sizes: { {k: len(v) for k, v in dataset.items()} }")

    # Optionally cap training set
    train_ds = dataset["train"]
    if args.max_train_samples and args.max_train_samples < len(train_ds):
        train_ds = train_ds.select(range(args.max_train_samples))
        print(f"  Train (capped): {len(train_ds):,}  "
              f"Val: {len(dataset['validation']):,}  "
              f"Test: {len(dataset['test']):,}")
    else:
        print(f"  Train: {len(train_ds):,}  "
              f"Val: {len(dataset['validation']):,}  "
              f"Test: {len(dataset['test']):,}")

    # ------------------------------------------------------------------
    # Load GPT-2 and resize embeddings
    # New embedding rows are randomly initialised — expected and intentional.
    # The model must learn token associations from scratch for the new vocab.
    # ------------------------------------------------------------------
    print("\nLoading GPT-2 and resizing embedding layer...")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    old_vocab  = base_model.config.vocab_size
    base_model.resize_token_embeddings(len(tokenizer))
    new_vocab  = base_model.config.vocab_size
    print(f"  Embedding layer resized: {old_vocab:,} → {new_vocab:,}")
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model = base_model.to(device)

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
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
    # TrainingArguments
    # Note: transformers ≥4.50 renamed evaluation_strategy → eval_strategy.
    # MPS is auto-detected by the Trainer — no use_mps_device/no_cuda needed.
    # fp16 is safe on CUDA (T4) and auto-disabled on MPS/CPU.
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
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=42,
        report_to="none",
    )

    collator = CLMCollator(tokenizer=tokenizer)

    steps_per_epoch = -(-len(train_ds) // args.batch_size)
    print(f"\n  Steps/epoch: {steps_per_epoch:,}  |  "
          f"Total steps: {steps_per_epoch * args.epochs:,}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
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
    # Save
    # ------------------------------------------------------------------
    print(f"\nSaving model → {output_dir}")
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
        curve_csv = report_dir / "perplexity_curve_custom_tok.csv"
        curve_df.to_csv(curve_csv, index=False)
        print(f"\nSaved: {curve_csv}")
    else:
        print("  No eval entries found in trainer_state.json.")

    # ------------------------------------------------------------------
    # Three-way comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("THREE-WAY COMPARISON — Week 1 test sentences")
    print("=" * 60)
    print("NOTE: PPL values are NOT directly comparable across tokenizers.")
    print("The custom tokenizer produces fewer tokens per sentence, which")
    print("mechanically lowers the per-token loss. Compare top-5 prediction")
    print("quality qualitatively; use PPL only for within-tokenizer trends.")
    print("=" * 60)

    # 1. Pretrained GPT-2 (original tokenizer, no fine-tuning)
    print("\nLoading pretrained GPT-2 (original tokenizer)...")
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    pretrained = AutoModelForCausalLM.from_pretrained("gpt2")
    pretrained.config.pad_token_id = gpt2_tok.eos_token_id
    pretrained.eval().to(device)

    # 2. Fine-tuned GPT-2 (original tokenizer + LoRA, from previous task)
    prev_model_dir  = Path(args.prev_model_dir)
    finetuned_orig  = None
    if prev_model_dir.exists() and any(prev_model_dir.iterdir()):
        print(f"Loading finetuned orig-tok model from {prev_model_dir}...")
        ft_base = AutoModelForCausalLM.from_pretrained("gpt2")
        ft_base.config.pad_token_id = gpt2_tok.eos_token_id
        finetuned_orig = PeftModel.from_pretrained(ft_base, str(prev_model_dir))
        finetuned_orig.eval().to(device)
    else:
        print(f"  WARNING: {prev_model_dir} not found or empty — skipping orig-tok column")

    # 3. Fine-tuned custom-tokenizer model (just trained)
    print(f"Loading custom-tok finetuned model from {output_dir}...")
    ct_base = AutoModelForCausalLM.from_pretrained("gpt2")
    ct_base.resize_token_embeddings(len(tokenizer))
    ct_base.config.pad_token_id = tokenizer.pad_token_id
    finetuned_ct = PeftModel.from_pretrained(ct_base, str(output_dir))
    finetuned_ct.eval().to(device)

    records = []
    for sent in TELUGU_SENTENCES:
        pre_ppl  = compute_perplexity(pretrained,     gpt2_tok,  sent)
        pre_top5 = get_top5(pretrained,               gpt2_tok,  sent)

        orig_ppl  = compute_perplexity(finetuned_orig, gpt2_tok, sent) if finetuned_orig else None
        orig_top5 = get_top5(finetuned_orig,           gpt2_tok, sent) if finetuned_orig else []

        ct_ppl  = compute_perplexity(finetuned_ct, tokenizer, sent)
        ct_top5 = get_top5(finetuned_ct,           tokenizer, sent)

        records.append({
            "sentence":                  sent,
            "pretrained_gpt2_ppl":       pre_ppl,
            "finetuned_orig_tok_ppl":    orig_ppl,
            "finetuned_custom_tok_ppl":  ct_ppl,
            "pretrained_gpt2_top5":      ", ".join(pre_top5),
            "finetuned_orig_tok_top5":   ", ".join(orig_top5),
            "finetuned_custom_tok_top5": ", ".join(ct_top5),
        })

        print(f"\n  [{sent}]")
        print(f"    pretrained GPT-2    PPL: {pre_ppl:>10.1f}  top5: {pre_top5}")
        if finetuned_orig:
            print(f"    finetuned orig-tok  PPL: {orig_ppl:>10.1f}  top5: {orig_top5}")
        print(f"    finetuned cust-tok  PPL: {ct_ppl:>10.1f}  top5: {ct_top5}")

    comp_df  = pd.DataFrame(records)
    comp_csv = report_dir / "custom_tok_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nComparison saved: {comp_csv}")

    avg_pre = comp_df["pretrained_gpt2_ppl"].mean()
    avg_ct  = comp_df["finetuned_custom_tok_ppl"].mean()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Avg pretrained GPT-2 PPL (original tokenizer): {avg_pre:.1f}")
    if finetuned_orig:
        avg_orig = comp_df["finetuned_orig_tok_ppl"].mean()
        print(f"Avg finetuned orig-tok PPL:                     {avg_orig:.1f}")
    print(f"Avg finetuned custom-tok PPL:                   {avg_ct:.1f}")
    print("\n  NOTE: PPL values across tokenizers are not directly comparable.")
    print("  Focus on top-5 prediction quality for cross-tokenizer comparison.")


if __name__ == "__main__":
    main()
