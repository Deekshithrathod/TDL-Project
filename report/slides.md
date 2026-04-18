---
marp: true
theme: default
paginate: true
---

# Language Modeling for Romanized Telugu WhatsApp Text

**Deekshith Rathod**
TDL Project · April 2026

---

## Agenda

1. What is Romanized Telugu?
2. Problem Definition
3. Dataset
4. Proposed Solution — 4-model comparison
5. Tokenizer Design
6. Fine-Tuning with LoRA
7. Results
8. Error Analysis
9. Key Takeaways & Future Work

---

## What is Romanized Telugu?

- Telugu (83M speakers) is a Dravidian language written in its own script
- In WhatsApp conversations, people write Telugu **phonetically in Latin script**
- The result freely mixes Telugu, English, and Hindi at the word level

**Example messages from the corpus:**

> *"nenu ikkadiki vastunnanu bro, okka nimisham agu"*
> ("I'm coming there bro, wait a moment")

> *"have some shame anna"*
> ("have some shame, brother" — English reprimand + Telugu address term)

- This register is used by **tens of millions** but is almost entirely absent from NLP corpora
- Standard autocomplete and language models perform poorly on it

---

## Problem Definition

**Task:** Given a prefix from a Romanized Telugu WhatsApp message, predict the most likely next word.

**Why is this hard?**

| Challenge | Example |
|-----------|---------|
| Tokenization mismatch | `cheppukovadam` → 6 GPT-2 tokens, meaningless consonant stubs |
| Code-mixing | Single sentence switches between 3 languages |
| Data scarcity | No existing benchmark; must build corpus from scratch |
| Metric distortion | Top-1 hit on first consonant ≠ correct word prediction |

**Research questions:**
1. Does fine-tuning GPT-2 on this corpus improve next-word prediction?
2. Does a domain-trained tokenizer further improve performance?

---

## Dataset — 298 WhatsApp Archives

<!-- [FIGURE: Data pipeline flowchart — Raw Chats → Clean → Split → Tokenize → Fine-tune → Evaluate] -->

**Cleaning pipeline:**
- Rejoined multi-line messages using timestamp detection
- Removed URL-only, emoji-only, short (<3 word) messages
- Anonymized 14,853 senders → `USER_A`, `USER_B`, ...

| Metric | Value |
|--------|-------|
| Raw messages | 325,478 |
| After filtering | **296,903** (91.2% kept) |
| Train / Val / Test | 237,522 / 29,690 / 29,691 |
| Corpus size | 373 MB |

Split: 80/10/10 with `seed=42`

---

## Proposed Solution — 4-Model Comparison

We isolate contributions of **multilingual pretraining** vs. **fine-tuning** vs. **tokenizer design**:

| Model | Tokenizer | Training | Tests |
|-------|-----------|----------|-------|
| GPT-2 pretrained | GPT-2 50k BPE | None | English baseline |
| mGPT pretrained | mGPT 60-lang | None | Multilingual baseline |
| GPT-2 fine-tuned (orig-tok) | GPT-2 50k BPE | LoRA on our corpus | Does fine-tuning help? |
| GPT-2 fine-tuned (custom-tok) | Our 12k BPE | LoRA on our corpus | Does custom tokenizer help? |

Each model evaluated on the **same 100 held-out next-word prediction examples**

---

## Tokenizer — The Fertility Problem

<!-- [FIGURE: Colored token block diagram — "cheppukovadam" split into 6 GPT-2 tokens vs. 2 custom tokens] -->

GPT-2 was built for English. Telugu words get shredded into useless sub-pieces:

| Word | GPT-2 tokens | Custom BPE tokens |
|------|-------------|-------------------|
| `cheppukovadam` | 6 | 2 |
| `meeru ela unnaru` | 6 | 3 |
| `vivarinchandi` | 6 | 1–2 |

**Fertility comparison** (tokens per word, 1,000-sentence sample):

| Tokenizer | Avg Tokens/Word |
|-----------|----------------|
| GPT-2 original | 2.99 |
| mGPT | 2.72 |
| **Custom BPE (ours)** | **1.63** — 45% lower |

Custom 12k BPE trained on full 296,903-sentence corpus using Byte-Level BPE, min frequency 2

---

## Fine-Tuning — LoRA on GPT-2

**Why LoRA?** Fine-tuning all 124M parameters is expensive. LoRA freezes the model and injects small trainable rank decomposition matrices into attention layers.

- Rank `r=8`, targeting `c_attn` attention matrices
- **294,912 trainable params — only 0.24% of GPT-2**
- Trained on 50k sentences, 4 epochs, Colab T4 GPU, ~20 minutes

<!-- [FIGURE: Training curve — Epoch vs. Val PPL; orig-tok descends from ~113 to 86; custom-tok stays flat ~3,200] -->

**Orig-tok training curve:**

| Epoch | Val Loss | Val PPL |
|-------|----------|---------|
| 1 | 4.72 | 112.6 |
| 2 | 4.56 | 95.8 |
| 4 | **4.46** | **86.2** |

---

## The Custom Tokenizer Problem

We resized GPT-2's embedding layer from 50,257 → 12,000 tokens for the custom vocabulary.
New embedding rows were **randomly initialized**.

**What happened:**

| Epoch | Val PPL |
|-------|---------|
| 1 | 3,314 |
| 4 | 3,192 — barely moved |

The model predicted `mariyu` ("and/also") for **55 of 100 test examples** — mode collapse to the corpus unigram.

**Root cause:** LoRA updates only attention weights. Randomly initialized embeddings never learned coherent representations — the attention weights had nothing meaningful to operate on.

**Fix (future work):** Initialize custom embeddings from subword-averaged GPT-2 embeddings *before* LoRA fine-tuning.

---

## Results — The Full Picture

<!-- [FIGURE: Grouped bar chart — Top-1/3/5 accuracy for all 4 models; mGPT bars clearly dominant] -->

| Model | Test PPL | Top-1 | Top-3 | Top-5 |
|-------|----------|-------|-------|-------|
| GPT-2 pretrained | 362.9 | 2% | 5% | 6% |
| **mGPT pretrained** | 213.3 | **15%** | **22%** | **25%** |
| GPT-2 fine-tuned (orig-tok) | **92.3** | 4% | 8% | 13% |
| GPT-2 fine-tuned (custom-tok) | 3,624 † | 1% | 1% | 1% |

*† Not directly comparable — different vocabulary size*

**The surprising result:** mGPT with zero fine-tuning beats everything we trained, by **7.5× on top-1 accuracy**

---

## What Does Fine-Tuning Actually Do?

Fine-tuning **dramatically reduces perplexity** (362.9 → 92.3, 4× improvement) but barely moves accuracy (2% → 4%).

**Why the gap?**

The fine-tuned model's top-1 predictions are dominated by Telugu consonants:

| Prediction | Count (of 100) |
|-----------|----------------|
| `v` | 16 |
| `k` | 11 |
| `mar` | 4 |

The model learned the *subword distribution* of Romanized Telugu — but the top-k metric counts predicting `v` as a hit only when the gold word happens to start with `v`. It's measuring **stub prediction, not word prediction**.

---

## The Code-Switching Wall

All models fail when the prompt is English but the correct answer is Telugu:

**Prefix:** `"have some shame"`
**Correct next word:** `anna` (Telugu: "brother", social address term)

| Model | Prediction | Correct? |
|-------|-----------|---------|
| GPT-2 pretrained | `in` | ✗ |
| mGPT pretrained | `on` | ✗ |
| GPT-2 fine-tuned | `about` | ✗ |
| GPT-2 custom-tok | `mariyu` | ✗ |

Every model sees "have some shame" as English and continues in English.
The Telugu social address marker is **invisible to any model** without explicit code-switch training data.

This pattern (English reprimand + Telugu address) is structurally common in WhatsApp chats — and structurally unlearnable from our setup.

---

## Error Analysis

<!-- [FIGURE: Horizontal bar chart of failure categories below] -->

From 30 examples where ≥3 of 4 models fail top-1:

| Failure Category | Count |
|-----------------|-------|
| Data sparsity — gold word too rare | 26 |
| Tokenization failure — subword fragmentation | 19 |
| Wrong but plausible — statistical vs. marked choice | 6 |
| English intrusion — fine-tuning hurt English | 5 |
| Extreme code-mixing | 3 |
| Hindi intrusion | 3 |

**Most failures are structural, not fixable by more training on the same setup.**
Data sparsity and tokenization together account for 45 of 58 recorded failure instances.

---

## Key Takeaways

1. **Multilingual pretraining > domain fine-tuning** for code-mixed South Asian text
   - mGPT (no fine-tuning): 15% top-1 vs. GPT-2 fine-tuned: 4% top-1

2. **Fine-tuning helps perplexity but not accuracy** — because GPT-2's tokenizer is structurally wrong for this language

3. **LoRA alone cannot fix a vocabulary transplant** — embedding retraining is required for custom tokenizer fine-tuning

4. **Top-k accuracy is a misleading metric** for this task — whole-word match is needed

5. **Code-switching is an unsolved structural problem** — no model here can cross a language boundary without bilingual pretraining

---

## Future Work

| Priority | Action | Expected Impact |
|---------|--------|----------------|
| 1 | **Fine-tune mGPT** (not GPT-2) on our 296k corpus | High — right initialization |
| 2 | **Fix embedding init** — subword-average GPT-2 embeddings into custom vocab slots | High — unblocks custom tokenizer |
| 3 | **Change metric** — whole-word top-k instead of first-subtoken | Medium — fairer evaluation |
| 4 | **Stratified eval set** — separate English-only, Telugu-only, code-mixed prefixes | Medium — clearer diagnostics |
| 5 | **Larger corpus** — more WhatsApp archives to reduce data sparsity failures | High but slow |

---

## Thank You

**Repository:** `github.com/Deekshithrathod/TDL-Project`

**Summary:**
- 296,903 WhatsApp sentences collected and cleaned
- Custom 12k BPE tokenizer: 45% lower fertility than GPT-2
- LoRA fine-tuning: 4× perplexity reduction in 20 minutes on a free GPU
- Main finding: **Use mGPT, not GPT-2, as the starting point for code-mixed South Asian NLP**

---

**Questions?**
