# Language Modeling for Romanized Telugu WhatsApp Text

**Deekshith Rathod** · TDL Project · April 2026

---

## 1. Introduction

Telugu is a Dravidian language with approximately 83 million speakers. In everyday digital communication — particularly on WhatsApp — speakers routinely write Telugu phonetically in the Latin script, a practice called Romanization. The resulting text freely interleaves Telugu, English, and Hindi at the word and phrase level, producing a code-mixed register that standard NLP pipelines handle poorly.

Code-mixed Romanized Telugu is largely absent from standard pretraining corpora. Existing language models, trained predominantly on formal English or high-resource languages, have little exposure to informal Romanized South Asian text. This creates a practical gap: users of this register receive poor autocomplete and next-word prediction on their devices.

This project investigates whether fine-tuning a pretrained language model on a domain-specific Romanized Telugu corpus can meaningfully improve next-word prediction, and whether a custom tokenizer trained on the same corpus further improves performance.

---

## 2. Problem Definition

**Task:** Given a prefix sequence of words from a Romanized Telugu WhatsApp message, predict the most likely next word.

**Challenges:**
- **Tokenization mismatch:** English BPE tokenizers fragment Telugu words into meaningless consonant stubs (e.g., `cheppukovadam` → 6 GPT-2 tokens), distorting model predictions.
- **Code-mixing:** A single sentence can switch between Telugu, English, and Hindi at word boundaries (e.g., *"have some shame anna"*). No standard model is trained for this pattern.
- **Data scarcity:** Romanized Telugu has no large curated NLP benchmark; all data must be sourced and cleaned from scratch.
- **Evaluation difficulty:** Standard top-k accuracy is confounded by subword fragmentation — predicting the first consonant of a Telugu word counts as a hit even when the model has not learned the word.

**Formal definition:** Let $x = w_1, w_2, \ldots, w_n$ be a sentence from the corpus. The model receives $w_1, \ldots, w_{n-1}$ and must rank $w_n$ as high as possible in its predicted distribution. We evaluate top-1, top-3, and top-5 accuracy on 100 held-out examples, alongside corpus-level perplexity.

---

## 3. Proposed Solution

We compare four model configurations to isolate the contribution of each design choice:

| Model | Tokenizer | Training |
|-------|-----------|----------|
| GPT-2 pretrained | GPT-2 original (50k BPE) | None (English baseline) |
| mGPT pretrained | mGPT multilingual (60-lang) | None (multilingual baseline) |
| GPT-2 fine-tuned (orig-tok) | GPT-2 original (50k BPE) | LoRA on 50k WhatsApp sentences |
| GPT-2 fine-tuned (custom-tok) | Custom 12k BPE (our corpus) | LoRA on 50k WhatsApp sentences |

**Data pipeline:** 298 WhatsApp chat archives are cleaned (URL/emoji/short-message removal, sender anonymization, lowercasing) to produce a 296,903-sentence corpus, split 80/10/10 for train/validation/test.

**Custom tokenizer:** A Byte-Level BPE tokenizer (vocabulary size 12,000, min frequency 2) is trained on the full corpus. This reduces token fertility from 2.99 tokens/word (GPT-2) to 1.63 — a 45% improvement — allowing Telugu words to be represented as single or double tokens.

**Fine-tuning method:** LoRA (Low-Rank Adaptation) [2] with rank $r=8$, targeting GPT-2's attention `c_attn` matrices. This yields 294,912 trainable parameters (0.24% of GPT-2's 124M), enabling efficient training on a single Colab T4 GPU in ~20 minutes.

<!-- FIGURE: Data pipeline flowchart — boxes: Raw Chats → Clean → Split → Tokenize → LoRA Fine-tune → Evaluate -->

---

## 4. Experiments

### 4.1 Dataset

| Metric | Value |
|--------|-------|
| Raw messages | 325,478 |
| After filtering | 296,903 (91.2% retained) |
| Train / Val / Test | 237,522 / 29,690 / 29,691 |
| Unique senders (anonymized) | 14,853 |
| Corpus size | 373 MB |

Filtering removed messages under 3 words, emoji-only (2,948), URL-only (449), and system messages.

### 4.2 Tokenizer Fertility

<!-- FIGURE: Tokenization diagram — side-by-side colored token blocks showing "cheppukovadam" split into 6 GPT-2 tokens vs. 2 custom tokens -->

| Tokenizer | Avg Tokens/Word | Reduction vs. GPT-2 |
|-----------|----------------|---------------------|
| GPT-2 original | 2.99 | — |
| mGPT | 2.72 | 9% |
| Custom BPE (ours) | **1.63** | **45%** |

Native Telugu words such as `unnaru`, `ledu`, `meeru`, and `nenu` each map to a single token in the custom vocabulary, compared to 2–4 tokens under GPT-2.

### 4.3 LoRA Fine-Tuning

**Orig-tok model** trained for 4 epochs (50k samples, lr=2e-4, cosine schedule, batch size 64):

| Epoch | Val Loss | Val PPL |
|-------|----------|---------|
| 1 | 4.72 | 112.6 |
| 2 | 4.56 | 95.8 |
| 3 | 4.49 | 88.9 |
| 4 | **4.46** | **86.2** |

**Custom-tok model** — GPT-2's embedding layer was resized from 50,257 to 12,000 tokens; new rows were randomly initialized. Val PPL stagnated near 3,192 across all 4 epochs, indicating that LoRA (which updates only attention weights) cannot learn coherent representations over randomly initialized embeddings.

<!-- FIGURE: Training curves — dual line chart, Epoch vs. Val PPL; orig-tok descends smoothly from ~113 to 86; custom-tok stays flat near 3,200 -->

---

## 5. Results & Observations

### 5.1 Main Results

<!-- FIGURE: Bar chart — grouped bars for top-1/3/5 accuracy across all 4 models; mGPT bars visually dominant -->

| Model | Test PPL | Top-1 | Top-3 | Top-5 |
|-------|----------|-------|-------|-------|
| GPT-2 pretrained | 362.9 | 2% | 5% | 6% |
| mGPT pretrained | 213.3 | **15%** | **22%** | **25%** |
| GPT-2 fine-tuned (orig-tok) | **92.3** | 4% | 8% | 13% |
| GPT-2 fine-tuned (custom-tok) | 3,624.8 † | 1% | 1% | 1% |

*† Custom-tok PPL uses a 12k-vocab tokenizer and is not directly comparable to the other three.*

**Observation 1 — mGPT dominates top-k accuracy (15% vs. 4%).**  
mGPT's 60-language pretraining covers Hindi, informal transliterated registers, and code-mixed South Asian text, giving it structural knowledge that a few thousand LoRA gradient steps on monolingual GPT-2 cannot replicate.

**Observation 2 — Fine-tuning substantially reduces perplexity (362.9 → 92.3) but only marginally improves accuracy (2% → 4%).**  
This gap is explained by a tokenization artifact: the fine-tuned GPT-2 has learned to predict Telugu-initial consonants (`v`, `k`, `n`) as top-1 outputs. Of 4 top-1 successes, all 4 are cases where the gold word happens to begin with one of those consonants. The model is predicting subword stubs, not words.

**Observation 3 — Custom tokenizer fine-tuning failed (mode collapse).**  
The model predicted `mariyu` ("and/also") for 55 of 100 examples — converging to the corpus unigram mode. Root cause: LoRA updates only attention matrices; randomly initialized embedding slots cannot form coherent token representations without full embedding retraining.

**Observation 4 — Code-switching is a universal failure.**  
On examples like `"have some shame" → anna` (English reprimand + Telugu address term), all four models predict English continuations. No model trained on either English or Telugu data has seen this bilingual pattern.

### 5.2 Error Analysis

<!-- FIGURE: Horizontal bar chart of error categories from the table below -->

| Failure Category | Count (of 30 failures) |
|-----------------|------------------------|
| Data sparsity (label word low-frequency) | 26 |
| Tokenization failure (subword fragmentation) | 19 |
| Wrong but plausible (statistical vs. marked choice) | 6 |
| English intrusion (fine-tuning overwrites English) | 5 |
| Extreme code-mixing | 3 |
| Hindi intrusion | 3 |

The largest failure class — data sparsity — is irreducible without a larger corpus. Tokenization failures are structurally fixable by changing the evaluation metric to whole-word match rather than first-subtoken match.

### 5.3 Key Takeaway

Multilingual pretraining (mGPT) is more valuable for code-mixed South Asian text than domain-specific fine-tuning of a monolingual model. The correct next step is fine-tuning mGPT itself on the Romanized Telugu corpus, not fine-tuning GPT-2. Additionally, custom tokenizer fine-tuning requires full embedding retraining — not LoRA alone — to avoid mode collapse.

---

## References

[1] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*, 1(8), 9.

[2] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

[3] Shliazhko, O., Fenogenova, A., Tikhonova, M., Mikhailov, V., Kozlova, A., & Shevelev, T. (2022). mGPT: Few-shot learners go multilingual. *arXiv:2204.07580*.

[4] Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *ACL 2016*.

[5] Bhat, I. A., Bhat, R. A., Bali, K., & Choudhury, M. (2018). Universal dependency parsing for Hindi-English code-switching. *NAACL 2018*.

[6] Khanuja, S., Dandapat, S., Srinivasan, A., Sitaram, S., & Choudhury, M. (2020). GLUECoS: An evaluation benchmark for code-switched NLP. *ACL 2020*.
