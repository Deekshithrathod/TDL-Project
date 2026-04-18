# Error Analysis: Romanized Telugu Language Model Evaluation

**Experiment:** Four model variants evaluated on 100 held-out Romanized Telugu WhatsApp prefixes.
**Models:** GPT-2 pretrained · mGPT pretrained · GPT-2 fine-tuned (orig tok) · GPT-2 fine-tuned (custom 12k BPE)
**Metrics:** Top-1/3/5 next-word accuracy, per-example perplexity.
**Overall top-1 scores:** mGPT 15% · GPT-2 fine-tuned orig 4% · GPT-2 pretrained 2% · Custom tok 1%.

---

## 1. Dominant Failure Modes

### 1.1 Data Sparsity (26/30 failures, ~87%)

By far the dominant category. Romanized Telugu WhatsApp text spans an enormous register range — formal news articles, movie discussions, technical explanations, Hindi/Bengali passages, colloquial slang — and the correct next word is often a low-frequency item in any training distribution. Examples include:

- `"login credentials" → "laantivi"` — a Telugu demonstrative suffix appended to an English phrase
- `"bere are" → "bradhas"` — WhatsApp slang for "brothers", essentially absent from standard corpora
- `"info edge" → "pyq"` — an acronym (Previous Year Questions) specific to Indian academic communities
- `"send ai day" → "photos"` — a short, ambiguous prompt where no prior context forces the correct content word

Data sparsity is almost never the sole failure cause; it combines with tokenization and code-mixing issues. However, it is the baseline failure mode: even with correct tokenization and architecture, the correct word simply isn't predictable from the prefix alone without domain-specific training.

### 1.2 Tokenization Failure — Subword Splitting (19/30 failures, ~63%)

GPT-2's BPE tokenizer fragments long Telugu compound verbs into many short pieces. A word like `vivarinchandi` (explain) becomes roughly `['v', 'ivar', 'inch', 'andi']`. The top-1 prediction task asks for the *first subtoken*, so predicting `'v'` is technically correct but useless for fluent generation.

This creates a systematic scoring artefact: GPT-2 fine-tuned (orig) predicts a single consonant (`v`, `k`, `n`, `b`) 60+ times in 100 examples. Its actual top-1 distribution is dominated by these single-letter predictions:

| Top-1 prediction | Count (out of 100) |
|---|---|
| `'v'` | 16 |
| `'k'` | 11 |
| `'a'` | 3 |
| `'am'` | 4 |
| `'mar'` | 4 |

When its four successes (`vivarinchandi` ×3, `bhai` ×1) are examined, all four are cases where `'v'` or `'b'` is the correct first subtoken. The model has learned the consonant-initial distribution of Romanized Telugu but cannot resolve the full word.

### 1.3 English and Hindi Intrusion (5 and 3 failures respectively)

WhatsApp corpora contain substantial English and Hindi content alongside Telugu. Several failures are structurally English prompts where the correct completion is a Telugu word, or vice versa:

- `"have some shame" → "anna"` — All models predict English continuations (`in`, `on`, `to`). The correct answer is the Telugu/Hindi address term *anna* ("brother"), a typical code-switching point in Telugu conversations.
- `"haan dekh rhe hain uspr" → "bhi"` — This prefix is entirely Hindi ("Come on, (we're) watching it on that side too"). GPT-2 fine-tuned interestingly predicts `"ab"` (another Hindi word: "now") — evidence that Hindi leaked into the training corpus — but still fails on the correct word.
- `"ye simple function hai toh woh ai aur iai waale form mai likh" → "de"` — Hindi imperative construction (*likh de* = "write it down"). Fine-tuned GPT-2 predicts `"inch"`, confusing the trailing `-inch` with the Telugu verb suffix *-inchandi*, a plausible but wrong cross-language transfer.

### 1.4 Wrong but Plausible (6/30 failures)

In roughly one in five failures the model's prediction is semantically reasonable in the context — it is simply not the word the author chose:

- `"yea they had done very" → "poor"` — every model predicts `"well"` or `"good"`. "Poorly" or "well" are overwhelmingly more frequent after "done very" in English text. The correct answer is the marked choice.
- `"6 credit course why not to sit for" → "placements"` — all models predict determiners (`a`, `the`, `an`). "Placements" is a campus-specific noun that completes an Indian student register.
- `"i asked mahvith took the" → "keys"` — mGPT predicts `"time"`, GPT-2 predicts `"opportunity"`. Both are common English nouns that follow "took the ___" at training frequency. "Keys" is too specific.

### 1.5 Custom Tokenizer Mode Collapse

The most striking single failure mode: the custom tok model predicts `"mariyu"` (Telugu: "and/also") for 55 out of 100 examples. Another 29 predict `"e"` (a common Telugu vowel/particle). Only 1 example in the entire eval set receives a correct top-1 prediction.

This is not a random failure — it is a degenerate solution. The model converged to the highest-frequency word in the training corpus regardless of prefix. The root cause is the embedding mismatch described in the training notes: GPT-2's original 50k BPE embeddings were truncated to 12k slots and the custom vocabulary's token IDs were assigned to these mismatched slots. Four epochs of LoRA training (which targets only the attention `c_attn` matrices, not the embeddings directly) was insufficient to learn coherent representations for the new vocabulary, leaving the model to rely on the pre-trained attention weights operating over essentially random embeddings.

---

## 2. Does Fine-Tuning Help?

**Yes, for same-tokenizer perplexity.** GPT-2 fine-tuned (orig tok) achieves PPL 92 vs GPT-2 pretrained's PPL 363 — a 4× reduction. The model clearly learned the distribution of Romanized Telugu tokens.

**Less clearly for top-k accuracy.** The fine-tuned model scores only 4% top-1 vs 2% for pretrained GPT-2. The improvement is real but modest. Fine-tuning reshapes the model to predict *Telugu-initial consonants* fluently, which counts as top-1 hits when the label's first subtoken happens to be that consonant — but this conflates tokenization success with genuine word prediction.

**For specific failure types, fine-tuning actively hurts:**
- On pure English or Hindi prompts, the fine-tuned model is *worse* than pretrained GPT-2: it has partially overwritten English grammatical knowledge with Telugu patterns, so it predicts Telugu subword fragments even in clearly English-language contexts.
- On the abbreviation `"hrs"` example, fine-tuned GPT-2 predicts `"days"` where pretrained GPT-2 predicts `"mins"` — both wrong, but fine-tuning replaced a semantically-close English unit with a weaker one.

**Fine-tuning does help on Telugu-dominant formal text.** The three `vivarinchandi` successes (examples 24, 60, 61) all have long, formal Telugu instruction prefixes. Fine-tuned GPT-2 predicts `'v'` (correct first subtoken) while pretrained GPT-2 predicts `'k'` (a learned-but-wrong Telugu consonant). The fine-tuned model has clearly absorbed the Telugu formal register.

---

## 3. What the Custom Tokenizer Buys — and Where It Still Falls Short

### Theoretical advantage

The custom 12k BPE tokenizer has fertility 1.63 vs GPT-2's 2.99 on Romanized Telugu — it splits words less aggressively. `"vivarinchandi"` is likely a single or two-token span under the custom vocab, meaning the model could in principle predict the whole word as its next token rather than just the first consonant.

### Practical result

In practice, the custom tok model's single top-1 success reveals the mechanism: for example 45 (`"...somavaaram 1-ganta card"` → `"roja"`), the word *roja* (day) is a single high-frequency token in the custom vocabulary. The model, despite mode collapse, occasionally outputs content tokens when the prefix strongly activates a frequent Telugu word.

### Why it still fails

1. **Embedding misalignment.** The custom vocabulary is mapped to truncated GPT-2 embedding slots. The attention mechanism, even after LoRA fine-tuning, operates over embeddings whose geometric structure does not correspond to the custom token semantics.

2. **Mode collapse to "mariyu".** The LoRA adapter only modifies 24 attention matrices (2 per layer × 12 layers). The final token distribution is overwhelmingly dominated by the un-updated embedding table. Because the embeddings are poorly learned, the model converges to the single safest output.

3. **Longer effective context is not exploited.** Fewer tokens per sentence means the model sees more semantic content per position — but this benefit is entirely wasted if the embeddings don't convey useful information.

The custom tokenizer experiment would need either: (a) full embedding retraining from scratch with more data and epochs, or (b) initialisation of the custom embedding table from subword-averaged GPT-2 embeddings rather than truncated slot assignment.

---

## 4. Highlighted Cases

### Case A — The Custom Tok's One Win
**Prefix:** `"e varam naa vyayam karyakalapala avalokanam ikkada vundi somavaaram 1-ganta card"`
**Correct:** `roja` (Telugu: day)

Custom tok predicts `"roja"` (top-1, ✓). GPT-2 fine-tuned and GPT-2 pretrained both fail; mGPT also gets `"roja"` via subword matching. This is the custom tok's only clean top-1 win. *Why?* "roja" is among the highest-frequency tokens in the custom BPE vocabulary — it appears in almost every day-of-week reference in the WhatsApp corpus. Even a partially collapsed model can lock onto very-high-frequency tokens when the prefix provides strong enough signal (the preceding words are all routine schedule-reporting Telugu). This case shows that the custom tokenizer does have the theoretical advantage of atomic whole-word tokens for common words, but that advantage only surfaces when frequency overwhelms the embedding disorder.

---

### Case B — mGPT Reads Hindi Fluently
**Prefix:** `"yeh africa ka gaav jaisa lag rha"`
**Correct:** `hai` (Hindi copula)

mGPT predicts `"hai"` (top-1, ✓). Every other model fails. The prefix is unambiguously Hindi — *"yeh...jaisa lag rha hai"* (this looks like...it is). mGPT was pretrained on a 60-language corpus including Hindi, so it has the syntactic knowledge to complete this construction. GPT-2 pretrained and fine-tuned both output noise (`"ay"`, `"n"`, `"k"`). This is the clearest demonstration that mGPT's multilingual pretraining covers not just Romanized Telugu but the full code-switching landscape of South Asian WhatsApp — Hindi, Bengali, and transliterated scripts all benefit from its training distribution in ways that monolingual GPT-2 fine-tuning cannot replicate.

---

### Case C — Abbreviation vs Spelled-Out Word
**Prefix:** `"no pa cancelled going next week didn t wake up till 2pm kada impossible for me to go to the station from kandi in 2 5"`
**Correct:** `hrs` (abbreviation for hours)

mGPT predicts `"hours"` (top-3, correct concept, wrong form). GPT-2 pretrained predicts `"mins"` (related unit, wrong). Fine-tuned GPT-2 predicts `"days"` (degraded). Custom tok predicts `"2"` (continuation of the numeral, not the unit). The correct answer is an abbreviation that GPT-2's tokenizer splits as `['h', 'rs']`; the top-1 target token is `'h'`, which no model predicts. *What this reveals:* Romanized WhatsApp text is dense with informal abbreviations (`hrs`, `mins`, `pg`, `pyq`) that are ambiguous tokens and fail the first-token matching criterion even when the model has the right semantic concept. mGPT comes closest because its multilingual training includes similar informal registers in other languages.

---

### Case D — Telugu Address Term after English Phrase
**Prefix:** `"have some shame"`
**Correct:** `anna` (Telugu: brother, used as respectful address)

All four models predict English continuations — `"in"`, `"on"`, `"to"`, `"about"`, `"mariyu"`. The correct next word is the Telugu address term *anna*, appended as a social marker (cf. English "have some shame, man"). This is a classic code-switch point: English reprimand + Telugu address. No model anticipates the switch because English phrases of this form almost never terminate with Telugu in standard training data. This type of failure — high-frequency English trigger phrase followed by an invariant Telugu social particle — probably accounts for a significant fraction of natural code-mixed conversation, and represents a structural gap in all four models.

---

### Case E — Wrong but Plausible: Antonym in Context
**Prefix:** `"yea they had done very"`
**Correct:** `poor` (the author's evaluative word)

Every model predicts `"well"` or `"good"`. This is the most honest failure in the set: statistically, "done very well" is far more common than "done very poor" in both English corpora and informal chat. The author chose the marked, negative evaluation. Fine-tuning on Telugu WhatsApp didn't help, because the fine-tuned model's English register knowledge was partially displaced without gaining any advantage for this English-only completion. *Lesson:* Next-word accuracy on naturally-occurring text is a lower bound on model intelligence; natural conversation is full of deliberate marked choices that any single-sequence-prediction model will statistically avoid.

---

### Case F — Hindi Leaking into Fine-Tuned GPT-2
**Prefix:** `"haan dekh rhe hain uspr"`
**Correct:** `bhi` (Hindi: also/too)

The fine-tuned GPT-2 (orig tok) predicts `"ab"` (Hindi: now) as top-1. GPT-2 pretrained predicts `"ay"` (nonsense). This is a subtle diagnostic: the fine-tuned model, trained on WhatsApp data that contains Hindi alongside Telugu, has absorbed enough Hindi to predict a grammatically plausible Hindi continuation — just not the right one. The label *bhi* would complete `"uspr bhi"` ("on that too"). The model predicts *ab* ("now"), a common Hindi discourse particle that often appears in conversational sentences. *Takeaway:* Fine-tuning on code-mixed corpora injects limited cross-lingual knowledge almost as a side effect; the model hasn't been explicitly trained for Hindi but shows traces of it. This cross-contamination is both a strength (partial multilingual coverage) and a risk (unpredictable mixing).

---

### Case G — Structural English Misread by All
**Prefix:** `"i asked mahvith took the"`
**Correct:** `keys`

GPT-2 pretrained: `"opportunity"`. mGPT: `"time"`. Fine-tuned GPT-2: `"place"`. Custom tok: `"mariyu"`. The prefix itself is syntactically ambiguous — "i asked mahvith took the ___" reads as a garbled merge of "I asked Mahvith" and "Mahvith took the ___". No model resolves the ambiguity correctly. *Mahvith* is a Telugu name, but its presence doesn't signal to any model that the context is a Telugu-register message in English. mGPT's `"time"` is the most structurally coherent English completion ("took the time"), but the author meant a physical object. This case illustrates how extreme code-mixing — English syntax with embedded Telugu names — creates prefixes that defeat both monolingual and multilingual models equally.

---

## 5. Patterns in Sentences That Break All Models

Four structural features reliably cause universal failure:

1. **Truncated context after tokenization.** Several prefixes are long enough (> 128 tokens) that the model sees only a final slice. What looks like a coherent short-form prefix to a human reader is actually late-context deprived of its discourse frame.

2. **Label words that are not among the top-1000 tokens.** Slang (`bradhas`), domain jargon (`placements`, `pyq`), and morphological variants (`laantivi`, `sadhyame`) all lie far outside any model's frequent-token head. Even if the model correctly understands the semantic direction, it cannot produce an infrequent token as a top-5 prediction.

3. **Script-agnostic label when the prefix ends mid-phrase.** Prefixes that end on a content word without syntactic closure (e.g., `"send ai day"`, `"was feeding amma"`) leave the label as a nearly arbitrary content choice. Human evaluators could produce dozens of plausible next words; accuracy on any single gold label understates how reasonable the model's predictions may be.

4. **Language of the prefix mismatches the label's language.** When the prefix is English and the label is Telugu (or vice versa), every model defaults to the distributional statistics of the prefix language. No model has learned to predict the code-switch point itself as a structural phenomenon.

---

## 6. Summary Diagnosis

| Failure type | Count | Primarily hurts |
|---|---|---|
| Data sparsity | 26 | All models equally |
| Tokenization failure (subword split) | 19 | GPT-2 orig tok + pretrained |
| Wrong but plausible | 6 | All; fine-tuning doesn't help |
| English intrusion | 5 | Fine-tuned GPT-2 (degrades English register) |
| Extreme code-mixing | 3 | All models equally |
| Hindi intrusion | 3 | GPT-2 variants; mGPT partially handles |
| Custom tok mode collapse | 100* | Custom tok exclusively |

*The custom tok mode collapse affects all 100 examples as a systematic failure, independent of other categories.

**The most actionable insights:**

- **mGPT is the correct baseline for code-mixed South Asian text**, not GPT-2. Its multilingual pretraining provides Hindi, transliteration, and informal register coverage that no amount of GPT-2 fine-tuning can replicate in a few epochs.
- **Tokenization is a harder problem than fine-tuning.** GPT-2's 50k English BPE is structurally wrong for Romanized Telugu. The top-k accuracy metric is largely measuring subword first-token prediction, not word prediction.
- **The custom tokenizer path is not abandoned, just under-resourced.** The theoretical argument (lower fertility, whole-word tokens for common Telugu words) is correct. What failed is the training recipe: embedding initialisation, training duration, and the LoRA-only update constraint together prevented convergence.
- **Code-switching at the phrase boundary is the unsolved problem.** When the correct next word is in a different language than the last five words, every model in this study fails with near-certainty. This is the frontier task for Romanized Telugu modelling.
