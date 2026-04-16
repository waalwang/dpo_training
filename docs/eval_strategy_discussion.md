# Evaluation Strategy Discussion

Working notes from the design discussion for evaluating DPO-trained models
and pretrained backbones on this project. Organized by topic rather than as
a transcript.

## 1. The four axes we want to measure

The target is a conversational model trained on Reddit / HN trajectory pairs.
We want automatic proxies for four qualities before committing to human eval:

- Diverse
- Natural (human-like)
- Novel (not memorized from training)
- Easy-going (casual, conversational tone)

## 2. Metric menu per axis

### Diversity

- **Distinct-n** (Li et al., 2016, "A Diversity-Promoting Objective Function
  for Neural Conversation Models"). Unique n-grams / total n-grams. Use
  distinct-1 and distinct-2.
- **Self-BLEU** (Zhu et al., 2018, "Texygen"). For each generation in a group
  of k samples from the same prompt, BLEU against the other k-1. Lower is
  more diverse. Direct mode-collapse detector for DPO.
- **Semantic diversity** via pairwise cosine distance of Sentence-BERT /
  MPNet embeddings. Catches semantic collapse that surface n-grams miss.
- **N-gram entropy** of the generation corpus. Complementary to distinct-n.

### Naturalness / human-likeness

- **MAUVE** (Pillutla et al., NeurIPS 2021). Embedding-based divergence
  between model and human distributions. Single number, widely cited, well
  suited to open-ended generation. PyPI: `mauve-text`.
- **Forward + reverse perplexity** under a held-out LM (Zellers et al.,
  "Defending Against Neural Fake News"; Caccia et al., "Language GANs Falling
  Short").
- **UniEval** (Zhong et al., EMNLP 2022). T5-based reference-free evaluator
  scoring coherence, consistency, fluency, relevance.
- **LLM-as-judge benchmarks** (paid proxy for human eval):
  - MT-Bench (Zheng et al., NeurIPS 2023).
  - AlpacaEval 2.0 with length-controlled win rate (Dubois et al., 2024).
  - Arena-Hard-Auto (Li et al., 2024).

### Novelty

- **Training-set n-gram overlap**. Fraction of generated 4- or 8-grams that
  appear verbatim in training. Carlini et al., 2023 ("Quantifying Memorization
  Across Neural Language Models") is the standard reference.
- **Novelty vs. reference chosen**: distinct n-grams in generations absent
  from test-set chosen trajectories.
- **Embedding nearest-neighbor distance** to training.

### Easy-going / casual tone

No single accepted metric. Options:

- **Formality classifier** trained on GYAFC (Rao and Tetreault, NAACL 2018).
  Off-the-shelf: `s-nlp/roberta-base-formality-ranker`.
- **Politeness classifier** (Danescu-Niculescu-Mizil et al., ACL 2013).
  Orthogonal axis; combine with formality.
- **Readability**: Flesch Reading Ease or Dale-Chall. Casual conversational
  English ~60-80 Flesch; formal prose is below 30.
- **Domain-matched classifier**: train a binary "human Reddit/HN vs model"
  classifier on held-out data. The classifier's P(human) score is a direct
  domain-specific naturalness-plus-tone score.
- **FED** (Mehri and Eskenazi, SIGDIAL 2020). Reference-free dialogue quality
  via DialoGPT probes.
- **USR** (Mehri and Eskenazi, ACL 2020). Companion reference-free metric.

## 3. Priority subset we actually wired in

From the menu above, we committed to the cheapest high-signal subset:

1. **MAUVE** - naturalness.
2. **Distinct-1/2 + Self-BLEU** - diversity + mode-collapse detector.
3. **Training-set 8-gram overlap** - novelty / memorization check.
4. **Formality classifier** - easy-going proxy.

Skipped for now: UniEval, FED, USR, MT-Bench / AlpacaEval (paid API),
domain-trained classifier, semantic embedding diversity. All can be added
later without touching the existing metric functions.

## 4. Implementation in `scripts/eval.py`

### New top-level flag

- `--extra-metrics` turns on MAUVE + diversity + training-overlap + formality.
- `--diversity-k` (default 8) samples per prompt.
- `--diversity-prompts` (default 30) prompts used for diversity and MAUVE.
- `--novelty-ngram` (default 8) n-gram size for training-set overlap.
- `--novelty-train-docs` (default 5000) cap on training corpus scanned.

### New functions

- `generate_multi_samples(model, tokenizer, dataset, num_prompts, k)`
- `compute_diversity_metrics(groups)` - distinct-1/2, self-BLEU.
- `compute_training_overlap(generations, training_texts, n, max_training_docs)`
- `compute_mauve_score(generations, references)`
- `compute_formality_scores(texts)`

All optional dependencies (`sacrebleu`, `mauve-text`) are lazy-imported with
graceful skip and a warning log if missing.

### Baseline mode

`--baseline` loads `cfg["model"]["name"]` directly with no adapter and no
fine-tune. `--model-override <preset>` swaps backbones. Baseline uses the
same 4-bit quantization as the QLoRA path so pre-training and post-training
numbers are numerically comparable. Baseline outputs land in
`outputs/baseline_<preset>/eval_results.json`.

Same metric code runs in both modes, so every new metric we add gives you
both a backbone number and a DPO number for free.

## 5. Backbone scan workflow

Run before starting training to pin down a pre-DPO baseline per backbone:

```
python scripts/eval.py --baseline --profile local --extra-metrics
python scripts/eval.py --baseline --profile cloud --model-override qwen-7b --extra-metrics
python scripts/eval.py --baseline --profile cloud_qlora_big --model-override qwen-14b --extra-metrics
python scripts/eval.py --baseline --profile cloud_full --model-override llama-8b --extra-metrics
```

Then after DPO, point `--checkpoint` at the output and compare.

## 6. Hardware notes

### M3 MacBook Pro

- bitsandbytes is CUDA-only, so the existing `local` / `cloud*` profiles
  fail to load.
- Flash-Attention 2 is CUDA-only (already disabled in `local`).
- Required to make it work: a `mac` profile with `bf16: true`, no bnb, eager
  attention, and a branch in `load_model` that loads fp16/bf16 onto MPS.
- Fits (fp16, no quant):
  - Qwen2.5-1.5B (~3 GB weights): any M3.
  - Qwen2.5-7B (~14 GB): tight on 18 GB, OK on 36 GB.
  - Llama-3.1-8B (~16 GB): OK on 36 GB.
  - Qwen2.5-14B (~28 GB): M3 Max 64 GB+ only.
- MAUVE runs CPU-only there; slow but workable for a one-shot backbone scan.

Recommendation: use Mac for the 1.5B backbone, use cloud for 7B / 8B / 14B.

### Local server: Titan RTX 24 GB + 1080 Ti 11 GB

- Both CUDA. bitsandbytes and the `local` profile work as-is.
- Titan RTX (Turing): fp16 OK, no bf16, no FA2. `local` profile already
  matches these constraints.
- 1080 Ti (Pascal): fp16 is slow (no tensor cores), would bottleneck a mixed
  shard.
- 4-bit footprints for eval (no optimizer state):
  - Qwen2.5-1.5B ~1 GB
  - Qwen2.5-7B ~4 GB
  - Llama-3.1-8B ~5 GB
  - Qwen2.5-14B ~8 GB
- All four backbones fit on the Titan RTX alone for eval. The 1080 Ti is not
  required to make any of them run.
- Suggested split: pin the base LM to the Titan RTX via
  `CUDA_VISIBLE_DEVICES`; if RM scoring is enabled, put the RM on the
  1080 Ti to keep the base-LM path off the slower GPU. Current code loads
  both with `device_map="auto"` which will stack them on GPU 0 - strict
  split would need a small code change.

Recommendation: use this box for the full backbone scan. No code changes
needed; just run `--baseline --profile local --extra-metrics --model-override <key>`
for each preset.

## 7. What is not yet implemented but worth adding later

- Semantic diversity via Sentence-BERT embeddings.
- UniEval or FED for a dialogue-specific reference-free signal without GPT-4.
- Domain-matched "human vs model" classifier trained on held-out Reddit / HN.
- LLM-as-judge head-to-head (MT-Bench / AlpacaEval) when you want a headline
  number against published models.
- Strict RM-on-GPU1 pinning so the 1080 Ti can actually contribute on the
  local server.
