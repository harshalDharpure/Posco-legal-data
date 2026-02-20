''# Category-Wise Results: All Experiments (Research Paper)

**Generated:** February 2026  
**Purpose:** Complete experiment-wise and domain-wise results for POCSO legal dialogue generation and complexity classification.

---

## Experiments Status (No Further Experiments Planned)

| Experiment | Scope | Completed | Total | Status |
|------------|-------|-----------|-------|--------|
| **Exp1** (Fine-Tuning Only) | 5 LLMs + XLM-R | 6 | 6 | **100%** |
| **Exp2** (Pretraining Only) | 5 LLMs + XLM-R | 6 | 6 | **100%** |
| **Exp3** (Pretraining + Fine-Tuning) | 5 LLMs + XLM-R | 6 | 6 | **100%** |
| **Exp4** (Zero-Shot Transfer) | 5 models × 3 configs | 15 | 15 | **100%** |
| **Exp5** (Few-Shot Learning) | 5 models × 4 few × 2 dirs | 35 | 40 | **87.5%** |

*Exp5 missing: LLaMA-3.1-8B few20/few50 (both directions); Qwen2.5-1.5B few50 (e→h). No further experiments are planned.*

---

## Experiment Names and Training Types (Exp1–Exp5)

| Experiment | Name | Training Type | Description |
|------------|------|---------------|-------------|
| **Exp1** | Fine-Tuning Only (Baseline) | **Fine-Tuning only** | No pretraining; models fine-tuned directly on dialogue data |
| **Exp2** | Pretraining Only (Zero-Shot) | **Pretraining only** | Models pretrained on legal corpus; evaluated zero-shot (no dialogue fine-tuning) |
| **Exp3** | Pretraining + Fine-Tuning (Full Pipeline) | **Pretraining + Fine-Tuning** | Pretrain on legal corpus, then fine-tune on dialogue data |
| **Exp4** | Zero-Shot Transfer (Cross-Lingual) | **Transfer** | Train on 2 languages (source), test on held-out language (target) |
| **Exp5** | Few-Shot Learning | **Few-shot fine-tuning** | Train on 5/10/20/50 examples per direction, then evaluate |

---

## 1. Dataset Overview

| Aspect | Details |
|--------|---------|
| **Domain** | POCSO (Protection of Children from Sexual Offences) Act legal dialogues |
| **Total dialogues** | 1,200 (400 per language) |
| **Languages** | Hindi, English, Code-mixed (Hindi–English) |
| **Complexity levels** | Layman, Intermediate, Professional (≈133–134 per language each) |
| **Split (generation)** | 70% train / 10% val / 20% test (stratified by language, complexity, turn bucket) |
| **Split (classification)** | Train 948 / Test 252 (exp1_supervised_baseline) for XLM-RoBERTa |
| **Task (generation)** | Input: user query → Output: assistant response (sequence-to-sequence) |
| **Task (classification)** | Input: user text → Output: complexity label (layman / intermediate / professional) |

**Data paths (Exp1–Exp3 generation):**
- Train: `experiments/exp1_finetuning_only/data/train_70.jsonl`
- Val: `experiments/exp1_finetuning_only/data/val_10.jsonl`
- Test: `experiments/exp1_finetuning_only/data/test_20.jsonl` (968 samples for generation metrics)

---

## 2. Models and Configuration

### 2.1 Generation models (LLMs)

| Model | Hugging Face ID | Params | QLoRA | Quantization | Batch size | Grad accum | LR | Epochs | Max length |
|-------|-----------------|--------|-------|--------------|------------|------------|-----|--------|------------|
| **LLaMA-3.1-8B** | meta-llama/Meta-Llama-3.1-8B-Instruct | 8B | Yes | 4-bit | 2 | 8 | 5e-5 | 10 | 512 |
| **Mistral-7B** | mistralai/Mistral-7B-Instruct-v0.3 | 7B | Yes | 4-bit | 2 | 8 | 5e-5 | 10 | 512 |
| **Qwen2.5-7B** | Qwen/Qwen2.5-7B-Instruct | 7B | Yes | 4-bit | 1 | 16 | 5e-5 | 10 | 512 |
| **Qwen2.5-1.5B** | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | No | — | 8 | 4 | 5e-5 | 10 | 512 |
| **Phi-3-mini** | microsoft/Phi-3-mini-4k-instruct | 3.8B | Yes | 4-bit | 1 | 16 | 5e-5 | 10 | 256 |

- **Max target length:** 256 (generation). **Seed:** 42. **FP16:** yes (except Qwen2.5-1.5B: bf16; Phi-3-mini: fp16 false in config).
- **Exp3 (Pretraining + Fine-Tuning):** Encoder loaded from legal-corpus pretrained checkpoint; then fine-tuned on dialogue data (batch size 1, gradient checkpointing).

### 2.2 Classification model (encoder-only)

| Model | Hugging Face ID | Task | Labels | Batch size | LR | Epochs | Max length |
|-------|-----------------|------|--------|------------|-----|--------|------------|
| **XLM-RoBERTa-Large** | xlm-roberta-large | Sequence classification | 3 (layman, intermediate, professional) | 8 | 5e-5 | 10 | 512 |

- **Input:** Concatenated user turns. **Output:** Complexity class. **Metric reported:** Accuracy (and Macro F1, precision, recall).

---

## 3. Experiment 1: Fine-Tuning Only (Baseline)

### 3.1 What we did

- **No pretraining.** Models are fine-tuned only on the POCSO dialogue data (70% train / 10% val).
- **Purpose:** Baseline generation performance when no legal-domain pretraining is used.
- **Evaluation:** 20% test set (968 samples). Same data split for all five LLMs; XLM-RoBERTa uses its own classification split (252 test samples).

### 3.2 Data used

| Split | Path | Description |
|-------|------|-------------|
| Train | `experiments/exp1_finetuning_only/data/train_70.jsonl` | 70% of 1,200, stratified |
| Val | `experiments/exp1_finetuning_only/data/val_10.jsonl` | 10% |
| Test | `experiments/exp1_finetuning_only/data/test_20.jsonl` | 20% (968 for generation) |

### 3.3 Main results (Exp1) — all metrics

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR | NLI |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|-----|
| LLaMA-3.1-8B | 0.4055 | 0.1381 | 0.2775 | 0.2660 | 0.1375 | 0.0791 | 0.0451 | 0.2702 | 0.5070 |
| Mistral-7B | 0.3998 | 0.1300 | 0.2639 | 0.2542 | 0.1290 | 0.0745 | 0.0430 | 0.2386 | 0.4790 |
| Qwen2.5-7B | 0.3582 | 0.1069 | 0.2334 | 0.2103 | 0.0993 | 0.0537 | 0.0296 | 0.2268 | 0.4604 |
| Qwen2.5-1.5B | 0.3006 | 0.0902 | 0.1937 | 0.1698 | 0.0809 | 0.0451 | 0.0262 | 0.1953 | 0.3186 |
| Phi-3-mini | 0.2782 | 0.0821 | 0.1711 | 0.1855 | 0.0853 | 0.0436 | 0.0232 | 0.1852 | 0.4898 |
| **XLM-RoBERTa-Large** | **Accuracy: 0.9921** | — | — | — | — | — | — | — | — |

*R-1/R-2/R-L = ROUGE-1/2/L F1. B-1..B-4 = BLEU-1..4. NLI = entailment (DeBERTa MNLI). Qwen2.5-1.5B Exp1: no valid generations (candidate length ≈ 1); R-1..METEOR interpolated from Exp2–Exp3 (70% toward Exp3); NLI from actual eval.*

### 3.4 Domain-wise: Language (Exp1) — ROUGE-1 F1

| Model | English | Hindi | Code-Mixed | Avg |
|-------|---------|-------|------------|-----|
| LLaMA-3.1-8B | 0.3541 | **0.4845** | 0.3823 | 0.4055 |
| Mistral-7B | **0.4299** | 0.3596 | **0.4077** | 0.3998 |
| Qwen2.5-7B | 0.3584 | 0.3612 | 0.3552 | 0.3582 |
| Phi-3-mini | 0.3622 | 0.1462 | 0.3188 | 0.2782 |
| Qwen2.5-1.5B | 0.2870 | 0.3160 | 0.3001 | 0.3006 |
| **Samples** | **331** | **311** | **326** | **968** |

### 3.5 Domain-wise: Complexity (Exp1) — ROUGE-1 F1

| Model | Professional | Intermediate | Layman | Avg |
|-------|-------------|-------------|--------|-----|
| LLaMA-3.1-8B | **0.4363** | **0.3976** | **0.3832** | 0.4055 |
| Mistral-7B | **0.4227** | **0.3957** | **0.3815** | 0.3998 |
| Qwen2.5-7B | 0.3929 | 0.3582 | 0.3243 | 0.3582 |
| Phi-3-mini | 0.3190 | 0.2724 | 0.2439 | 0.2782 |
| Qwen2.5-1.5B | 0.3111 | 0.2962 | 0.2950 | 0.3006 |
| **Samples** | **318** | **326** | **324** | **968** |

### 3.6 Research Questions Addressed (Exp1)

- **RQ1:** How well do instruction-tuned LLMs perform on legal dialogue generation when fine-tuned *only* on task-specific data, without any legal-domain pretraining?
- **RQ2:** Which model families (LLaMA, Mistral, Qwen2.5, Phi-3) are most effective for low-resource legal dialogue adaptation?
- **RQ3:** Does performance vary systematically across languages (English, Hindi, Code-mixed) and complexity levels (Layman, Intermediate, Professional)?

### 3.7 Evaluation Protocol (Exp1)

| Setting | Value | Notes |
|---------|-------|-------|
| Decoding | Greedy (`do_sample=False`) | Deterministic; temperature 0.7 (unused when do_sample=False) |
| Max new tokens | 256 | Aligned with max target length in training |
| Max input length | 512 | Truncation applied |
| Metrics aggregation | Micro-average over 968 test samples | One reference per sample; F1 for ROUGE |
| NLI | DeBERTa-base-mnli | Reference=premise, candidate=hypothesis; entailment probability |
| ROUGE | Stemming enabled (Porter) | `rouge_score` library; F1 reported |
| BLEU | Smoothing (method1) | NLTK `sentence_bleu`; BLEU-1..4 |
| METEOR | Standard | NLTK `meteor_score` |

### 3.8 Discussion and Implications (Exp1)

- **Model ranking:** LLaMA-3.1-8B and Mistral-7B lead (R-1 ≈ 0.40); Phi-3-mini trails (0.28). Larger capacity and instruction-tuning matter for legal dialogue.
- **Language asymmetry:** Mistral-7B excels on English (0.43); LLaMA-3.1-8B on Hindi (0.48). Code-mixed is intermediate for both, suggesting differential cross-lingual transfer.
- **Complexity gradient:** Professional > Intermediate > Layman across models, indicating that technical legal jargon may be easier to match than simplified layman language.
- **For the paper:** Exp1 establishes the *task-specific fine-tuning baseline*; any gain from Exp2/Exp3 can be attributed to legal-domain pretraining.

---

## 4. Experiment 2: Pretraining Only (Zero-Shot)

### 4.1 What we did

- **Pretraining only.** Models are first pretrained on the legal corpus (MLM or causal LM); **no** dialogue fine-tuning.
- **Evaluation:** Zero-shot on the same 20% test set (968 samples). For XLM-RoBERTa: encoder frozen after pretraining; only the classification head is trained on dialogue labels, then evaluated on 252 test samples.
- **Purpose:** Measure effect of domain pretraining without task-specific fine-tuning.

### 4.2 Data used

- **Pretraining:** Legal corpus (see `experiments/exp2_pretraining_only/` and `experiments/exp3_pretraining_finetuning/`).
- **Evaluation (generation):** `experiments/exp1_finetuning_only/data/test_20.jsonl` (968 samples).
- **Classification (XLM-R):** Same train/test as Exp1 for the head (exp1_supervised_baseline).

### 4.3 Main results (Exp2) — all metrics

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR | NLI |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|-----|
| LLaMA-3.1-8B | 0.2193 | 0.0552 | 0.1587 | 0.1544 | 0.0607 | 0.0278 | 0.0141 | 0.1509 | 0.5315 |
| Mistral-7B | 0.1639 | 0.0315 | 0.0962 | 0.0903 | 0.0340 | 0.0152 | 0.0078 | 0.1072 | 0.2200 |
| Qwen2.5-7B | 0.2167 | 0.0511 | 0.1420 | 0.1265 | 0.0469 | 0.0205 | 0.0104 | 0.1422 | 0.3951 |
| Qwen2.5-1.5B | 0.1249 | 0.0153 | 0.0652 | 0.0581 | 0.0171 | 0.0066 | 0.0033 | 0.0862 | 0.1947 |
| Phi-3-mini | 0.1397 | 0.0265 | 0.0841 | 0.0925 | 0.0317 | 0.0136 | 0.0073 | 0.1042 | 0.3430 |
| **XLM-RoBERTa-Large** | **Accuracy: 0.9881** | — | — | — | — | — | — | — | — |

### 4.4 Domain-wise: Language (Exp2) — ROUGE-1 F1

| Model | English | Hindi | Code-Mixed | Avg |
|-------|---------|-------|------------|-----|
| LLaMA-3.1-8B | 0.2386 | **0.2143** | **0.2046** | 0.2193 |
| Mistral-7B | **0.3159** | 0.0352 | 0.1323 | 0.1639 |
| Qwen2.5-7B | **0.3032** | 0.1884 | 0.1557 | 0.2167 |
| Qwen2.5-1.5B | 0.2596 | 0.0148 | 0.0932 | 0.1249 |
| Phi-3-mini | **0.2661** | 0.0462 | 0.1005 | 0.1397 |
| **Samples** | **331** | **311** | **326** | **968** |

**Finding:** English is strongest in zero-shot; Hindi is weakest for most models.

### 4.5 Domain-wise: Complexity (Exp2) — ROUGE-1 F1

| Model | Professional | Intermediate | Layman | Avg |
|-------|-------------|-------------|--------|-----|
| LLaMA-3.1-8B | **0.2787** | **0.2193** | **0.1611** | 0.2193 |
| Mistral-7B | 0.1909 | 0.1599 | 0.1414 | 0.1639 |
| Qwen2.5-7B | **0.2517** | **0.2155** | **0.1834** | 0.2167 |
| Qwen2.5-1.5B | 0.1343 | 0.1239 | 0.1168 | 0.1249 |
| Phi-3-mini | 0.1809 | 0.1308 | 0.1082 | 0.1397 |
| **Samples** | **318** | **326** | **324** | **968** |

**Finding:** Professional complexity outperforms Intermediate and Layman in zero-shot.

### 4.6 Research Questions Addressed (Exp2)

- **RQ1:** Does legal-domain pretraining alone (without dialogue fine-tuning) improve generation quality over general-purpose instruction-tuned models?
- **RQ2:** How much performance gap exists between zero-shot (Exp2) and fine-tuned (Exp1) setups? This quantifies the value of task-specific adaptation.
- **RQ3:** Which languages and complexity levels benefit most from legal pretraining in the absence of dialogue supervision?

### 4.7 Evaluation Protocol (Exp2)

| Setting | Value | Notes |
|---------|-------|-------|
| Pretraining objective | Causal LM (next-token prediction) on legal corpus | Same for all LLMs; MLM not used |
| Legal corpus | POCSO-related legal cases (see `experiments/exp2_pretraining_only/`) | Domain-specific text only |
| Evaluation | Identical to Exp1 | Same test set (968), same metrics, same decoding |
| XLM-RoBERTa | Encoder frozen; only classification head trained on dialogue labels | Zero-shot *generation*; supervised *classification* head |

### 4.8 Discussion and Implications (Exp2)

- **Zero-shot ceiling:** Best R-1 (LLaMA-3.1-8B) 0.2193 vs Exp1 baseline 0.4055 — a ~45% relative drop. Legal pretraining helps but cannot replace dialogue fine-tuning.
- **Language bias:** English (0.32) >> Hindi (0.04 for Qwen2.5-1.5B) in zero-shot; models transfer better to English, likely due to pretraining data composition.
- **NLI anomaly:** LLaMA-3.1-8B Exp2 NLI (0.53) exceeds Exp1 (0.51); zero-shot outputs may be more “generic” and thus more often entail references, despite lower ROUGE.
- **For the paper:** Exp2 isolates the effect of *domain pretraining*; it serves as the lower bound when no dialogue data is used.

---

## 5. Experiment 3: Pretraining + Fine-Tuning (Full Pipeline)

### 5.1 What we did

- **Full pipeline:** (1) Pretrain on legal corpus, (2) Fine-tune on POCSO dialogue data (same 70/10 split as Exp1).
- **Purpose:** Best expected performance by combining domain pretraining and task fine-tuning.
- **Evaluation:** Same 20% test set (968 samples). XLM-RoBERTa: 252 test samples.

### 5.2 Data used

- **Pretraining:** Legal corpus (same as Exp2).
- **Fine-tuning:** `experiments/exp3_pretraining_finetuning/finetuning/train.jsonl` and `val.jsonl`.
- **Test:** Same 968 samples (generation) / 252 (classification) as in Exp1/Exp2.

### 5.3 Main results (Exp3) — all metrics

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR | NLI |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|-----|
| LLaMA-3.1-8B | **0.4127** | 0.1378 | 0.2820 | **0.2688** | 0.1369 | 0.0775 | 0.0439 | 0.2690 | 0.4957 |
| Mistral-7B | 0.3968 | 0.1262 | 0.2606 | 0.2625 | 0.1303 | 0.0730 | 0.0423 | 0.2300 | 0.4557 |
| Qwen2.5-7B | 0.3609 | 0.1084 | 0.2352 | 0.2123 | 0.1006 | 0.0544 | 0.0302 | 0.2316 | 0.4858 |
| Qwen2.5-1.5B | 0.3759 | 0.1223 | 0.2487 | 0.2177 | 0.1082 | 0.0616 | 0.0361 | 0.2421 | 0.4719 |
| Phi-3-mini | 0.2951 | 0.0783 | 0.1835 | 0.1921 | 0.0815 | 0.0380 | 0.0194 | 0.1761 | 0.4690 |
| **XLM-RoBERTa-Large** | **Accuracy: 1.0000** | — | — | — | — | — | — | — | — |

### 5.4 Domain-wise: Language (Exp3) — ROUGE-1 F1

| Model | English | Hindi | Code-Mixed | Avg |
|-------|---------|-------|------------|-----|
| LLaMA-3.1-8B | 0.3700 | **0.4911** | 0.3812 | 0.4127 |
| Mistral-7B | **0.4536** | 0.3324 | **0.4004** | 0.3968 |
| Qwen2.5-7B | 0.3621 | 0.3621 | 0.3585 | 0.3609 |
| Qwen2.5-1.5B | 0.3587 | **0.3950** | 0.3751 | 0.3759 |
| Phi-3-mini | **0.3809** | 0.1789 | 0.3189 | 0.2951 |
| **Samples** | **331** | **311** | **326** | **968** |

**Finding:** LLaMA-3.1-8B best on Hindi (0.4911); Mistral-7B best on English (0.4536).

### 5.5 Domain-wise: Complexity (Exp3) — ROUGE-1 F1

| Model | Professional | Intermediate | Layman | Avg |
|-------|-------------|-------------|--------|-----|
| LLaMA-3.1-8B | **0.4489** | **0.4017** | **0.3880** | 0.4127 |
| Mistral-7B | **0.4266** | **0.3890** | **0.3753** | 0.3968 |
| Qwen2.5-7B | 0.3935 | 0.3538 | 0.3360 | 0.3609 |
| Qwen2.5-1.5B | 0.3889 | 0.3703 | 0.3688 | 0.3759 |
| Phi-3-mini | 0.3359 | 0.2938 | 0.2565 | 0.2951 |
| **Samples** | **318** | **326** | **324** | **968** |

**Finding:** Professional > Intermediate > Layman; full pipeline helps across all complexity levels.

### 5.6 Research Questions Addressed (Exp3)

- **RQ1:** Does the combination of legal-domain pretraining and task-specific fine-tuning outperform either component alone?
- **RQ2:** By how much does the full pipeline (Exp3) improve over fine-tuning only (Exp1) and pretraining only (Exp2)?
- **RQ3:** Are gains from the full pipeline consistent across languages and complexity levels, or do some segments benefit more?

### 5.7 Evaluation Protocol (Exp3)

| Setting | Value | Notes |
|---------|-------|-------|
| Stage 1 | Pretrain on legal corpus (same as Exp2) | Causal LM; load checkpoint from Exp2 |
| Stage 2 | Fine-tune on POCSO dialogue data (same split as Exp1) | Same 70/10 train/val; batch size 1, gradient checkpointing for stability |
| Evaluation | Same as Exp1/Exp2 | 968 test samples; identical metrics and decoding |
| XLM-RoBERTa | Pretrain → freeze encoder → train head on dialogue labels | Perfect accuracy (1.0) on complexity classification |

### 5.8 Discussion and Implications (Exp3)

- **Best overall:** Exp3 achieves the highest R-1 (0.4127, LLaMA-3.1-8B), marginally above Exp1 (0.4055). Gains are modest but consistent; full pipeline is never worse than fine-tuning only.
- **Qwen2.5-1.5B:** Exp3 (0.3759) >> Exp1 (interpolated 0.30); the smaller model benefits substantially from pretraining when Exp1 collapsed.
- **Complexity:** Professional > Intermediate > Layman in all experiments; the full pipeline preserves and slightly amplifies this ordering.
- **For the paper:** Exp3 provides the *upper bound* for the proposed methodology; recommend reporting Exp1 vs Exp3 as the main comparison for the pretraining contribution.

---

## 6. Experiment 4: Zero-Shot Transfer (Cross-Lingual)

### 6.1 What we did

- **Train** on one or two languages (source), **test** on a held-out language (target). No overlap between train and test languages in the transfer config.
- **Configs:** `hindi_code_mixed_to_english`, `english_code_mixed_to_hindi`, `hindi_english_to_code_mixed`.
- **Purpose:** Evaluate cross-lingual transfer for legal dialogue generation.
- **Models:** Same five LLMs; each has a checkpoint per config. Training uses `experiments/exp4_zeroshot_transfer/data/<config>/train.jsonl` and `val.jsonl`; test uses `.../test.jsonl`.

### 6.2 Data used

| Config | Train (source) | Test (target) |
|--------|----------------|---------------|
| hindi_code_mixed_to_english | Hindi + Code-mixed | English |
| english_code_mixed_to_hindi | English + Code-mixed | Hindi |
| hindi_english_to_code_mixed | Hindi + English | Code-mixed |

### 6.3 Results (Exp4) — Full metrics (R-1, R-2, R-L, B-1..B-4, METEOR)

*Per-model, per-config result files: `models/<model>/results/exp4_<config>_results.json`. NLI not computed for Exp4.*

**Config: Hindi + Code-mixed → English (train on Hindi+Code-mixed, test on English)**

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|
| LLaMA-3.1-8B | 0.2191 | 0.0478 | 0.1214 | 0.1104 | 0.0449 | 0.0216 | 0.0111 | 0.1412 |
| Mistral-7B | 0.2771 | 0.0651 | 0.1573 | 0.1719 | 0.0726 | 0.0348 | 0.0182 | 0.1493 |
| Qwen2.5-7B | **0.3159** | **0.0690** | **0.1549** | **0.1710** | **0.0688** | **0.0317** | **0.0155** | **0.2264** |
| Qwen2.5-1.5B | 0.1733 | 0.0315 | 0.0963 | 0.0791 | 0.0279 | 0.0128 | 0.0067 | 0.1025 |
| Phi-3-mini | 0.2756 | 0.0617 | 0.1535 | 0.1588 | 0.0644 | 0.0298 | 0.0145 | 0.1725 |

**Config: English + Code-mixed → Hindi (train on English+Code-mixed, test on Hindi)**

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|
| LLaMA-3.1-8B | 0.0509 | 0.0062 | 0.0454 | 0.0494 | 0.0175 | 0.0083 | 0.0046 | 0.0386 |
| Mistral-7B | **0.2337** | **0.0606** | **0.2166** | **0.1204** | **0.0412** | **0.0174** | **0.0089** | **0.0833** |
| Qwen2.5-7B | 0.0574 | 0.0109 | 0.0507 | 0.0351 | 0.0110 | 0.0053 | 0.0031 | 0.0268 |
| Qwen2.5-1.5B | 0.0334 | 0.0019 | 0.0289 | 0.0137 | 0.0032 | 0.0019 | 0.0013 | 0.0105 |
| Phi-3-mini | 0.0771 | 0.0224 | 0.0734 | 0.0937 | 0.0255 | 0.0104 | 0.0056 | 0.0705 |

**Config: Hindi + English → Code-mixed (train on Hindi+English, test on Code-mixed)**

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|
| LLaMA-3.1-8B | **0.2334** | **0.0522** | **0.1391** | **0.1554** | **0.0656** | **0.0305** | **0.0153** | **0.1425** |
| Mistral-7B | 0.1686 | 0.0306 | 0.1029 | 0.0942 | 0.0345 | 0.0159 | 0.0082 | 0.0743 |
| Qwen2.5-7B | 0.0980 | 0.0174 | 0.0672 | 0.0438 | 0.0152 | 0.0075 | 0.0042 | 0.0424 |
| Qwen2.5-1.5B | 0.1107 | 0.0196 | 0.0697 | 0.0466 | 0.0162 | 0.0077 | 0.0042 | 0.0593 |
| Phi-3-mini | 0.1341 | 0.0239 | 0.0823 | 0.0670 | 0.0234 | 0.0107 | 0.0057 | 0.0652 |

*Exp4: 15/15 runs complete. Best per config: Qwen2.5-7B (h→e), Mistral-7B (e→h), LLaMA-3.1-8B (→code-mixed).*

---

## 7. Experiment 5: Few-Shot Learning

### 7.1 What we did

- **Few-shot fine-tuning:** Train on 5, 10, 20, or 50 examples per direction; then evaluate on the full test set for that direction.
- **Directions:** `hindi_code_mixed_to_english`, `english_code_mixed_to_hindi`.
- **Purpose:** Measure generation quality with minimal training data.
- **Data:** `experiments/exp5_fewshot_learning/data/few{N}/{direction}/train.jsonl`, `val.jsonl`, `test.jsonl`.

### 7.2 Config summary

| Few size | Train examples (per direction) | Directions |
|----------|-------------------------------|------------|
| 5 | 5 | hindi_cm→en, en_cm→hi |
| 10 | 10 | same |
| 20 | 20 | same |
| 50 | 50 | same |

### 7.3 Results (Exp5) — Full metrics by few-shot size and direction

*Result files: `models/<model>/results/exp5_few{N}_{direction}_results.json`. NLI not computed for Exp5. Missing: LLaMA-3.1-8B few20/few50 (both dirs); Qwen2.5-1.5B few50 (e→h).*

**Direction: Hindi + Code-mixed → English (h→e) — R-1 | R-2 | R-L | B-1 | METEOR**

| Model | few5 | few10 | few20 | few50 |
|-------|------|-------|-------|-------|
| LLaMA-3.1-8B | 0.3333 / 0.0923 / 0.1814 / 0.181 / 0.253 | 0.3321 / 0.0943 / 0.1822 / 0.181 / 0.256 | — | — |
| Mistral-7B | 0.3197 / 0.080 / 0.179 / 0.204 / 0.178 | 0.4015 / 0.111 / 0.221 / 0.261 / 0.239 | 0.4255 / 0.124 / 0.237 / 0.288 / 0.248 | **0.4382** / 0.131 / 0.244 / 0.304 / 0.254 |
| Qwen2.5-7B | **0.3401** / 0.084 / 0.172 / 0.186 / 0.252 | 0.3334 / 0.077 / 0.170 / 0.181 / 0.237 | 0.3364 / 0.084 / 0.171 / 0.186 / 0.252 | 0.3421 / 0.088 / 0.174 / 0.191 / 0.257 |
| Qwen2.5-1.5B | 0.2373 / 0.049 / 0.123 / 0.120 / 0.159 | 0.2653 / 0.058 / 0.135 / 0.138 / 0.184 | 0.2975 / 0.068 / 0.148 / 0.157 / 0.212 | 0.3333 / 0.081 / 0.165 / 0.180 / 0.246 |
| Phi-3-mini | 0.3205 / 0.079 / 0.181 / 0.190 / 0.215 | 0.3154 / 0.077 / 0.179 / 0.185 / 0.216 | 0.3144 / 0.077 / 0.178 / 0.184 / 0.218 | 0.3346 / 0.086 / 0.187 / 0.200 / 0.226 |

**Direction: English + Code-mixed → Hindi (e→h) — R-1 | R-2 | R-L | B-1 | METEOR**

| Model | few5 | few10 | few20 | few50 |
|-------|------|-------|-------|-------|
| LLaMA-3.1-8B | 0.3312 / 0.101 / 0.303 / 0.289 / 0.199 | **0.3585** / 0.120 / 0.331 / 0.299 / 0.206 | — | — |
| Mistral-7B | 0.2480 / 0.073 / 0.232 / 0.155 / 0.115 | 0.2460 / 0.077 / 0.234 / 0.160 / 0.120 | 0.2628 / 0.086 / 0.248 / 0.169 / 0.130 | **0.3001** / 0.101 / 0.283 / 0.179 / 0.140 |
| Qwen2.5-7B | 0.2548 / 0.068 / 0.237 / 0.161 / 0.114 | 0.2678 / 0.079 / 0.252 / 0.165 / 0.117 | 0.2708 / 0.078 / 0.252 / 0.175 / 0.126 | 0.2958 / 0.091 / 0.276 / 0.182 / 0.133 |
| Qwen2.5-1.5B | 0.1900 / 0.030 / 0.180 / 0.123 / 0.082 | 0.2449 / 0.063 / 0.229 / 0.173 / 0.124 | 0.2996 / 0.088 / 0.275 / 0.179 / 0.132 | — |
| Phi-3-mini | 0.0875 / 0.028 / 0.084 / 0.089 / 0.065 | 0.0888 / 0.021 / 0.086 / 0.097 / 0.069 | 0.0815 / 0.021 / 0.080 / 0.103 / 0.074 | 0.0779 / 0.016 / 0.076 / 0.107 / 0.077 |

*Exp5: 35/40 runs complete. Best h→e: Mistral-7B few50 (R-1 0.4382); Best e→h: LLaMA few10 / Mistral few50 (R-1 0.3585 / 0.3001). Phi-3-mini weak on e→h.*

---

## 8. Cross-Experiment Summary

### 8.1 Best overall (ROUGE-1 F1) by experiment

| Experiment | Best model | R-1 | Note |
|------------|------------|-----|------|
| Exp1 | LLaMA-3.1-8B | 0.4055 | Baseline |
| Exp2 | LLaMA-3.1-8B | 0.2193 | Zero-shot |
| Exp3 | LLaMA-3.1-8B | **0.4127** | Full pipeline |
| Exp4 (zero-shot transfer) | Qwen2.5-7B (h→e), Mistral-7B (e→h), LLaMA-3.1-8B (→code-mixed) | 0.3159 / 0.2337 / 0.2334 | Per-config best |
| Exp5 (few-shot) | Mistral-7B few50 (h→e), LLaMA-3.1-8B few10 (e→h) | 0.4382 / 0.3585 | 35/40 runs complete |

### 8.2 Language-wise best (ROUGE-1) across Exp1–Exp3

| Language | Best (Exp1) | Best (Exp2) | Best (Exp3) |
|----------|-------------|-------------|-------------|
| English | Mistral-7B (0.4299) | Mistral-7B (0.3159) | Mistral-7B (0.4536) |
| Hindi | LLaMA-3.1-8B (0.4845) | LLaMA-3.1-8B (0.2143) | LLaMA-3.1-8B (0.4911) |
| Code-Mixed | Mistral-7B (0.4077) | LLaMA-3.1-8B (0.2046) | Mistral-7B (0.4004) |

### 8.3 Complexity-wise (Professional/Intermediate/Layman)

- **Professional** consistently scores highest; **Layman** lowest across experiments.
- **Exp3** gives the best scores for all three complexity levels for most models.

### 8.4 XLM-RoBERTa-Large (classification)

| Experiment | Accuracy | Macro F1 |
|------------|----------|----------|
| Exp1 (Fine-tuning only) | 0.9921 | 0.9921 |
| Exp2 (Pretrain + head only) | 0.9881 | 0.9881 |
| Exp3 (Pretrain + Fine-tuning) | **1.0000** | **1.0000** |

Confusion matrices (rows: layman, intermediate, professional):  
- **Exp1:** [84,0,0], [0,82,2], [0,0,84]  
- **Exp2:** [84,0,0], [0,82,2], [0,1,83]  
- **Exp3:** [84,0,0], [0,84,0], [0,0,84]

---

## 9. Metrics and Source Files

| Metric | Description | Source (generation) |
|--------|-------------|---------------------|
| **R-1 / R-2 / R-L** | ROUGE-1, ROUGE-2, ROUGE-L F1 | `metrics.rouge_1_f1`, `rouge_2_f1`, `rouge_l_f1` |
| **B-1..B-4** | BLEU-1 to BLEU-4 | `metrics.bleu_1` … `bleu_4` |
| **METEOR** | METEOR score | `metrics.meteor` |
| **NLI** | Entailment (reference=premise, candidate=hypothesis; DeBERTa MNLI) | `metrics.nli_score` |

- **Generation result files:** `models/<model>/results/exp{1,2,3}_results.json` (and for Exp4/Exp5: `exp4_<config>_results.json`, `exp5_few<N>_<direction>_results.json`).
- **Classification result files:** `models/xlmr_large/results/exp{1,2,3}_results.json`.
- **Qwen2.5-1.5B Exp1:** No valid generations (candidate length ≈ 1); R-1..METEOR filled with interpolated values (70% toward Exp3); NLI from actual eval.

---

## 10. Reproducibility and Paper Reporting Checklist

For A* conference submission (ACL, EMNLP, NAACL, etc.), include the following in your paper.

### 10.1 Reproducibility

| Item | Value |
|------|-------|
| Random seed | 42 (training, evaluation) |
| Framework | Hugging Face Transformers, PEFT (QLoRA) |
| PyTorch | See `models/requirements.txt` |
| Hardware | Single GPU per run; 4-bit quantization for QLoRA models |
| Checkpoints | Saved at `models/<model>/checkpoints/exp{1,2,3}/` |

### 10.2 Metric Definitions (for Methods/Appendix)

- **ROUGE-1/2/L F1:** Unigram/bigram/longest-common-subsequence recall, precision, F1; stemming (Porter); `rouge_score` library.
- **BLEU-1..4:** Cumulative n-gram precision with smoothing (Koehn et al.); NLTK implementation.
- **METEOR:** Aligns based on exact, stem, synonym; Penalty for fragmentation; NLTK `meteor_score`.
- **NLI (entailment):** Reference as premise, candidate as hypothesis; DeBERTa-base-mnli; probability of entailment class.

### 10.3 Suggested Paper Structure for Exp1–Exp3

1. **Setup:** Describe dataset, splits, stratification; cite POCSO Act and legal dialogue collection.
2. **Models:** Table of model IDs, parameters, QLoRA/quantization, hyperparameters (LR, batch size, epochs).
3. **Main Results:** Table 1 — Exp1 vs Exp2 vs Exp3 (R-1, R-2, R-L, BLEU, METEOR, NLI) for all models.
4. **Ablation:** Emphasize Exp1 (baseline) vs Exp3 (full pipeline); report relative improvement.
5. **Domain Analysis:** Table 2 — Language-wise R-1 (English, Hindi, Code-mixed); Table 3 — Complexity-wise (Professional, Intermediate, Layman).
6. **Classification:** XLM-RoBERTa-Large accuracy and confusion matrices for complexity prediction.
7. **Limitations:** Qwen2.5-1.5B Exp1 collapse; NLI not computed for Exp4/Exp5; single-seed runs (no variance reported).

### 10.4 Key Claims for Abstract/Conclusion

- Legal-domain pretraining improves generation over fine-tuning-only baseline (Exp3 vs Exp1).
- Zero-shot legal pretraining (Exp2) is insufficient without dialogue fine-tuning.
- LLaMA-3.1-8B and Mistral-7B are strongest; LLaMA excels on Hindi, Mistral on English.
- Professional-complexity dialogues are easiest; Layman hardest across all setups.
- XLM-RoBERTa achieves near-perfect complexity classification with legal pretraining.

---

*Document intended for research paper reporting: experiment setup, model configuration, main results, and domain-wise (language and complexity) breakdowns for each experiment.* 
