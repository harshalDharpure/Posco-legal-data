# Metric alignment and Exp2 detailed results

## 1. Is Avg aligned with the organizers’ DEV/TEST metric?

**Yes.** The **Avg** column in your tables is the same metric the organizers use for Subtask 1 (DimASR):

- **Official metric**: **RMSE_VA** from `evaluation_script/metrics_subtask_1_2_3.py`:
  - Formula: \(\sqrt{\frac{1}{n}\sum_i \bigl[ (\hat{v}_i - v_i)^2 + (\hat{a}_i - a_i)^2 \bigr]}\) (unnormalized).
  - Optional: with `--do_norm`, the script divides by \(\sqrt{128}\), so reported values can be lower.
- Your **Avg** is computed the same way (overall RMSE over the 2D VA error per sample). So Avg is directly comparable to organizer DEV/TEST leaderboard numbers.

**Why your 20% split Avg can be &lt; 1 while DEV/TEST might be around 1:**

- Different **splits**: your 20% test is from your own 80/20 split of the **training** set; DEV/TEST are the organizers’ held-out sets (different documents, possibly different difficulty).
- **Normalization**: If organizers report normalized RMSE (divide by \(\sqrt{128}\)), their numbers are smaller; if you use unnormalized, your numbers are directly in the [1,9] scale. Check the task’s evaluation description to match.
- **Val and Ars** are **not** the official metric: they are per-dimension RMSEs (valence-only and arousal-only). They are useful for analysis but go beyond what organizers use; dropping them from the main table is consistent with reporting only the official metric (Avg).

**Summary:** Avg = official metric; your lower Avg on the 20% split is consistent with using the same formula; Val/Ars are extra and can be dropped as you planned.

---

## 2. Limitations you outlined for Exp3

- **Outliers**: model struggles on extreme VA values.
- **Non-cropped predictions**: some predicted scores fall in 1–3, which are rarely used in the annotation scale (effective range often 4–6 or similar); clipping or post-processing could help.

These fit well in the “Limitations” subsection of the paper.

---

## 3. Exp2 detailed results (same structure as Exp3, Avg only)

**Exp2** = Pretrained model, **inference only** (no fine-tuning). Same 20% test split as Exp3. Below: **base** model only, **Avg** (overall RMSE) per language–domain. Val/Ars omitted so the table matches the organizer metric only.

| Lang | Domain    | 20% Test RMSE ↓ (Avg) |
|------|-----------|------------------------|
| eng  | Restaurant | 0.8841 |
| eng  | Laptop     | 0.7751 |
| jpn  | Hotel      | 0.4589 |
| jpn  | Finance    | 0.5887 |
| rus  | Restaurant | 1.0062 |
| ukr  | Restaurant | 1.0685 |
| tat  | Restaurant | 1.2505 |
| zho  | Restaurant | 0.4707 |
| zho  | Laptop     | 0.6058 |
| zho  | Finance    | 0.4681 |
| **Overall Avg** | | **0.8577** |

*Overall average computed over the 10 language–domain combinations above (same as Exp3 table).*

Comparison with Exp3 (base): Exp3 overall Avg is **0.5114** vs Exp2 **0.8577** — fine-tuning (Exp3) gives a large reduction in the official metric on this split.

---

## 4. Source

Exp2 numbers are taken from `COMPREHENSIVE_RESULTS_DOMAIN_WISE.md`, section “Experiment 2: VA-xlm-Roberta (Pretrained, Inference Only)”, **xlm-Roberta-base** rows only (RMSE column = Avg). No jpn Restaurant in that table for Exp2, so the Exp2 table has 10 rows vs Exp3’s 10 (jpn: Hotel + Finance only).
