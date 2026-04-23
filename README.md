# MIP-BENCH: A Benchmark for Instructional Grounding in Medical Imaging Presentations

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-zabir1996%2Fmip--bench-yellow)](https://huggingface.co/datasets/zabir1996/mip-bench)
[![Code](https://img.shields.io/badge/GitHub-zabirul--islam%2Fmip--bench-blue)](https://github.com/zabirul-islam/mip-bench)

Official repository for the paper:

> **MIP-BENCH: A Benchmark for Instructional Grounding in Medical Imaging Presentations**
> Md Zabirul Islam, Md Motaleb Hossen Manik, Jianxu (Jayson) Wang, Yuxuan Liang, Ge Wang
> Submitted to NeurIPS 2026 (Datasets & Benchmarks Track)

---

## Quick Links

| Resource | URL |
|---|---|
| Code (this repo) | https://github.com/zabirul-islam/mip-bench |
| Dataset + Annotations | https://huggingface.co/datasets/zabir1996/mip-bench |

---

## Overview

MIP-BENCH is a diagnostic benchmark of **1,117 lecture slides** from a structured medical imaging
curriculum spanning 23 lectures, each paired with a structured concept graph encoding instructional intent:

```
I(S) = <C, R, B>
  C = target concepts the slide is designed to teach
  R = directed relational links among concepts
  B = scope boundary (what the slide covers and does not cover)
```

Models are evaluated with the **Instructional Grounding Score (IGS)**:

```
IGS = 0.40 * CR + 0.35 * RV + 0.25 * SF
```

where CR = Concept Recall, RV = Relational Validity, SF = Scope Fidelity.

The default IGS judge is GPT-4o, operating against fixed extracted ground truth `<C, R, B>` rather than free-form quality rating. IGS is cross-validated against (a) a structurally independent graph-based metric (r = 0.41–0.54), (b) pooled judgments of three independent human annotators (r = 0.729 on overall IGS, 88% of slides agreeing within 0.2), and (c) a second non-OpenAI LLM judge (Anthropic `claude-opus-4-6`) with per-model Pearson r_IGS ∈ [0.694, 0.835] and **identical cross-provider model ordering** (Kendall τ = 1.0).

---

## Key Results

| Model | IGS | CR | RV | SF | SF−CR | n |
|---|---|---|---|---|---|---|
| Qwen2-VL-7B | **0.642** | 0.633 | 0.563 | 0.768 | +0.134 | 183 |
| InternVL2-40B | 0.619 | 0.615 | 0.537 | 0.740 | +0.126 | 183 |
| InternVL2-26B | 0.596 | 0.597 | 0.521 | 0.699 | +0.103 | 183 |
| InternVL2-8B | 0.595 | 0.592 | 0.525 | 0.699 | +0.107 | 183 |
| LLaVA-1.6-34B | 0.501 | 0.492 | 0.421 | 0.629 | +0.137 | 183 |
| **Human (pooled, n=3 annotators)** | **0.690** | **0.717** | **0.644** | **0.711** | **−0.006** | **50** |

No VLM exceeds IGS = 0.642. Every VLM exhibits a universal **scope–concept gap**: SF consistently exceeds CR by +0.10 to +0.14, revealing implicit topic classification rather than explicit concept grounding. Pooled human annotators show no such gap (SF − CR = −0.006), confirming the pattern is a property of current VLMs rather than a rubric artifact.

### Two-Judge Ensemble (Claude + GPT-4o)

Identical rubric re-run with Anthropic `claude-opus-4-6` on the same 183 clean test slides. Model rankings are **identical** across providers (Kendall τ = 1.0, p = 0.017).

| Model | r_IGS (GPT vs Claude) | MAD | Ensemble IGS | 95% CI |
|---|---:|---:|---:|---|
| Qwen2-VL-7B | 0.767 | 0.133 | 0.614 | [0.581, 0.646] |
| InternVL2-40B | 0.835 | 0.119 | 0.594 | [0.560, 0.628] |
| InternVL2-26B | 0.736 | 0.148 | 0.573 | [0.541, 0.606] |
| InternVL2-8B | 0.788 | 0.129 | 0.571 | [0.537, 0.603] |
| LLaVA-1.6-34B | 0.694 | 0.168 | 0.457 | [0.425, 0.488] |

Claude is uniformly ~0.05–0.09 IGS points stricter than GPT-4o on absolute level; ordering is preserved. Compare models **within a fixed judge**. See paper §4.2 and Appendix `sec:supp_multi_judge`.

### Robustness Probes (reviewer-facing)

- **Recall-perturbation sweep** (`analysis/step15_recall_perturbation.py`): simulated 10/20/30 % gold-concept drop; rank Spearman ρ̄ = 0.94, max flip = 1.
- **Judge-family bias ablation** (`analysis/step17_judge_bias_ablation.py`): OLS Δβ = β_C − β_N ∈ [−0.69, +0.63]; judge does not preferentially reward GPT-4o-extractor vocabulary.
- **Pipeline transfer probe** (`evaluation/step18_transfer_probe.py`): extraction pipeline runs unchanged on a non-medical corpus (existence proof, no IGS reported).

---

## Repository Structure

```
mip-bench/                              ← this GitHub repo (code only)
├── data/
│   └── build_dataset_index.py          # Step 0: build master index + stratified splits
├── evaluation/
│   ├── step3_gpt4o.py                  # Step 1: GPT-4o concept graph extraction
│   ├── step4_vlm_inference.py          # Step 2: VLM inference (image-only, zero-shot)
│   ├── step5_gpt4o_judge.py            # Step 3: GPT-4o judge → IGS scores
│   ├── step5_claude_judge.py           # Step 3b: Claude (opus-4-6) judge → IGS scores
│   ├── step6_v4.py                     # Step 4: baseline metrics (BLEU, ROUGE, BERTScore)
│   ├── step7_v4.py                     # Step 5: human study validation (3 annotators)
│   └── step18_transfer_probe.py        # Step 11: pipeline transfer probe (non-medical corpus)
├── analysis/
│   ├── step10_gpt4o.py                 # main results (Section 6, Table 7)
│   ├── step11_v4.py                    # weight ablation (Appendix B, 171 combinations)
│   ├── step13_v4.py                    # curriculum concept graph construction
│   ├── step14_v4.py                    # graph topology + centrality analysis (Section 5)
│   ├── step15_recall_perturbation.py   # Step 9: extraction-recall drop sweep
│   ├── step16_multi_judge_agreement.py # Step 5b: GPT-4o vs Claude judge agreement
│   └── step17_judge_bias_ablation.py   # Step 10: judge-family OLS bias ablation
├── finetuning/
│   └── step12_eval_claims_v4.py        # LoRA fine-tuning + evaluation (Section 7)
├── paper/
│   ├── make_fig1.py                    # Figure 1 source (matplotlib)
│   ├── medlecturebench_fig_1.png       # Figure 1 raster (300 dpi)
│   ├── medlecturebench_fig_1.pdf       # Figure 1 vector
│   ├── medlecturebench_fig_qual.pdf     
│   └── neurips_2024.pdf                # Compiled paper (22 pages)
├── results/
│   ├── multi_judge_agreement.json      # per-model r, MAD, ensemble IGS + Kendall τ
│   ├── recall_perturbation_results.json# 10/20/30 % recall drop → ρ̄, max flip
│   └── judge_bias_ablation.json        # OLS β_C vs β_N per model
├── requirements.txt
├── environment.yml
└── README.md
```

> **Data lives on Hugging Face**, not in this repo. See the [dataset page](https://huggingface.co/datasets/zabir1996/mip-bench) for all slide images, concept graphs, IGS scores, splits, and human study files.

---

## What Is on Hugging Face

```
zabir1996/mip-bench/
├── Lectures/                ← slide images (.JPG) + aligned narration (.txt), 23 lectures
├── claims_v5/               ← GPT-4o extracted concept graphs, one JSON per slide (1,117 total)
├── igs_scores_gpt4o_v5/     ← GPT-4o judge IGS scores for all five evaluated VLMs
├── igs_scores_claude_v5/    ← Claude (opus-4-6) judge IGS scores (5 models × 183 clean-test slides)
├── vlm_outputs/             ← VLM zero-shot free-form responses for all five models
├── concept_graphs/          ← curriculum knowledge graph (3,712 nodes, 12,419 edges)
├── human_study_v4/          ← human annotation scores (3 annotators, 50 stratified slides)
├── robustness/              ← multi-judge agreement, recall perturbation, judge-bias OLS (3 JSON)
└── splits/
    ├── train.json            ← 724-slide training split
    ├── val.json              ← 182-slide validation split
    ├── test.json             ← 211-slide test split (183 clean after 28 exclusions)
    ├── master_index.json     ← all 1,117 slides with full metadata
    ├── split_info.json       ← split strategy and lecture assignments
    └── excluded_slides.json  ← 28 excluded slides with exclusion reason
```

> **Tip:** To reproduce the main results table without any API calls or GPU, download only
> `igs_scores_gpt4o_v5/` and run `python analysis/step10_gpt4o.py`.

---

## Installation

### Option A — pip

```bash
git clone https://github.com/zabirul-islam/mip-bench.git
cd mip-bench
pip install -r requirements.txt
```

### Option B — conda

```bash
git clone https://github.com/zabirul-islam/mip-bench.git
cd mip-bench
conda env create -f environment.yml
conda activate mip-bench
```

Set your OpenAI API key (required for Steps 1 and 3 only):

```bash
export OPENAI_API_KEY="sk-..."
```

Set your Anthropic API key (required only for Step 3b — Claude judge replication):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Dataset Download

### Download everything

```bash
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset \
    --local-dir ./medlecture_bench
```

### Download only what you need

```bash
# Slide images + narration (for VLM inference)
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset --local-dir ./medlecture_bench \
    --include "Lectures/*"

# Concept graphs only (skip GPT-4o extraction, saves API cost)
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset --local-dir ./medlecture_bench \
    --include "claims_v5/*"

# IGS scores only (reproduce main results — no API key or GPU needed)
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset --local-dir ./medlecture_bench \
    --include "igs_scores_gpt4o_v5/*"

# Claude-judge IGS scores (multi-judge replication)
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset --local-dir ./medlecture_bench \
    --include "igs_scores_claude_v5/*"

# Robustness probe outputs (multi-judge agreement, recall perturbation, judge bias)
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset --local-dir ./medlecture_bench \
    --include "robustness/*"

# Dataset splits only
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset --local-dir ./medlecture_bench \
    --include "splits/*"
```

After downloading, your working directory will look like:

```
medlecture_bench/
├── Lectures/
│   ├── Lecture 1/
│   │   ├── Images/       ← slide images (.JPG)
│   │   └── Texts/        ← aligned instructor narration (.txt)
│   ├── Lecture 2/
│   └── ... Lecture 23/
├── claims_v5/
│   ├── L01_S001.json
│   └── ...
├── igs_scores_gpt4o_v5/
├── igs_scores_claude_v5/
├── vlm_outputs/
├── concept_graphs/
├── human_study_v4/
├── robustness/
└── splits/
    ├── train.json
    ├── val.json
    ├── test.json
    ├── master_index.json
    ├── split_info.json
    └── excluded_slides.json
```

---

## Reproducibility Pipeline

Run steps in order. Steps 0 and 2 do not require an OpenAI API key.

### Step 0 — Build dataset index and splits

```bash
python data/build_dataset_index.py
```

Reads all `Lectures/` image-text pairs, assigns unique IDs (`L01_S001` format), and produces
stratified train/val/test splits by lecture category:

- **Test** (Lectures 1, 3, 4, 9, 21): 211 slides → 183 clean after 28 exclusions
- **Val** (Lectures 5, 7, 15, 19): 182 slides
- **Train** (remaining 17 lectures): 724 slides

*Skip if you downloaded `splits/` from Hugging Face.*

---

### Step 1 — Extract instructional intent with GPT-4o

```bash
# Sanity check: 5 slides
python evaluation/step3_gpt4o.py --split test --validate

# Full extraction: all 1,117 slides
python evaluation/step3_gpt4o.py --split all
```

*Skip if you downloaded `claims_v5/` from Hugging Face.*

Output — one JSON per slide in `claims_v5/`:
```json
{
  "id": "L01_S003",
  "concepts": ["Fourier transform", "frequency domain", "convolution"],
  "relations": [["convolution", "Fourier transform", "equivalent_to_multiplication"]],
  "scope_boundary": "Covers the convolution theorem and its frequency-domain interpretation."
}
```

---

### Step 2 — VLM inference (image-only, zero-shot)

```bash
python evaluation/step4_vlm_inference.py --model all --split test
```

Feeds each slide image to five open-weight VLMs via vLLM with the fixed prompt:

> *"You are a student viewing a medical imaging lecture slide. Explain what this slide is teaching."*

No question, no multiple-choice, no supplementary text — image only. No API key required.

| Model | Parameters |
|---|---|
| Qwen2-VL-7B | 7B |
| InternVL2-8B | 8B |
| InternVL2-26B | 26B |
| InternVL2-40B | 40B |
| LLaVA-1.6-34B | 34B |

Inference times on H100 (183 test slides, vLLM bfloat16, excluding model loading):
Qwen2-VL-7B 51s · InternVL2-8B 1m28s · InternVL2-26B 4m23s · InternVL2-40B 8m26s · LLaVA-1.6-34B 17m36s.
Models ≤26B used `tensor_parallel=1`; LLaVA-1.6-34B and InternVL2-40B used `tensor_parallel=2`.

*Skip if you downloaded `vlm_outputs/` from Hugging Face.*

---

### Step 3 — Score with GPT-4o judge

```bash
python evaluation/step5_gpt4o_judge.py --model all --split test
```

Scores each (slide, response) pair on five-point ordinal scale {0.0, 0.3, 0.5, 0.7, 1.0} for CR, RV, SF.
Full judge prompt is reproduced in Appendix I of the paper.

*Skip if you downloaded `igs_scores_gpt4o_v5/` from Hugging Face.*

---

### Step 3b — Second LLM judge (Claude, non-OpenAI replication)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python evaluation/step5_claude_judge.py --model all --split test --resume
```

Identical rubric and anchor-snapping as Step 3, but served by Anthropic `claude-opus-4-6`.
Writes per-model scores to `igs_scores_claude_v5/`. Used to defend against single-provider
judge bias (paper §4.2, Appendix `sec:supp_multi_judge`).

*Skip if you downloaded `igs_scores_claude_v5/` from Hugging Face.*

---

### Step 4 — Baseline metrics

```bash
python evaluation/step6_v4.py
```

Computes BLEU-1, BLEU-4, ROUGE-L, BERTScore against instructor narration.
All correlate below r = 0.38 with IGS (paper Table 5), confirming standard
similarity metrics are blind to instructional grounding failure.

---

### Step 5 — Main results

```bash
python analysis/step10_gpt4o.py
```

Produces the main IGS results table (paper Table 7) and 95% bootstrap CIs (10,000 resamples).
**No API key or GPU required** if `igs_scores_gpt4o_v5/` is downloaded.

---

### Step 5b — Multi-judge agreement (GPT-4o vs Claude)

```bash
python analysis/step16_multi_judge_agreement.py
```

Reads `igs_scores_gpt4o_v5/` and `igs_scores_claude_v5/` and emits per-model Pearson r,
Spearman ρ, MAD, ensemble IGS with 10,000-resample 95 % CI, and the cross-judge Kendall τ
over VLM rankings. Result: **Kendall τ = 1.0 (p = 0.017)**, r_IGS ∈ [0.694, 0.835], Claude
uniformly ~0.05–0.09 IGS stricter than GPT-4o on level while preserving ordering. Output:
`results/multi_judge_agreement.json`.

---

### Step 6 — Weight ablation (Appendix B)

```bash
python analysis/step11_v4.py
```

Ablates 171 weight combinations. Confirms mean Spearman ρ = 0.998,
top model unchanged in 100% of configurations, full ranking preserved in 97.7%.

---

### Step 7 — Concept graph and centrality analysis (Section 5, Table 6)

```bash
python analysis/step13_v4.py
python analysis/step14_v4.py
```

Builds curriculum knowledge graph and confirms cross-cutting concepts
(Fourier transform, convolution, inner product) are a universal grounding bottleneck —
all five models perform worse on high-centrality concepts.

---

### Step 8 — Human study validation (Section 8, Table 10)

```bash
python evaluation/step7_v4.py
```

Compares GPT-4o judge scores against pooled scores from three independent annotators
on a 50-slide stratified subset. Reproduces:

- Per-annotator Pearson r vs. judge: 0.463–0.652 (Table 9)
- Pooled judge–human Pearson r on overall IGS: **0.729** (Table 10)
- 88% of slides agree within 0.2; near-zero aggregate bias (−0.015)
- Inter-annotator Krippendorff's α decomposition (Appendix F)

Annotation protocol and rubrics are documented in Appendix H of the paper. Annotators were
volunteer Ph.D. students at RPI; no compensation was provided. IRB Protocol #2321,
Exempt under 45 CFR 46.101(b)(3), April 6, 2026.

---

### Step 9 — Recall-perturbation robustness sweep

```bash
python analysis/step15_recall_perturbation.py
```

Simulates 10 / 20 / 30 % gold-concept recall drop on the extraction pipeline and recomputes
IGS and VLM rankings. Result: mean rank Spearman ρ̄ = 0.94, max flip = 1 (neighbour swap).
Rankings are stable under imperfect extraction. Output: `results/recall_perturbation_results.json`.

---

### Step 10 — Judge-family bias ablation

```bash
python analysis/step17_judge_bias_ablation.py
```

OLS regression `IGS ~ β_C · Jaccard(ŷ, 𝓒) + β_N · Jaccard(ŷ, narration)` per model.
Result: Δβ = β_C − β_N spans [−0.69, +0.63]; β_N matches β_C on average — the judge is
**not** preferentially rewarding GPT-4o-extractor vocabulary. Output: `results/judge_bias_ablation.json`.

---

### Step 11 — Pipeline transfer probe

```bash
python evaluation/step18_transfer_probe.py --probe_dir /path/to/non_medical_corpus
```

Runs the extraction pipeline unchanged on a non-medical corpus (e.g., MIT OCW Fourier lecture).
Existence proof only — no IGS is reported, since calibration of CR/RV/SF anchors is
curriculum-specific. Produces `<C, R, B>` JSON per slide under the probe directory.

---

## LoRA Fine-Tuning (Section 7)

Qwen2-VL-7B, LoRA r=16 α=32, AdamW, cosine LR, 3 epochs, 719 training slides.

```bash
# Dry run
python finetuning/step12_eval_claims_v4.py --validate

# Full training
python finetuning/step12_eval_claims_v4.py

# Evaluate fine-tuned model
python finetuning/step12_eval_claims_v4.py --eval-only
python evaluation/step5_gpt4o_judge.py --model qwen2vl_7b_lora_claims --split test
python analysis/step10_gpt4o.py
```

| Configuration | IGS | ΔIGS |
|---|---|---|
| Baseline (Qwen2-VL-7B) | 0.642 | — |
| + Teacher-narration LoRA | 0.637 | −0.007 |
| + Claims-structured LoRA | 0.483 | −0.161 |

For these two LoRA target formulations on this benchmark, supervised imitation does not
improve IGS. We do not claim supervised fine-tuning is categorically ineffective; we view
direct optimization against grounding-aware rewards as a promising future direction
(see Section 7 and Appendix E for full interpretation).

---

## Compute Requirements

| Resource | Details |
|---|---|
| GPU | 2× NVIDIA H100 PCIe (80 GB each) |
| CUDA | 12.8 |
| Inference framework | vLLM v0.11.0, bfloat16 |
| Total inference time (5 models, 183 slides) | ~32 min |
| Total GPU compute (all experiments) | ~2 H100-hours |
| LoRA fine-tuning | ~16 min (teacher), ~18 min (claims), single H100 |
| GPT-4o API cost (extraction + judging) | $10.56 USD |

---

## License

MIP-BENCH is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The underlying lecture materials are the intellectual property of Dr. Ge Wang,
Rensselaer Polytechnic Institute, Troy, NY, and are used with explicit permission
for research and benchmarking purposes only.

---

## Citation

```bibtex
@inproceedings{islam2026mipbench,
  title     = {{MIP-BENCH}: A Benchmark for Instructional Grounding
               in Medical Imaging Presentations},
  author    = {Islam, Md Zabirul and Manik, Md Motaleb Hossen and
               Wang, Jianxu and Liang, Yuxuan and Wang, Ge},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)
               Datasets and Benchmarks Track},
  year      = {2026}
}
```
