# MIP-BENCH: Benchmarking Instructional Grounding in Vision-Language Models for Medical Education

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-zabir1996%2Fmip--bench-yellow)](https://huggingface.co/datasets/zabir1996/mip-bench)
[![Code](https://img.shields.io/badge/GitHub-zabirul--islam%2Fmip--bench-blue)](https://github.com/zabirul-islam/mip-bench)

Official repository for the paper:

> **MIP-BENCH: Are Open-Weight Vision-Language Models Reliable for Instructional Grounding in Medical Imaging Presentations?**  
> Md Zabirul Islam, Md Motaleb Hossen Manik, Ge Wang — Submitted to NeurIPS 2026

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

---

## Key Results

| Model | IGS | CR | RV | SF | SF−CR | n |
|---|---|---|---|---|---|---|
| Qwen2-VL-7B | **0.642** | 0.633 | 0.563 | 0.768 | +0.134 | 183 |
| InternVL2-40B | 0.619 | 0.615 | 0.537 | 0.740 | +0.126 | 183 |
| InternVL2-26B | 0.596 | 0.597 | 0.521 | 0.699 | +0.103 | 183 |
| InternVL2-8B | 0.595 | 0.592 | 0.525 | 0.699 | +0.107 | 183 |
| LLaVA-1.6-34B | 0.501 | 0.492 | 0.421 | 0.629 | +0.137 | 183 |
| Human (1 annotator) | 0.578 | 0.604 | 0.532 | 0.602 | −0.002 | 50 |

No model exceeds IGS = 0.642. Every model exhibits a universal **scope–concept gap**: SF consistently exceeds CR by +0.10–+0.14, revealing implicit topic classification rather than explicit concept grounding.

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
│   ├── step6_v4.py                     # Step 4: baseline metrics (BLEU, ROUGE, BERTScore)
│   └── step7_v4.py                     # Step 5: human study validation
├── analysis/
│   ├── step10_gpt4o.py                 # main results tables (Table 9)
│   ├── step11_v4.py                    # weight ablation (171 combinations)
│   ├── step13_v4.py                    # curriculum concept graph construction
│   └── step14_v4.py                    # graph topology + centrality analysis
├── finetuning/
│   └── step12_eval_claims_v4.py        # LoRA fine-tuning + evaluation
├── requirements.txt
├── environment.yml
└── README.md
```

> **Data lives on Hugging Face**, not in this repo. See the [dataset page](https://huggingface.co/datasets/zabir1996/mip-bench) for all slide images, concept graphs, IGS scores, and splits.

---

## What Is on Hugging Face

```
zabir1996/mip-bench/
├── Lectures/                ← slide images (.JPG) + aligned narration (.txt), 23 lectures
├── claims_v5/               ← GPT-4o extracted concept graphs, one JSON per slide (1,117 total)
├── igs_scores_gpt4o_v5/     ← GPT-4o judge IGS scores for all five evaluated VLMs
├── vlm_outputs/             ← VLM zero-shot free-form responses for all five models
├── concept_graphs/          ← curriculum knowledge graph (3,712 nodes, 12,419 edges)
├── human_study_v4/          ← human annotation scores (50 slides, single annotator)
└── splits/
    ├── train.json            ← 724-slide training split
    ├── val.json              ← 182-slide validation split
    ├── test.json             ← 211-slide test split (183 clean after 28 exclusions)
    ├── master_index.json     ← all 1,117 slides with full metadata
    ├── split_info.json       ← split strategy and lecture assignments
    └── excluded_slides.json  ← 28 excluded slides with exclusion reason
```

> **Tip:** To reproduce Table 9 without any API calls or GPU, download only
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

# Concept graphs only (skip GPT-4o extraction, saves ~$5)
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset --local-dir ./medlecture_bench \
    --include "claims_v5/*"

# IGS scores only (reproduce Table 9 — no API key or GPU needed)
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset --local-dir ./medlecture_bench \
    --include "igs_scores_gpt4o_v5/*"

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
├── vlm_outputs/
├── concept_graphs/
├── human_study_v4/
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
# Sanity check: 5 slides (~$0.10)
python evaluation/step3_gpt4o.py --split test --validate

# Full extraction: all 1,117 slides (~$3–5)
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

Wall-clock times on a single H100 (183 test slides):
Qwen2-VL-7B ≈2h12m · InternVL2-8B ≈8m · InternVL2-26B ≈2h56m · LLaVA-1.6-34B ≈39m · InternVL2-40B ≈2h30m

*Skip if you downloaded `vlm_outputs/` from Hugging Face.*

---

### Step 3 — Score with GPT-4o judge

```bash
python evaluation/step5_gpt4o_judge.py --model all --split test
```

Scores each (slide, response) pair on five-point ordinal scale {0.0, 0.3, 0.5, 0.7, 1.0} for CR, RV, SF.
Full judge prompt is reproduced in Appendix C of the paper.

*Skip if you downloaded `igs_scores_gpt4o_v5/` from Hugging Face.*

---

### Step 4 — Baseline metrics

```bash
python evaluation/step6_v4.py
```

Computes BLEU-1, BLEU-4, ROUGE-L, BERTScore against instructor narration.
All correlate below r = 0.38 with IGS (paper Table 6).

---

### Step 5 — Main results (Table 9)

```bash
python analysis/step10_gpt4o.py
```

Produces Table 9 (main IGS results) and 95% bootstrap CIs (10,000 resamples).
**No API key or GPU required** if `igs_scores_gpt4o_v5/` is downloaded.

---

### Step 6 — Weight ablation (§2.4)

```bash
python analysis/step11_v4.py
```

Ablates 171 weight combinations. Confirms mean Spearman ρ = 0.998,
top model unchanged in 100% of configurations.

---

### Step 7 — Concept graph and centrality analysis (Table 8)

```bash
python analysis/step13_v4.py
python analysis/step14_v4.py
```

Builds curriculum knowledge graph and confirms cross-cutting concepts
(Fourier transform, convolution) are a universal grounding bottleneck.

---

### Step 8 — Human study validation (Table 13)

```bash
python evaluation/step7_v4.py
```

Compares GPT-4o judge scores against human-assigned scores on 50 stratified slides.

---

## LoRA Fine-Tuning

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

Both fine-tuning strategies degrade IGS, motivating IGS-rewarded reinforcement learning.

---

## Compute Requirements

| Resource | Details |
|---|---|
| GPU | 2× NVIDIA H100 PCIe (80 GB each) |
| CUDA | 12.8, driver 570.195.03 |
| Inference framework | vLLM, bfloat16, 1 GPU per model |
| Total GPU-hours | ~15 H100 GPU-hours (all experiments) |
| LoRA fine-tuning | ~16 min (teacher), ~18 min (claims), single H100 |
| GPT-4o API cost | ~$3–5 for claim extraction over 1,117 slides |

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
  title     = {{MIP-BENCH}: Are Open-Weight Vision-Language Models Reliable
               for Instructional Grounding in Medical Imaging Presentations?},
  author    = {Islam, Md Zabirul and Manik, Md Motaleb Hossen and Wang, Ge},
  booktitle = {},
  year      = {2026}
}
```
