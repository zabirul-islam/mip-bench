# MIP-BENCH: Benchmarking Instructional Grounding in Vision-Language Models for Medical Education

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-zabir1996%2Fmip--bench-yellow)](https://huggingface.co/datasets/zabir1996/mip-bench/tree/main)
[![Code](https://img.shields.io/badge/GitHub-zabirul--islam%2Fmip--bench-blue)](https://github.com/zabirul-islam/mip-bench)

Official repository for the paper:

> **MIP-BENCH: Are Open-Weight Vision-Language Models Reliable for Instructional Grounding in Medical Imaging Presentations?**  
> Anonymous Author(s) — Submitted to NeurIPS 2024

---

## Quick Links

| Resource | URL |
|---|---|
| Code (this repo) | https://github.com/zabirul-islam/mip-bench |
| Dataset + Annotations (Hugging Face) | https://huggingface.co/datasets/zabir1996/mip-bench/tree/main |

---

## Overview

MIP-BENCH is a diagnostic benchmark of **1,117 lecture slides** from a structured medical imaging
curriculum (23 lectures), each paired with a structured concept graph encoding instructional intent:

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

## Repository Structure

```
mip-bench/                            <- this GitHub repo (code only)
├── data/
│   ├── build_dataset_index.py        # Step 0: Build master index + stratified splits
│   └── splits/                       # mirrors the JSON files on HuggingFace
│       ├── master_index.json
│       ├── train.json
│       ├── val.json
│       ├── test.json
│       └── split_info.json
├── evaluation/
│   ├── step3_gpt4o.py                # Step 1: GPT-4o claim extraction
│   ├── step4_vlm_inference.py        # Step 2: VLM inference (image-only, zero-shot)
│   ├── step5_gpt4o_judge.py          # Step 3: GPT-4o judge -> IGS scores
│   ├── step6_v4.py                   # Step 4: Baseline metrics
│   └── step7_v4.py                   # Step 5: Human study validation
├── analysis/
│   ├── step10_gpt4o.py               # Main results tables
│   ├── step11_v4.py                  # Weight ablation (171 combinations)
│   ├── step13_v4.py                  # Curriculum concept graph construction
│   └── step14_v4.py                  # Graph topology + centrality analysis
├── finetuning/
│   └── step12_eval_claims_v4.py      # LoRA fine-tuning + evaluation
├── requirements.txt
├── environment.yml
└── README.md
```

---

## What Is on Hugging Face

All data, annotations, and precomputed results are hosted at:  
**https://huggingface.co/datasets/zabir1996/mip-bench/tree/main**

| Directory / File | Contents | Required for |
|---|---|---|
| `Lectures/` | Slide images (.JPG) + aligned instructor narration (.txt) | Steps 0, 1, 2 |
| `claims_v5/` | GPT-4o extracted concept graphs — C, R, B per slide (1,117 JSONs) | Steps 3, 4, 6, 7, 8 |
| `igs_scores_gpt4o_v5/` | GPT-4o judge IGS scores for all five models | Steps 5, 6, 7, 8 |
| `vlm_outputs/` | VLM zero-shot free-form responses for all five models | Step 3 |
| `concept_graphs/` | Curriculum knowledge graph (3,712 nodes, 12,419 edges) | Step 7 |
| `human_study_v4/` | Human annotation scores (50 slides, 1 annotator) | Step 8 |
| `master_index.json` | All 1,117 slides with full metadata | Step 0 |
| `train.json` | 724-slide training split | Steps 1, 2 |
| `val.json` | 182-slide validation split | Steps 1, 2 |
| `test.json` | 211-slide test split (183 clean after exclusions) | All eval steps |
| `split_info.json` | Split strategy, lecture assignments, category map | Reference |
| `excluded_slides.json` | 28 excluded slides with exclusion reason | Reference |

> **Tip:** To reproduce Table 9 (main results) without any API calls or GPU,
> download only `igs_scores_gpt4o_v5/` and run `python analysis/step10_gpt4o.py`.

---

## Installation

### Option 1 — pip

```bash
git clone https://github.com/zabirul-islam/mip-bench.git
cd mip-bench
pip install -r requirements.txt
```

### Option 2 — conda

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
pip install huggingface_hub
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset \
    --local-dir ./medlecture_bench
```

### Download only what you need

```bash
# Slides and narration only (for running VLM inference)
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset \
    --local-dir ./medlecture_bench \
    --include "Lectures/*"

# Precomputed claims only (skip GPT-4o extraction, saves ~$5)
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset \
    --local-dir ./medlecture_bench \
    --include "claims_v5/*"

# Precomputed IGS scores only (reproduce Table 9 with no API or GPU)
huggingface-cli download zabir1996/mip-bench \
    --repo-type dataset \
    --local-dir ./medlecture_bench \
    --include "igs_scores_gpt4o_v5/*"
```

After downloading, your working directory should look like:

```
medlecture_bench/
  Lectures/
    Lecture 1/
      Images/    <- slide images (.JPG)
      Texts/     <- aligned instructor narration (.txt)
    Lecture 2/
    ...
    Lecture 23/
  claims_v5/
    L01_S001.json
    ...
  igs_scores_gpt4o_v5/
    qwen2vl_7b.json
    ...
  vlm_outputs/
  concept_graphs/
  human_study_v4/
  master_index.json
  train.json
  val.json
  test.json
  split_info.json
  excluded_slides.json
```

---

## Full Reproducibility Pipeline

Run steps in order. Steps 0 and 2 do not require an OpenAI API key.

### Step 0 — Build dataset index and stratified splits

```bash
python data/build_dataset_index.py
```

Reads all `Lectures/` image-text pairs, assigns unique IDs (`L01_S001` format),
applies quality filtering (excludes slides < 20 words of narration), and produces
stratified train/val/test splits by lecture category:

- **Test** (Lectures 1, 3, 4, 9, 21): 211 slides → 183 clean after 28 exclusions
- **Val** (Lectures 5, 7, 15, 19): 182 slides
- **Train** (all remaining): 724 slides

*Skip this step if you downloaded the split JSONs from Hugging Face.*

---

### Step 1 — Extract instructional intent with GPT-4o

```bash
# Sanity check: 5 slides (~$0.10)
python evaluation/step3_gpt4o.py --split test --validate

# Full extraction: all 1,117 slides (~$3-5)
python evaluation/step3_gpt4o.py --split all
```

*Skip this step if you downloaded `claims_v5/` from Hugging Face.*

**Output:** `medlecture_bench/claims_v5/` — one JSON per slide:
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

No question, no multiple-choice, no supplementary text — image only. Does **not** require an API key.

| Model | Parameters |
|---|---|
| Qwen2-VL-7B | 7B |
| InternVL2-8B | 8B |
| InternVL2-26B | 26B |
| InternVL2-40B | 40B |
| LLaVA-1.6-34B | 34B |

Approximate wall-clock times on a single H100 (183 test slides):
Qwen2-VL-7B ≈2h12m · InternVL2-8B ≈8m · InternVL2-26B ≈2h56m · LLaVA-1.6-34B ≈39m · InternVL2-40B ≈2h30m

*Skip this step if you downloaded `vlm_outputs/` from Hugging Face.*

---

### Step 3 — Score with GPT-4o judge

```bash
python evaluation/step5_gpt4o_judge.py --model all --split test
```

Scores each (slide, VLM response) pair on five-point ordinal scale {0.0, 0.3, 0.5, 0.7, 1.0}
for CR, RV, SF. Full judge prompt is in Appendix C of the paper.

*Skip this step if you downloaded `igs_scores_gpt4o_v5/` from Hugging Face.*

---

### Step 4 — Baseline metrics

```bash
python evaluation/step6_v4.py   # set CLAIMS_DIR = "medlecture_bench/claims_v5/"
```

Computes BLEU-1, BLEU-4, ROUGE-L, BERTScore vs. instructor narration.
All fall below Pearson r = 0.38 with IGS (Table 6).

---

### Step 5 — Main results tables

```bash
python analysis/step10_gpt4o.py  # set SCORES_DIR = "medlecture_bench/igs_scores_gpt4o_v5/"
```

Produces Table 9 (main IGS results) and 95% bootstrap CIs (10,000 resamples).
**This step requires no API key or GPU if scores are downloaded from Hugging Face.**

---

### Step 6 — Weight ablation (171 combinations)

```bash
python analysis/step11_v4.py    # set SCORES_DIR = "medlecture_bench/igs_scores_gpt4o_v5/"
```

Confirms mean Spearman ρ = 0.998, top model unchanged in 100% of 171 configurations.

---

### Step 7 — Concept graph and topology analysis

```bash
python analysis/step13_v4.py    # set CLAIMS_DIR = "medlecture_bench/claims_v5/"
python analysis/step14_v4.py
```

Builds curriculum knowledge graph (3,712 nodes, 12,419 edges) and confirms
cross-cutting concepts are universal grounding bottleneck (Table 8).

---

### Step 8 — Human study validation

```bash
python evaluation/step7_v4.py
```

Compares human-assigned vs. GPT-4o judge scores on 50 stratified slides (Table 13).

---

## LoRA Fine-Tuning Pipeline

Qwen2-VL-7B, LoRA r=16 α=32, AdamW, cosine LR, 3 epochs, 719 training slides.

```bash
# 1. Dry run
python finetuning/step12_eval_claims_v4.py --validate

# 2. Full training
python finetuning/step12_eval_claims_v4.py

# 3. Generate fine-tuned outputs
python finetuning/step12_eval_claims_v4.py --eval-only

# 4. Score with GPT-4o judge
python evaluation/step5_gpt4o_judge.py --model qwen2vl_7b_lora_claims --split test

# 5. Final results table
python analysis/step10_gpt4o.py
```

| Configuration | IGS | ΔIGS |
|---|---|---|
| Baseline (Qwen2-VL-7B) | 0.642 | — |
| + Teacher-narration LoRA | 0.637 | −0.007 |
| + Claims-structured LoRA | 0.483 | −0.161 |

---

## Key Results

| Model | IGS | CR | RV | SF | SF-CR | n |
|---|---|---|---|---|---|---|
| Qwen2-VL-7B | **0.642** | 0.633 | 0.563 | 0.768 | +0.134 | 183 |
| InternVL2-40B | 0.619 | 0.615 | 0.537 | 0.740 | +0.126 | 183 |
| InternVL2-26B | 0.596 | 0.597 | 0.521 | 0.699 | +0.103 | 183 |
| InternVL2-8B | 0.595 | 0.592 | 0.525 | 0.699 | +0.107 | 183 |
| LLaVA-1.6-34B | 0.501 | 0.492 | 0.421 | 0.629 | +0.137 | 183 |
| Human (1 annotator) | 0.578 | 0.604 | 0.532 | 0.602 | −0.002 | 50 |

---

## Compute Requirements

| Resource | Details |
|---|---|
| GPU | 2× NVIDIA H100 PCIe (80 GB each) |
| CUDA | 12.8, driver 570.195.03 |
| Inference | vLLM, bfloat16, 1 GPU per model |
| Total GPU-hours | ~15 H100 GPU-hours |
| LoRA fine-tuning | ~16 min (teacher), ~18 min (claims), single H100 |
| GPT-4o API | ~$3–5 for claim extraction |

---

## License

MIP-BENCH is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).  
The underlying lecture materials are the intellectual property of Dr. Ge Wang,
Rensselaer Polytechnic Institute, Troy, NY, and are used with explicit permission
for research and benchmarking purposes only.

---

## Citation

```bibtex
@inproceedings{mipbench2026,
  title     = {{MIP-BENCH}: Are Open-Weight Vision-Language Models Reliable
               for Instructional Grounding in Medical Imaging Presentations?},
  author    = {Md Zabirul Islam, Md Motaleb Hossen Manik, Ge Wang},
  booktitle = {},
  year      = {2026}
}
```
