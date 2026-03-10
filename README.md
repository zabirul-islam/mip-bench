# MIP-BENCH: Benchmarking Instructional Grounding in Vision-Language Models for Medical Education

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-zabir1996%2Fmip--bench-yellow)](https://huggingface.co/zabir1996/mip-bench/tree/main)
[![Code](https://img.shields.io/badge/GitHub-zabirul--islam%2Fmip--bench-blue)](https://github.com/zabirul-islam/mip-bench)

Official repository for the paper:

> **MIP-BENCH: Are Open-Weight Vision-Language Models Reliable for Instructional Grounding in Medical Imaging Presentations?**  
> Anonymous Author(s) — Submitted to NeurIPS 2024

---

## Quick Links

| Resource | URL |
|---|---|
| Code (this repo) | https://github.com/zabirul-islam/mip-bench |
| Dataset (slides + narration) | https://huggingface.co/zabir1996/mip-bench/tree/main |

---

## Overview

MIP-BENCH is a diagnostic benchmark of **1,117 lecture slides** from a structured medical imaging curriculum (23 lectures), each paired with a structured concept graph encoding instructional intent:

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
mip-bench/
├── data/
│   ├── build_dataset_index.py        # Step 0: Build master index + stratified splits
│   └── splits/
│       ├── master_index.json         # All 1,117 slides with full metadata
│       ├── train.json                # 724 slides
│       ├── val.json                  # 182 slides
│       ├── test.json                 # 211 slides (183 clean after exclusions)
│       └── split_info.json           # Split strategy and category map
├── evaluation/
│   ├── step3_gpt4o.py                # Step 1: GPT-4o claim extraction
│   ├── step4_vlm_inference.py        # Step 2: VLM inference (image-only, zero-shot)
│   ├── step5_gpt4o_judge.py          # Step 3: GPT-4o judge -> IGS scores
│   ├── step6_v4.py                   # Step 4: Baseline metrics
│   └── step7_v4.py                   # Step 5: Human study validation
├── analysis/
│   ├── step10_gpt4o.py               # Main results tables
│   ├── step11_v4.py                  # Weight ablation (171 combinations)
│   ├── step13_v4.py                  # Curriculum concept graph
│   └── step14_v4.py                  # Graph topology + centrality analysis
├── finetuning/
│   └── step12_eval_claims_v4.py      # LoRA fine-tuning + evaluation
├── requirements.txt
├── environment.yml
└── README.md
```

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

Slide images and aligned instructor narration are hosted on Hugging Face:

**https://huggingface.co/zabir1996/mip-bench/tree/main**

### Option A — Hugging Face CLI

```bash
pip install huggingface_hub
huggingface-cli download zabir1996/mip-bench --repo-type dataset --local-dir ./Lectures
```

### Option B — Python

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="zabir1996/mip-bench",
    repo_type="dataset",
    local_dir="./Lectures"
)
```

After downloading, your directory should look like:

```
Lectures/
  Lecture 1/
    Images/    <- slide images (.JPG)
    Texts/     <- aligned instructor narration (.txt)
  Lecture 2/
  ...
  Lecture 23/
```

The `data/splits/` JSON files are already in this repository and do not require downloading raw slides to inspect metadata.

---

## Full Reproducibility Pipeline

### Step 0 — Build dataset index and stratified splits

```bash
python data/build_dataset_index.py
```

Reads all Lectures/ image-text pairs, assigns unique IDs (L01_S001 format), applies quality filtering (excludes slides < 20 words of narration), and produces stratified train/val/test splits by lecture category.

**Outputs:** `medlecture_bench/master_index.json`, `train.json`, `val.json`, `test.json`, `split_info.json`

---

### Step 1 — Extract instructional intent with GPT-4o

```bash
# Sanity check: 5 slides
python evaluation/step3_gpt4o.py --split test --validate

# Full extraction: all 1,117 slides (~$3-5 in API costs)
python evaluation/step3_gpt4o.py --split all
```

**Output:** `claims_v5/` — one JSON per slide with `{concepts, relations, scope_boundary}`

---

### Step 2 — VLM inference (image-only, zero-shot)

```bash
python evaluation/step4_vlm_inference.py --model all --split test
```

Feeds each slide image to five open-weight VLMs via vLLM with the fixed prompt:
*"You are a student viewing a medical imaging lecture slide. Explain what this slide is teaching."*

Does **not** require an OpenAI API key.

| Model | Parameters |
|---|---|
| Qwen2-VL-7B | 7B |
| InternVL2-8B | 8B |
| InternVL2-26B | 26B |
| InternVL2-40B | 40B |
| LLaVA-1.6-34B | 34B |

**Output:** `vlm_responses/` — one JSON per model

---

### Step 3 — Score with GPT-4o judge

```bash
python evaluation/step5_gpt4o_judge.py --model all --split test
```

**Output:** `igs_scores_gpt4o_v5/` — per-slide IGS scores for all models

---

### Step 4 — Baseline metrics

```bash
python evaluation/step6_v4.py   # set CLAIMS_DIR = "claims_v5/" in script
```

---

### Step 5 — Main results tables

```bash
python analysis/step10_gpt4o.py
```

---

### Step 6 — Weight ablation (171 combinations)

```bash
python analysis/step11_v4.py    # set SCORES_DIR = "igs_scores_gpt4o_v5/" in script
```

---

### Step 7 — Concept graph and topology analysis

```bash
python analysis/step13_v4.py    # set CLAIMS_DIR = "claims_v5/" in script
python analysis/step14_v4.py
```

---

### Step 8 — Human study validation

```bash
python evaluation/step7_v4.py
```

---

## LoRA Fine-Tuning Pipeline

```bash
# 1. Dry run
python finetuning/step12_eval_claims_v4.py --validate

# 2. Full training (Qwen2-VL-7B, LoRA r=16 alpha=32, 3 epochs, 719 slides)
python finetuning/step12_eval_claims_v4.py

# 3. Generate fine-tuned outputs
python finetuning/step12_eval_claims_v4.py --eval-only

# 4. Score with GPT-4o judge
python evaluation/step5_gpt4o_judge.py --model qwen2vl_7b_lora_claims --split test

# 5. Final results table
python analysis/step10_gpt4o.py
```

---

## Key Results

| Model | IGS | CR | RV | SF | SF-CR | n |
|---|---|---|---|---|---|---|
| Qwen2-VL-7B | **0.642** | 0.633 | 0.563 | 0.768 | +0.134 | 183 |
| InternVL2-40B | 0.619 | 0.615 | 0.537 | 0.740 | +0.126 | 183 |
| InternVL2-26B | 0.596 | 0.597 | 0.521 | 0.699 | +0.103 | 183 |
| InternVL2-8B | 0.595 | 0.592 | 0.525 | 0.699 | +0.107 | 183 |
| LLaVA-1.6-34B | 0.501 | 0.492 | 0.421 | 0.629 | +0.137 | 183 |
| Human (1 annotator) | 0.578 | 0.604 | 0.532 | 0.602 | -0.002 | 50 |

---

## Compute Requirements

| Resource | Details |
|---|---|
| GPU | 2x NVIDIA H100 PCIe (80 GB each) |
| CUDA | 12.8, driver 570.195.03 |
| Inference | vLLM, bfloat16, 1 GPU per model |
| Total GPU-hours | ~15 H100 GPU-hours |
| LoRA fine-tuning | ~16 min (teacher), ~18 min (claims), single H100 |
| GPT-4o API | ~$3-5 for claim extraction |

---

## License

MIP-BENCH is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).  
The underlying lecture materials are the intellectual property of Dr. Ge Wang, Rensselaer Polytechnic Institute, Troy, NY, and are used with explicit permission for research and benchmarking purposes only.

---

## Citation

```bibtex
@inproceedings{mipbench2024,
  title     = {{MIP-BENCH}: Are Open-Weight Vision-Language Models Reliable
               for Instructional Grounding in Medical Imaging Presentations?},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```
