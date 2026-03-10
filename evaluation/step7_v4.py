#!/usr/bin/env python3
"""
Generate human annotation package for IGS validation study.
Creates:
  1. annotation_guidelines.md  — instructions for annotators
  2. annotation_batch.json     — 50 stratified slides to annotate
  3. annotation_sheet.csv      — spreadsheet for recording scores

Run from: ~/islamm11/MedLecture/
Usage: python step7_human_study.py --sample 50 --seed 42
"""

import json
import csv
import random
import argparse
from pathlib import Path
from collections import defaultdict

SCORES_DIR    = "medlecture_bench/igs_scores_gpt4o_v5"
CLAIMS_DIR    = "medlecture_bench/claims_v5"
VLM_OUT_DIR   = "medlecture_bench/vlm_outputs"
DATASET_DIR   = "medlecture_bench"
LECTURES_ROOT = "Lectures"
OUTPUT_DIR    = "medlecture_bench/human_study_v4"

import re

def load_teacher_text(sid):
    match = re.match(r'L(\d+)_S(\d+)', sid)
    if not match: return ""
    lec, slide = int(match.group(1)), int(match.group(2))
    p = Path(LECTURES_ROOT) / f"Lecture {lec}" / "Texts" / f"Slide{slide}.txt"
    return p.read_text().strip() if p.exists() else ""

def load_image_path(sid):
    match = re.match(r'L(\d+)_S(\d+)', sid)
    if not match: return ""
    lec, slide = int(match.group(1)), int(match.group(2))
    return str(Path(LECTURES_ROOT) / f"Lecture {lec}" / "Images" / f"Slide{slide}.JPG")

def main(n_slides=50, seed=42, reference_model="qwen2vl_7b"):
    random.seed(seed)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test scores for reference model ─────────────────
    score_file = Path(SCORES_DIR) / f"{reference_model}_test_scores.json"
    scores = json.load(open(score_file))

    # Filter: non-zero IGS, has claims, has teacher text
    valid = {}
    for sid, s in scores.items():
        if s["igs"] == 0.0:
            continue
        claims_file = Path(CLAIMS_DIR) / f"{sid}.json"
        if not claims_file.exists():
            continue
        teacher_text = load_teacher_text(sid)
        if len(teacher_text) < 50:
            continue
        vlm_file = Path(VLM_OUT_DIR) / f"{reference_model}_test.json"
        vlm_data = json.load(open(vlm_file))
        if sid not in vlm_data or not vlm_data[sid].get("response", "").strip():
            continue
        valid[sid] = s

    print(f"Valid slides for annotation: {len(valid)}")

    # ── Stratified sampling ────────────────────────────────────
    # Stratify by: content_type AND IGS tercile (low/mid/high)
    strata = defaultdict(list)
    igs_vals = sorted(s["igs"] for s in valid.values())
    lo_thresh = igs_vals[len(igs_vals)//3]
    hi_thresh = igs_vals[2*len(igs_vals)//3]

    for sid, s in valid.items():
        ct = s.get("content_type", "unknown")
        igs = s["igs"]
        tercile = "low" if igs < lo_thresh else ("high" if igs > hi_thresh else "mid")
        strata[f"{ct}_{tercile}"].append(sid)

    # Sample proportionally
    selected = []
    per_stratum = max(1, n_slides // len(strata))
    for stratum, sids in strata.items():
        n = min(per_stratum, len(sids))
        selected.extend(random.sample(sids, n))

    # Top up if needed
    remaining = set(valid.keys()) - set(selected)
    while len(selected) < n_slides and remaining:
        pick = random.choice(list(remaining))
        selected.append(pick)
        remaining.remove(pick)

    selected = selected[:n_slides]
    random.shuffle(selected)

    print(f"Selected {len(selected)} slides for annotation")

    # ── Build annotation batch ────────────────────────────────
    vlm_data = json.load(open(Path(VLM_OUT_DIR) / f"{reference_model}_test.json"))
    
    batch = []
    for i, sid in enumerate(selected, 1):
        s = scores[sid]
        claims = json.load(open(Path(CLAIMS_DIR) / f"{sid}.json"))
        teacher_text = load_teacher_text(sid)
        vlm_response = vlm_data[sid].get("response", "")
        
        batch.append({
            "annotation_id": i,
            "slide_id": sid,
            "image_path": load_image_path(sid),
            "lecture_category": s.get("lecture_category", ""),
            "content_type": s.get("content_type", ""),
            "teacher_text": teacher_text,
            "vlm_response": vlm_response,
            "ground_truth_claims": {
                "key_concepts": claims.get("key_concepts", []),
                "relations": claims.get("relations", []),
                "scope_boundary": claims.get("scope_boundary", ""),
            },
            "model_igs_scores": {
                "igs": s["igs"],
                "cr": s["concept_recall"],
                "rv": s["relational_validity"],
                "sf": s["scope_fidelity"],
            },
            "annotation": {
                "annotator_1": {"cr": None, "rv": None, "sf": None, "notes": ""},
                "annotator_2": {"cr": None, "rv": None, "sf": None, "notes": ""},
                "annotator_3": {"cr": None, "rv": None, "sf": None, "notes": ""},
            }
        })

    # Save batch JSON
    batch_file = out_dir / "annotation_batch.json"
    with open(batch_file, "w") as f:
        json.dump(batch, f, indent=2)
    print(f"✅ Saved annotation batch: {batch_file}")

    # ── Build CSV annotation sheet ────────────────────────────
    csv_file = out_dir / "annotation_sheet.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "annotation_id", "slide_id", "content_type", "lecture_category",
            "model_igs", "model_cr", "model_rv", "model_sf",
            "annotator_1_cr", "annotator_1_rv", "annotator_1_sf", "annotator_1_notes",
            "annotator_2_cr", "annotator_2_rv", "annotator_2_sf", "annotator_2_notes",
            "annotator_3_cr", "annotator_3_rv", "annotator_3_sf", "annotator_3_notes",
        ])
        for item in batch:
            writer.writerow([
                item["annotation_id"], item["slide_id"],
                item["content_type"], item["lecture_category"],
                item["model_igs_scores"]["igs"],
                item["model_igs_scores"]["cr"],
                item["model_igs_scores"]["rv"],
                item["model_igs_scores"]["sf"],
                "", "", "", "",
                "", "", "", "",
                "", "", "", "",
            ])
    print(f"✅ Saved annotation sheet: {csv_file}")

    # ── Write annotation guidelines ───────────────────────────
    guidelines = """# MedLectureBench — Human Annotation Guidelines
## IGS Validation Study

**Purpose:** Validate that the Instructional Grounding Score (IGS) 
computed by an LLM judge aligns with human judgment.  
You will score 50 slides, each taking ~2-3 minutes. Total time: ~2 hours.

---

## Your Task

For each slide, you will see:
1. The **slide image** (open the image_path file)
2. The **teacher's explanation** (what the instructor said about this slide)
3. The **VLM's response** (what the AI model said about the slide)
4. The **key concepts and relations** extracted from the teacher's text

You score the VLM response on **three dimensions**, each from 0 to 1:

---

## Scoring Rubric

### CR — Concept Recall (0.0 / 0.5 / 1.0)
**Question: What fraction of the teacher's key concepts did the VLM correctly address?**

| Score | Meaning | Example |
|-------|---------|---------|
| **1.0** | VLM covers ALL key concepts the teacher intended | Teacher: {Fourier transform, frequency domain, k-space} → VLM addresses all three |
| **0.5** | VLM covers SOME but misses important ones | Teacher: {Fourier transform, frequency domain, k-space} → VLM only covers Fourier transform |
| **0.0** | VLM misses MOST or ALL key concepts | Teacher: {Fourier transform, frequency domain, k-space} → VLM talks about image contrast instead |

**Important:** A concept counts as "covered" if the VLM conveys the same idea, not necessarily the same words.

---

### RV — Relational Validity (0.0 / 0.5 / 1.0)
**Question: Are the relationships between concepts stated correctly?**

| Score | Meaning | Example |
|-------|---------|---------|
| **1.0** | All relations are stated correctly | Teacher: "higher kVp → greater photon penetration" → VLM states this correctly |
| **0.5** | Some relations correct, some wrong or missing | VLM gets one relation right but states another backwards |
| **0.0** | Relations are wrong, reversed, or absent | VLM says "higher kVp → less penetration" (reversed) |

**Note:** If the slide has no explicit relations (e.g., a motivational slide), score 0.5 by default.

---

### SF — Scope Fidelity (0.0 / 0.5 / 1.0)
**Question: Does the VLM stay within what this slide is teaching?**

| Score | Meaning | Example |
|-------|---------|---------|
| **1.0** | VLM explains exactly what this slide teaches, nothing more | Slide teaches X-ray attenuation basics → VLM explains exactly that |
| **0.5** | VLM is mostly on-scope but adds some off-topic content | VLM explains attenuation correctly but also brings in CT Hounsfield units (not taught here) |
| **0.0** | VLM goes significantly off-scope | Slide teaches signal processing but VLM talks extensively about clinical diagnosis |

**Note:** Off-scope content is not the same as wrong content. The VLM might say correct things that simply weren't the point of this slide.

---

## Step-by-Step Process

For each annotation item in `annotation_batch.json`:

1. **Open the slide image** (image_path field)
2. **Read the teacher's explanation** (teacher_text field)  
3. **Read the extracted key concepts and relations** (ground_truth_claims field)
4. **Read the VLM response** (vlm_response field)
5. **Score CR, RV, SF** using the rubric above (use only 0.0, 0.5, or 1.0)
6. **Add a brief note** (1 sentence) if your score differs significantly from model_igs_scores

Record your scores in `annotation_sheet.csv` in your designated column (annotator_1, 2, or 3).

---

## Worked Examples

### Example 1: Strong Instructional Alignment (CR=1.0, RV=1.0, SF=1.0)
**Teacher:** "This slide introduces the Fourier transform. The key idea is that any signal can be decomposed into sinusoidal components. The transform moves us from time domain to frequency domain, which lets us analyze which frequencies are present."

**Key concepts:** {Fourier transform, signal decomposition, time domain, frequency domain}
**Relations:** {signal → sinusoidal components, FT → frequency domain representation}

**VLM:** "This slide explains the Fourier transform, showing how any signal can be broken down into its constituent sine waves. By converting from the time domain to the frequency domain, we can identify which frequencies make up the signal."

**Scores:** CR=1.0 (all 4 concepts covered), RV=1.0 (both relations correct), SF=1.0 (stays exactly on scope)

---

### Example 2: On-Topic but Missing Concepts (CR=0.5, RV=0.5, SF=1.0)
**Teacher:** "This slide shows the relationship between k-space and image space in MRI. k-space is the raw frequency data. Filling k-space and then applying the inverse Fourier transform gives us the MRI image."

**Key concepts:** {k-space, image space, raw frequency data, inverse Fourier transform, MRI image reconstruction}
**Relations:** {k-space → frequency data, inverse FT → MRI image}

**VLM:** "This slide is about MRI data acquisition. k-space stores the MRI data, and processing this data produces the final image."

**Scores:** CR=0.5 (mentions k-space and image but misses IFT and frequency nature), RV=0.5 (one relation implied, other missing), SF=1.0 (stays on MRI topic)

---

### Example 3: Off-Scope Hallucination (CR=0.5, RV=0.5, SF=0.0)
**Teacher:** "This slide introduces X-ray attenuation. Different tissues absorb X-rays differently based on their density. Bone has high attenuation, soft tissue has low attenuation."

**Key concepts:** {X-ray attenuation, tissue density, differential absorption}
**Relations:** {high density → high attenuation}

**VLM:** "This slide covers X-ray imaging principles, including attenuation. It also discusses CT scanning protocols, Hounsfield units for tissue classification, and radiation dose considerations in clinical practice."

**Scores:** CR=0.5 (attenuation mentioned but density/differential not explicit), RV=0.5 (relation implied), SF=0.0 (significant off-scope content about CT, HU, dose)

---

## Important Notes

- **Don't compare to the model's own scores** until after you've finished scoring. The model_igs_scores are there to help us measure agreement, not to guide you.
- **Be consistent:** Apply the same rubric throughout. If in doubt, use 0.5.
- **It's okay to disagree** with the model scores — that's the point of this study.
- **Time yourself:** If a slide takes more than 5 minutes, write a note and move on.

---

## Contact

If you have questions about any slide or the rubric, reach out before scoring that item.
Do not discuss your scores with other annotators until everyone has finished.

**Thank you for your help with this study.**
"""

    guidelines_file = out_dir / "annotation_guidelines.md"
    with open(guidelines_file, "w") as f:
        f.write(guidelines)
    print(f"✅ Saved guidelines: {guidelines_file}")

    # ── Print summary ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  HUMAN STUDY PACKAGE READY")
    print(f"{'='*55}")
    ct_counts = defaultdict(int)
    for item in batch:
        ct_counts[item["content_type"]] += 1
    print(f"\n  Total slides: {len(batch)}")
    print(f"  Content type distribution:")
    for ct, n in sorted(ct_counts.items()):
        print(f"    {ct:<20} n={n}")
    
    igs_vals = [item["model_igs_scores"]["igs"] for item in batch]
    low = sum(1 for v in igs_vals if v < lo_thresh)
    high = sum(1 for v in igs_vals if v > hi_thresh)
    mid = len(igs_vals) - low - high
    print(f"\n  IGS tercile distribution:")
    print(f"    Low IGS  (<{lo_thresh:.2f}): n={low}")
    print(f"    Mid IGS:               n={mid}")
    print(f"    High IGS (>{hi_thresh:.2f}): n={high}")
    
    print(f"\n  Files written to: {out_dir}/")
    print(f"    annotation_batch.json     ← full data for annotators")
    print(f"    annotation_sheet.csv      ← spreadsheet for scores")
    print(f"    annotation_guidelines.md  ← instructions")
    print(f"\n  Share with annotators:")
    print(f"    1. annotation_guidelines.md (read first)")
    print(f"    2. annotation_batch.json (slide data)")
    print(f"    3. annotation_sheet.csv (record scores here)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="qwen2vl_7b")
    args = parser.parse_args()
    main(args.sample, args.seed, args.model)
