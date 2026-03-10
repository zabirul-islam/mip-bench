# File: build_dataset_index.py
# Purpose: Create master JSON index of all slides
# Updated: Correct topic categories from course schedule
# Course: Undergraduate Medical Imaging Engineering

import os
import json
import random
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURE THESE
# ─────────────────────────────────────────────
LECTURES_ROOT = "Lectures"
OUTPUT_DIR = "./medlecture_bench"
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# GROUND TRUTH TOPIC MAP
# Source: Course schedule slide (verified manually)
# ─────────────────────────────────────────────
LECTURE_TOPIC_MAP = {
    1:  {
        "topic": "Introduction",
        "category": "fundamentals",
        "subcategory": "course_overview",
        "has_clinical_images": False,
        "modality": "none",
        "content_type": "conceptual",
        "description": "Course introduction and medical imaging overview"
    },
    2:  {
        "topic": "MatLab I - Basics",
        "category": "computational",
        "subcategory": "matlab",
        "has_clinical_images": False,
        "modality": "none",
        "content_type": "computational",
        "description": "MATLAB programming basics for imaging"
    },
    3:  {
        "topic": "System",
        "category": "signal_theory",
        "subcategory": "linear_systems",
        "has_clinical_images": False,
        "modality": "none",
        "content_type": "mathematical",
        "description": "Linear systems theory fundamentals"
    },
    4:  {
        "topic": "Convolution",
        "category": "mathematical_foundation",
        "subcategory": "convolution",
        "has_clinical_images": False,
        "modality": "none",
        "content_type": "mathematical",
        "description": "Convolution theory and applications"
    },
    5:  {
        "topic": "Fourier Series",
        "category": "mathematical_foundation",
        "subcategory": "fourier",
        "has_clinical_images": False,
        "modality": "none",
        "content_type": "mathematical",
        "description": "Fourier series decomposition and analysis"
    },
    6:  {
        "topic": "Fourier Transform",
        "category": "mathematical_foundation",
        "subcategory": "fourier",
        "has_clinical_images": False,
        "modality": "none",
        "content_type": "mathematical",
        "description": "Continuous Fourier transform theory"
    },
    7:  {
        "topic": "Signal Processing",
        "category": "signal_theory",
        "subcategory": "signal_processing",
        "has_clinical_images": False,
        "modality": "none",
        "content_type": "mathematical",
        "description": "Signal processing fundamentals"
    },
    8:  {
        "topic": "Discrete FT and FFT",
        "category": "mathematical_foundation",
        "subcategory": "fourier",
        "has_clinical_images": False,
        "modality": "none",
        "content_type": "mathematical",
        "description": "Discrete Fourier transform and fast FFT algorithms"
    },
    9:  {
        "topic": "MatLab II - Homework",
        "category": "computational",
        "subcategory": "matlab",
        "has_clinical_images": False,
        "modality": "none",
        "content_type": "computational",
        "description": "MATLAB signal processing exercises"
    },
    10: {
        "topic": "Network",
        "category": "systems_theory",
        "subcategory": "network_theory",
        "has_clinical_images": False,
        "modality": "none",
        "content_type": "mathematical",
        "description": "Network and systems theory for imaging"
    },
    11: {
        "topic": "Quality and Performance",
        "category": "image_quality",
        "subcategory": "quality_metrics",
        "has_clinical_images": True,
        "modality": "mixed",
        "content_type": "mixed",
        "description": "Image quality metrics SNR MTF NPS and performance"
    },
    12: {
        "topic": "X-ray and Radiography",
        "category": "imaging_modality",
        "subcategory": "xray",
        "has_clinical_images": True,
        "modality": "xray",
        "content_type": "mixed",
        "description": "X-ray physics attenuation and radiography systems"
    },
    13: {
        "topic": "CT Reconstruction",
        "category": "imaging_modality",
        "subcategory": "ct",
        "has_clinical_images": True,
        "modality": "ct",
        "content_type": "mixed",
        "description": "CT image reconstruction algorithms backprojection"
    },
    14: {
        "topic": "CT Scanner",
        "category": "imaging_modality",
        "subcategory": "ct",
        "has_clinical_images": True,
        "modality": "ct",
        "content_type": "mixed",
        "description": "CT scanner hardware geometry and acquisition"
    },
    15: {
        "topic": "MatLab III - CT",
        "category": "computational",
        "subcategory": "matlab",
        "has_clinical_images": True,
        "modality": "ct",
        "content_type": "computational",
        "description": "MATLAB CT reconstruction implementation exercises"
    },
    16: {
        "topic": "Nuclear Physics",
        "category": "physics_foundation",
        "subcategory": "nuclear_physics",
        "has_clinical_images": False,
        "modality": "nuclear",
        "content_type": "conceptual",
        "description": "Nuclear physics radioactive decay and interactions"
    },
    17: {
        "topic": "PET and SPECT",
        "category": "imaging_modality",
        "subcategory": "nuclear_imaging",
        "has_clinical_images": True,
        "modality": "nuclear",
        "content_type": "mixed",
        "description": "PET and SPECT imaging systems and reconstruction"
    },
    18: {
        "topic": "MRI I",
        "category": "imaging_modality",
        "subcategory": "mri",
        "has_clinical_images": True,
        "modality": "mri",
        "content_type": "mixed",
        "description": "MRI physics spin dynamics and signal generation"
    },
    19: {
        "topic": "MRI II",
        "category": "imaging_modality",
        "subcategory": "mri",
        "has_clinical_images": True,
        "modality": "mri",
        "content_type": "mixed",
        "description": "MRI pulse sequences contrast mechanisms and k-space"
    },
    20: {
        "topic": "MRI III",
        "category": "imaging_modality",
        "subcategory": "mri",
        "has_clinical_images": True,
        "modality": "mri",
        "content_type": "mixed",
        "description": "Advanced MRI topics artifacts and applications"
    },
    21: {
        "topic": "Ultrasound I",
        "category": "imaging_modality",
        "subcategory": "ultrasound",
        "has_clinical_images": True,
        "modality": "ultrasound",
        "content_type": "mixed",
        "description": "Ultrasound physics transducers and wave propagation"
    },
    22: {
        "topic": "Ultrasound II",
        "category": "imaging_modality",
        "subcategory": "ultrasound",
        "has_clinical_images": True,
        "modality": "ultrasound",
        "content_type": "mixed",
        "description": "Ultrasound imaging modes beamforming and applications"
    },
    23: {
        "topic": "Optical Imaging",
        "category": "imaging_modality",
        "subcategory": "optical",
        "has_clinical_images": False,
        "modality": "optical",
        "content_type": "mixed",
        "description": "Optical imaging methods microscopy and endoscopy"
    }
}

# ─────────────────────────────────────────────
# STRATIFIED SPLIT STRATEGY
# Ensures test/val cover ALL category types
# ─────────────────────────────────────────────
# Groups by category for stratified splitting
CATEGORY_GROUPS = {
    "fundamentals":           [1],
    "computational":          [2, 9, 15],
    "signal_theory":          [3, 7],
    "mathematical_foundation":[4, 5, 6, 8],
    "systems_theory":         [10],
    "image_quality":          [11],
    "physics_foundation":     [16],
    "imaging_modality": {
        "xray":           [12],
        "ct":             [13, 14],
        "nuclear_imaging":[17],
        "mri":            [18, 19, 20],
        "ultrasound":     [21, 22],
        "optical":        [23]
    }
}


STRATIFIED_SPLITS = {
    # TEST: fundamentals + math + signal_theory + imaging + computational
    # L1=intro(fundamentals), L3=system(signal_theory),
    # L4=convolution(math), L9=matlab II(computational, only 29 slides)
    # L21=ultrasound(imaging)
    "test": [1, 3, 4, 9, 21],

    # VAL: math + signal_theory + imaging + computational
    # L5=fourier(math), L7=signal_processing(signal_theory),
    # L15=matlab III CT(computational, only 30 slides — small cost)
    # L19=MRI II(imaging)
    "val": [5, 7, 15, 19],

    # TRAIN: everything else
    "train": [2, 6, 8, 10, 11, 12, 13, 14,
              16, 17, 18, 20, 22, 23]
}

def assess_text_quality(text):
    """
    Quality assessment of teacher text.
    Returns quality flags and statistics.
    """
    word_count = len(text.split())
    sentence_count = len([
        s for s in text.split('.') if s.strip()
    ])

    quality = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "is_too_short": word_count < 20,
        "is_very_short": word_count < 50,
        "is_short": word_count < 100,
        "is_adequate": 100 <= word_count < 200,
        "is_long": word_count >= 200,
        "is_very_long": word_count >= 400,
        "quality_flag": "good"
    }

    if word_count < 20:
        quality["quality_flag"] = "too_short_exclude"
    elif word_count < 50:
        quality["quality_flag"] = "short_review"
    else:
        quality["quality_flag"] = "good"

    return quality


def assign_split(lecture_number):
    """
    Assign split based on stratified split design.
    Uses fixed assignment to ensure reproducibility
    and category balance.
    """
    for split_name, lectures in STRATIFIED_SPLITS.items():
        if lecture_number in lectures:
            return split_name
    return "train"  # fallback


def build_master_index(root_path, output_dir):
    """
    Build complete master index of all slides.
    Each entry contains full metadata including
    correct topic categories from course schedule.
    """

    root = Path(root_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    master_index = []
    slide_id = 0

    # Get lecture folders sorted numerically
    lecture_folders = sorted([
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("Lecture")
    ], key=lambda x: int(''.join(filter(str.isdigit, x.name))))

    print(f"Building master index from {len(lecture_folders)} "
          f"lectures...")
    print(f"Using stratified splits for category balance\n")

    for lecture_dir in lecture_folders:
        lecture_name = lecture_dir.name
        lecture_num = int(''.join(
            filter(str.isdigit, lecture_name)
        ))

        images_dir = lecture_dir / "Images"
        texts_dir = lecture_dir / "Texts"

        if not images_dir.exists() or not texts_dir.exists():
            print(f"  Skipping {lecture_name}: "
                  f"missing Images or Texts folder")
            continue

        # Get topic info from ground truth map
        topic_info = LECTURE_TOPIC_MAP.get(lecture_num, {
            "topic": "Unknown",
            "category": "unknown",
            "subcategory": "unknown",
            "has_clinical_images": False,
            "modality": "unknown",
            "content_type": "unknown",
            "description": "No description available"
        })

        # Get paired slides only
        image_files = {
            f.stem: f for f in images_dir.iterdir()
            if f.suffix.upper() in ['.JPG', '.JPEG', '.PNG']
        }
        text_files = {
            f.stem: f for f in texts_dir.iterdir()
            if f.suffix == '.txt'
        }

        paired = set(image_files.keys()) & set(text_files.keys())

        # Sort slides numerically
        sorted_slides = sorted(
            paired,
            key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
        )

        for slide_name in sorted_slides:
            slide_num = int(
                ''.join(filter(str.isdigit, slide_name)) or 0
            )

            # Read teacher text
            text_content = text_files[slide_name].read_text(
                encoding='utf-8', errors='ignore'
            ).strip()

            # Assess text quality
            quality = assess_text_quality(text_content)

            # Create unique ID — zero padded for clean sorting
            unique_id = f"L{lecture_num:02d}_S{slide_num:03d}"

            # Assign split
            split = assign_split(lecture_num)

            # Build complete entry
            entry = {
                # ── Identifiers ──────────────────────────
                "id": unique_id,
                "slide_id": slide_id,
                "lecture_name": lecture_name,
                "lecture_number": lecture_num,
                "slide_name": slide_name,
                "slide_number": slide_num,

                # ── File Paths (relative to LECTURES_ROOT) ──
                "image_path": str(
                    image_files[slide_name].relative_to(root)
                ),
                "text_path": str(
                    text_files[slide_name].relative_to(root)
                ),

                # ── Content ───────────────────────────────
                "teacher_text": text_content,

                # ── Topic Metadata (from course schedule) ─
                "lecture_topic": topic_info["topic"],
                "lecture_category": topic_info["category"],
                "lecture_subcategory": topic_info["subcategory"],
                "lecture_description": topic_info["description"],
                "modality": topic_info["modality"],
                "has_clinical_images": topic_info["has_clinical_images"],
                "content_type": topic_info["content_type"],

                # ── Quality ───────────────────────────────
                "quality": quality,

                # ── Split Assignment ──────────────────────
                "split": split,

                # ── Position in Lecture ───────────────────
                "position_in_lecture": sorted_slides.index(
                    slide_name
                ) + 1,
                "total_slides_in_lecture": len(sorted_slides),
                "lecture_position_normalized": round(
                    (sorted_slides.index(slide_name) + 1)
                    / len(sorted_slides), 4
                )
            }

            master_index.append(entry)
            slide_id += 1

        print(f"  {lecture_name:<12} | "
              f"Topic: {topic_info['topic']:<30} | "
              f"Category: {topic_info['category']:<25} | "
              f"Split: {assign_split(lecture_num):<5} | "
              f"Slides: {len(sorted_slides)}")

    print(f"\nTotal slides indexed: {len(master_index)}")
    return master_index


def print_split_summary(master_index):
    """
    Print detailed split summary with category breakdown.
    Verifies stratified balance across splits.
    """
    print(f"\n{'='*70}")
    print(f"SPLIT SUMMARY")
    print(f"{'='*70}")

    for split in ["train", "val", "test"]:
        split_entries = [
            e for e in master_index if e["split"] == split
        ]

        # Category breakdown
        cat_counts = defaultdict(int)
        content_counts = defaultdict(int)
        modality_counts = defaultdict(int)
        lecture_list = sorted(set(
            e["lecture_number"] for e in split_entries
        ))

        for e in split_entries:
            cat_counts[e["lecture_category"]] += 1
            content_counts[e["content_type"]] += 1
            modality_counts[e["modality"]] += 1

        print(f"\n{split.upper()} SPLIT "
              f"({len(split_entries)} slides, "
              f"Lectures: {lecture_list})")
        print(f"  {'Category':<30} {'Count':>6}")
        print(f"  {'-'*38}")
        for cat, count in sorted(
            cat_counts.items(), key=lambda x: -x[1]
        ):
            pct = count / len(split_entries) * 100
            print(f"  {cat:<30} {count:>4} ({pct:5.1f}%)")

        print(f"\n  Content Types:")
        for ct, count in sorted(
            content_counts.items(), key=lambda x: -x[1]
        ):
            pct = count / len(split_entries) * 100
            print(f"    {ct:<20} {count:>4} ({pct:5.1f}%)")


def print_overall_statistics(master_index):
    """
    Print complete dataset statistics for the paper.
    These numbers go directly into your paper's
    dataset description section.
    """
    print(f"\n{'='*70}")
    print(f"DATASET STATISTICS FOR PAPER")
    print(f"{'='*70}")

    total = len(master_index)
    word_counts = [
        e["quality"]["word_count"] for e in master_index
    ]

    print(f"\nOverall:")
    print(f"  Total slides:      {total}")
    print(f"  Total lectures:    23")
    print(f"  Total words (all): {sum(word_counts):,}")
    print(f"\nText Length:")
    print(f"  Min:               {min(word_counts)} words")
    print(f"  Max:               {max(word_counts)} words")
    print(f"  Mean:              {sum(word_counts)/total:.1f} words")
    sorted_wc = sorted(word_counts)
    print(f"  Median:            {sorted_wc[total//2]} words")

    print(f"\nBy Category:")
    cat_counts = defaultdict(int)
    for e in master_index:
        cat_counts[e["lecture_category"]] += 1
    for cat, count in sorted(
        cat_counts.items(), key=lambda x: -x[1]
    ):
        pct = count / total * 100
        print(f"  {cat:<30} {count:>4} ({pct:5.1f}%)")

    print(f"\nBy Content Type:")
    ct_counts = defaultdict(int)
    for e in master_index:
        ct_counts[e["content_type"]] += 1
    for ct, count in sorted(
        ct_counts.items(), key=lambda x: -x[1]
    ):
        pct = count / total * 100
        print(f"  {ct:<20} {count:>4} ({pct:5.1f}%)")

    print(f"\nBy Imaging Modality (imaging lectures only):")
    mod_counts = defaultdict(int)
    for e in master_index:
        if e["modality"] != "none":
            mod_counts[e["modality"]] += 1
    for mod, count in sorted(
        mod_counts.items(), key=lambda x: -x[1]
    ):
        pct = count / total * 100
        print(f"  {mod:<20} {count:>4} ({pct:5.1f}%)")

    clinical = sum(
        1 for e in master_index if e["has_clinical_images"]
    )
    non_clinical = total - clinical
    print(f"\nClinical vs Non-Clinical:")
    print(f"  Has clinical images:     {clinical} "
          f"({clinical/total*100:.1f}%)")
    print(f"  No clinical images:      {non_clinical} "
          f"({non_clinical/total*100:.1f}%)")


def check_split_balance(master_index):
    """
    Verify test and val sets have adequate category coverage.
    Critical for valid benchmark evaluation.
    """
    print(f"\n{'='*70}")
    print(f"SPLIT BALANCE VERIFICATION")
    print(f"{'='*70}")

    required_categories = [
        "mathematical_foundation",
        "signal_theory",
        "imaging_modality",
        "computational"
    ]

    for split in ["val", "test"]:
        split_entries = [
            e for e in master_index if e["split"] == split
        ]
        split_cats = set(
            e["lecture_category"] for e in split_entries
        )

        print(f"\n{split.upper()} coverage:")
        all_good = True
        for req_cat in required_categories:
            present = req_cat in split_cats
            status = "✅" if present else "❌ MISSING"
            print(f"  {req_cat:<30} {status}")
            if not present:
                all_good = False

        if all_good:
            print(f"  → {split.upper()} has adequate coverage ✅")
        else:
            print(f"  → WARNING: {split.upper()} missing categories")


if __name__ == "__main__":

    # ── Step 1: Build index ───────────────────────────────
    master_index = build_master_index(LECTURES_ROOT, OUTPUT_DIR)

    # ── Step 2: Quality filter ────────────────────────────
    good_slides = [
        e for e in master_index
        if e["quality"]["quality_flag"] != "too_short_exclude"
    ]
    excluded = len(master_index) - len(good_slides)
    print(f"\nQuality filtering: excluded {excluded} slides "
          f"(< 20 words)")

    # ── Step 3: Print statistics ──────────────────────────
    print_overall_statistics(good_slides)
    print_split_summary(good_slides)
    check_split_balance(good_slides)

    # ── Step 4: Save all files ────────────────────────────
    output = Path(OUTPUT_DIR)

    # Master index
    with open(output / "master_index.json", "w") as f:
        json.dump(good_slides, f, indent=2)
    print(f"\nSaved master_index.json: {len(good_slides)} slides")

    # Split files
    split_counts = defaultdict(int)
    for split in ["train", "val", "test"]:
        split_data = [
            e for e in good_slides if e["split"] == split
        ]
        with open(output / f"{split}.json", "w") as f:
            json.dump(split_data, f, indent=2)
        split_counts[split] = len(split_data)
        print(f"Saved {split}.json: {len(split_data)} slides")

    # Split info summary
    split_info = {
        "strategy": "stratified_by_lecture_category",
        "seed": RANDOM_SEED,
        "train_lectures": STRATIFIED_SPLITS["train"],
        "val_lectures": STRATIFIED_SPLITS["val"],
        "test_lectures": STRATIFIED_SPLITS["test"],
        "split_counts": dict(split_counts),
        "category_map": LECTURE_TOPIC_MAP
    }
    with open(output / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Dataset index complete!")
    print(f"Files saved to: {OUTPUT_DIR}/")
    print(f"  master_index.json")
    print(f"  train.json         ({split_counts['train']} slides)")
    print(f"  val.json           ({split_counts['val']} slides)")
    print(f"  test.json          ({split_counts['test']} slides)")
    print(f"  split_info.json")

