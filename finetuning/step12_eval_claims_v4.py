#!/usr/bin/env python3
"""
Fine-tune Qwen2-VL-7B on MedLectureBench train split using LoRA.
Target: improve IGS on test split, especially on visually-grounded slides.

Install deps:
  pip install peft transformers accelerate bitsandbytes --break-system-packages
  pip install qwen-vl-utils --break-system-packages

Run from: ~/islamm11/MedLecture/
Usage:
  python step12_finetune.py --validate        # dry run, 5 samples
  python step12_finetune.py                   # full training
  python step12_finetune.py --eval-only       # eval checkpoint on test split
"""

import json, re, argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────
DATASET_DIR    = "/home/islamm11/islamm11/MedLecture/medlecture_bench"
LECTURES_ROOT  = "/home/islamm11/islamm11/MedLecture/Lectures"
DECISIONS_CSV  = "/home/islamm11/islamm11/MedLecture/medlecture_bench/manual_review/review_decisions.csv"
VLM_OUT_DIR    = "/home/islamm11/islamm11/MedLecture/medlecture_bench/vlm_outputs"
SCORES_DIR     = "/home/islamm11/islamm11/MedLecture/medlecture_bench/igs_scores_gpt4o_v5"
OUTPUT_DIR     = "/home/islamm11/islamm11/MedLecture/medlecture_bench/finetuned"

BASE_MODEL     = "Qwen/Qwen2-VL-7B-Instruct"
# ─────────────────────────────────────────────────────────────

# LoRA config — conservative for single H100
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAIN_CONFIG = {
    "num_epochs":          3,
    "batch_size":          2,       # per device
    "grad_accum":          8,       # effective batch = 16
    "learning_rate":       2e-4,
    "warmup_ratio":        0.05,
    "lr_scheduler":        "cosine",
    "max_seq_len":         2048,
    "save_steps":          50,
    "eval_steps":          50,
    "logging_steps":       10,
    "fp16":                False,
    "bf16":                True,    # H100 prefers bf16
    "dataloader_workers":  2,
    "seed":                42,
}

# ── System prompt for fine-tuning ─────────────────────────────
SYSTEM_PROMPT = """You are an AI teaching assistant for a medical imaging course.
Given a lecture slide image, explain what this slide is teaching to a student.
Focus on the pedagogical content: key concepts, relationships between ideas,
and the instructional purpose of the slide.
Be specific and accurate. Do not describe the slide's visual layout or metadata."""

def get_teacher_text(sid):
    m = re.match(r'L(\d+)_S(\d+)', sid)
    if not m: return ""
    p = Path(LECTURES_ROOT)/f"Lecture {int(m.group(1))}"/f"Texts/Slide{int(m.group(2))}.txt"
    return p.read_text().strip() if p.exists() else ""

def get_image_path(sid):
    m = re.match(r'L(\d+)_S(\d+)', sid)
    if not m: return None
    p = Path(LECTURES_ROOT)/f"Lecture {int(m.group(1))}"/f"Images/Slide{int(m.group(2))}.JPG"
    return str(p) if p.exists() else None

def load_decisions():
    import csv
    p = Path(DECISIONS_CSV)
    if not p.exists(): return set()
    excl = set()
    with open(p) as f:
        for row in csv.DictReader(f):
            d = row.get("decision","").strip()
            if d.startswith("B") or d.startswith("C"):
                excl.add(row.get("slide_id","").strip())
    return excl

def build_dataset(split, excl, validate=False):
    """Build image-text pairs for fine-tuning."""
    data_file = Path(DATASET_DIR) / f"{split}.json"
    slides = json.load(open(data_file))

    samples = []
    skipped_no_img, skipped_no_text, skipped_excl = 0, 0, 0

    for slide in slides:
        sid = slide["id"]

        # Skip narration-grounded and administrative slides
        # (model can't learn from slides where image doesn't encode the lesson)
        if sid in excl:
            skipped_excl += 1
            continue

        img_path = get_image_path(sid)
        if not img_path:
            skipped_no_img += 1
            continue

        teacher_text = get_teacher_text(sid)
        if not teacher_text or len(teacher_text) < 50:
            skipped_no_text += 1
            continue

        samples.append({
            "id":               sid,
            "image_path":       img_path,
            "teacher_text":     teacher_text,
            "content_type":     slide.get("content_type", ""),
            "lecture_category": slide.get("lecture_category", ""),
            "lecture_topic":    slide.get("lecture_topic", ""),
        })

    print(f"\n  {split} dataset:")
    print(f"    Total slides:         {len(slides)}")
    print(f"    Excluded (B+C):       {skipped_excl}")
    print(f"    Skipped (no image):   {skipped_no_img}")
    print(f"    Skipped (no text):    {skipped_no_text}")
    print(f"    Training samples:     {len(samples)}")

    if validate:
        samples = samples[:5]
        print(f"    VALIDATE: using {len(samples)} samples only")

    return samples

def build_conversation(sample):
    """Build Qwen2-VL chat format for one sample."""
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text",  "text":  (
                        f"This is a slide from a medical imaging lecture on: "
                        f"{sample['lecture_topic']}.\n"
                        f"Content type: {sample['content_type']}.\n\n"
                        f"What is this slide teaching?"
                    )}
                ]
            },
            {
                "role": "assistant",
                "content": sample["teacher_text"]
            }
        ]
    }

def train(validate=False):
    """Main fine-tuning loop."""
    try:
        import torch
        from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                                   TrainingArguments, Trainer)
        from peft import LoraConfig, get_peft_model, TaskType
        from torch.utils.data import Dataset
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install: pip install peft transformers accelerate qwen-vl-utils --break-system-packages")
        return

    print(f"\n{'='*60}")
    print(f"  FINE-TUNING: {BASE_MODEL}")
    print(f"  LoRA r={LORA_CONFIG['r']}  epochs={TRAIN_CONFIG['num_epochs']}")
    print(f"{'='*60}")

    excl = load_decisions()
    train_samples = build_dataset("train", excl, validate)
    val_samples   = build_dataset("val",   excl, validate)

    if not train_samples:
        print("❌ No training samples found"); return

    # Load model + processor
    print(f"\n  Loading {BASE_MODEL}...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("  Model loaded ✅")

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset class
    class MedLectureDataset(Dataset):
        def __init__(self, samples, processor, max_len):
            self.samples   = samples
            self.processor = processor
            self.max_len   = max_len

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            conv   = build_conversation(sample)
            text   = self.processor.apply_chat_template(
                conv["messages"], tokenize=False, add_generation_prompt=False
            )
            image_inputs, video_inputs = process_vision_info(conv["messages"])
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
            )
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs

    train_ds = MedLectureDataset(train_samples, processor, TRAIN_CONFIG["max_seq_len"])
    val_ds   = MedLectureDataset(val_samples,   processor, TRAIN_CONFIG["max_seq_len"])

    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=1 if validate else TRAIN_CONFIG["num_epochs"],
        per_device_train_batch_size=TRAIN_CONFIG["batch_size"],
        gradient_accumulation_steps=TRAIN_CONFIG["grad_accum"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
        lr_scheduler_type=TRAIN_CONFIG["lr_scheduler"],
        bf16=TRAIN_CONFIG["bf16"],
        fp16=TRAIN_CONFIG["fp16"],
        logging_steps=TRAIN_CONFIG["logging_steps"],
        save_steps=10 if validate else TRAIN_CONFIG["save_steps"],
        eval_steps=10 if validate else TRAIN_CONFIG["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=TRAIN_CONFIG["dataloader_workers"],
        seed=TRAIN_CONFIG["seed"],
        report_to="none",
        max_steps=5 if validate else -1,
    )

    from transformers import DataCollatorWithPadding

    class VLCollator:
        def __call__(self, features):
            import torch
            keys = features[0].keys()
            batch = {}
            for k in keys:
                tensors = [f[k] for f in features]
                max_len = max(t.shape[0] for t in tensors)
                if k == "labels":
                    padded = [torch.nn.functional.pad(t, (0, max_len-t.shape[0]), value=-100) for t in tensors]
                else:
                    padded = [torch.nn.functional.pad(t, (0, max_len-t.shape[0]), value=0) for t in tensors]
                batch[k] = torch.stack(padded)
            return batch

    trainer = Trainer(
        data_collator=VLCollator(),
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print(f"\n  Starting training...")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    trainer.train()

    # Save LoRA adapter
    adapter_path = out_path / "lora_claims_v4"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"\n  ✅ LoRA adapter saved → {adapter_path}")
    print(f"\n  Next: python step12_finetune.py --eval-only")

def eval_finetuned(split="test"):
    """
    Run fine-tuned model inference on test split and save to vlm_outputs.
    Then re-score IGS to measure improvement.
    """
    try:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from peft import PeftModel
        from vllm import LLM, SamplingParams
        from qwen_vl_utils import process_vision_info
        from PIL import Image
    except ImportError as e:
        print(f"❌ Missing: {e}"); return

    adapter_path = Path(OUTPUT_DIR) / "lora_claims_v4"
    if not adapter_path.exists():
        print(f"❌ No adapter found at {adapter_path}")
        print("   Run training first: python step12_finetune.py")
        return

    print(f"\n{'='*60}")
    print(f"  EVALUATING FINE-TUNED MODEL on {split} split")
    print(f"{'='*60}")

    excl = load_decisions()
    data_file = Path(DATASET_DIR) / f"{split}.json"
    slides = json.load(open(data_file))
    slides = [s for s in slides if s["id"] not in excl]

    print(f"  Loading base model + LoRA adapter...")
    processor = AutoProcessor.from_pretrained(str(adapter_path), trust_remote_code=True)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    print("  Model loaded ✅")

    out_dir = Path(VLM_OUT_DIR)
    results = {}

    for slide in slides:
        sid      = slide["id"]
        img_path = get_image_path(sid)
        if not img_path: continue

        messages = [{
            "role": "system", "content": SYSTEM_PROMPT
        }, {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text",  "text": (
                    f"This is a slide from a medical imaging lecture on: "
                    f"{slide['lecture_topic']}.\n"
                    f"Content type: {slide.get('content_type','')}.\n\n"
                    f"What is this slide teaching?"
                )}
            ]
        }]

        import torch
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = processor.decode(
            out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        results[sid] = {
            "response":         response,
            "content_type":     slide.get("content_type",""),
            "lecture_category": slide.get("lecture_category",""),
            "lecture_topic":    slide.get("lecture_topic",""),
            "model":            "qwen2vl_7b_finetuned",
        }
        print(f"  ✅ {sid}: {response[:80]}...")

    out_file = out_dir / f"qwen2vl_7b_lora_claims_{split}.json"
    json.dump(results, open(out_file,"w"), indent=2)
    print(f"\n  ✅ Saved {len(results)} outputs → {out_file}")
    print(f"\n  Next: score the fine-tuned model:")
    print(f"    python step5_gpt4o_judge.py --model qwen2vl_7b_lora_claims --split test")

def print_training_plan():
    excl = load_decisions()
    train_data = json.load(open(Path(DATASET_DIR)/"train.json"))
    val_data   = json.load(open(Path(DATASET_DIR)/"val.json"))

    train_usable = sum(1 for s in train_data
                       if s["id"] not in excl and get_image_path(s["id"])
                       and get_teacher_text(s["id"]) and len(get_teacher_text(s["id"]))>=50)
    val_usable   = sum(1 for s in val_data
                       if s["id"] not in excl and get_image_path(s["id"])
                       and get_teacher_text(s["id"]) and len(get_teacher_text(s["id"]))>=50)

    steps_per_epoch = train_usable // (TRAIN_CONFIG["batch_size"] * TRAIN_CONFIG["grad_accum"])
    total_steps     = steps_per_epoch * TRAIN_CONFIG["num_epochs"]

    print(f"\n{'='*60}")
    print(f"  FINE-TUNING PLAN")
    print(f"{'='*60}")
    print(f"  Base model:        {BASE_MODEL}")
    print(f"  LoRA r:            {LORA_CONFIG['r']}")
    print(f"  Target modules:    {len(LORA_CONFIG['target_modules'])} attention layers")
    print(f"  Trainable params:  ~0.5% of total (LoRA efficiency)")
    print(f"\n  Dataset (excl {len(excl)} narration/admin slides):")
    print(f"    Train samples:   {train_usable}")
    print(f"    Val samples:     {val_usable}")
    print(f"\n  Training config:")
    print(f"    Epochs:          {TRAIN_CONFIG['num_epochs']}")
    print(f"    Batch size:      {TRAIN_CONFIG['batch_size']} × {TRAIN_CONFIG['grad_accum']} accum = {TRAIN_CONFIG['batch_size']*TRAIN_CONFIG['grad_accum']} effective")
    print(f"    Learning rate:   {TRAIN_CONFIG['learning_rate']}")
    print(f"    Steps/epoch:     ~{steps_per_epoch}")
    print(f"    Total steps:     ~{total_steps}")
    print(f"    Est. time:       ~{total_steps*30//3600}h on single H100")
    print(f"\n  Output:            {OUTPUT_DIR}/")
    print(f"\n  Run order:")
    print(f"    1. python step12_finetune.py --validate    # dry run (5 min)")
    print(f"    2. python step12_finetune.py               # full training")
    print(f"    3. python step12_finetune.py --eval-only   # generate outputs")
    print(f"    4. python step5_gpt4o_judge.py --model qwen2vl_7b_lora_claims --split test")
    print(f"    5. python step10_final_results.py          # compare base vs finetuned")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate",  action="store_true", help="Dry run with 5 samples")
    parser.add_argument("--eval-only", action="store_true", help="Run inference with saved checkpoint")
    parser.add_argument("--plan",      action="store_true", help="Show training plan without running")
    parser.add_argument("--split",     default="test")
    args = parser.parse_args()

    if args.plan:
        print_training_plan()
    elif args.eval_only:
        eval_finetuned(args.split)
    else:
        train(validate=args.validate)
