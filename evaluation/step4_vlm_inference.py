# File: ~/islamm11/MedLecture/step4_vlm_inference.py
# VLM inference — vLLM 0.11.0 compatible
# Feeds slide images to VLMs, collects explanations

import json
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from PIL import Image

# ─────────────────────────────────────────────
DATASET_DIR   = "/home/islamm11/islamm11/MedLecture/medlecture_bench"
LECTURES_ROOT = "/home/islamm11/islamm11/MedLecture/Lectures"
OUTPUT_DIR    = "/home/islamm11/islamm11/MedLecture/medlecture_bench/vlm_outputs"
# ─────────────────────────────────────────────

VLM_CONFIGS = {
    "qwen2vl_7b": {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "tensor_parallel": 1,
        "max_model_len": 4096,
        "chat_template": "qwen2vl"
    },
    "internvl2_40b": {
        "model": "OpenGVLab/InternVL2-40B",
        "tensor_parallel": 2,
        "max_model_len": 4096,
        "chat_template": "internvl2",
        "batch_size": 4,
        "max_num_seqs": 8,
        "enforce_eager": True
    },
    "internvl2_8b": {
        "model": "OpenGVLab/InternVL2-8B",
        "tensor_parallel": 1,
        "max_model_len": 4096,
        "chat_template": "internvl"
    },
    "internvl2_26b": {
        "model": "OpenGVLab/InternVL2-26B",
        "tensor_parallel": 1,
        "max_model_len": 4096,
        "chat_template": "internvl"
    },
    "llava_next_34b": {
        "model": "llava-hf/llava-v1.6-34b-hf",
        "tensor_parallel": 2,
        "max_model_len": 3072,
        "chat_template": "llava_next",
        "batch_size": 1,
        "enforce_eager": True
    }
}

# ─────────────────────────────────────────────
# STANDARDIZED PROMPT — same for ALL VLMs
# ─────────────────────────────────────────────
INFERENCE_PROMPT = (
    "You are a student viewing a medical imaging lecture slide.\n"
    "Explain what this slide is teaching.\n\n"
    "Your explanation should:\n"
    "- Describe the main concept or principle being presented\n"
    "- Explain any equations, diagrams, or visual elements shown\n"
    "- State what a student should understand after seeing this slide\n"
    "- Be specific to what is actually shown, not general knowledge\n\n"
    "Provide a clear, educational explanation in 3-5 sentences."
)


def load_image(image_path):
    """Load PIL Image for vLLM 0.11.0 multimodal input."""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def build_input(config, image_path, prompt_text):
    """
    Build vLLM 0.11.0 multimodal input.
    vLLM 0.11.0 accepts PIL images directly via
    multi_modal_data dict — no base64 needed.
    """
    template = config["chat_template"]
    img = load_image(image_path)

    if template == "qwen2vl":
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"{prompt_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif template in ("internvl", "internvl2"):
        prompt = (
            "<|system|>\nYou are a helpful assistant.<|end|>\n"
            f"<|user|>\n<image>\n{prompt_text}<|end|>\n"
            "<|assistant|>\n"
        )
    elif template == "llava_next":
        # LLaVA-NeXT HF format — uses USER/ASSISTANT
        prompt = (
            f"USER: <image>\n{prompt_text}\nASSISTANT:"
        )
    elif template == "llava":
        prompt = (
            f"USER: <image>\n{prompt_text}\nASSISTANT:"
        )
    else:
        prompt = f"<image>\n{prompt_text}"

    return {
        "prompt": prompt,
        "multi_modal_data": {"image": img}
    }


def run_inference(vlm_name, split="test", validate=False):
    config = VLM_CONFIGS[vlm_name]
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{vlm_name}_{split}.json"
    checkpoint_file = (
        output_dir / f"ckpt_{vlm_name}_{split}.json"
    )

    # Load split data
    with open(f"{DATASET_DIR}/{split}.json") as f:
        slides = json.load(f)

    if validate:
        slides = slides[:5]
        print(f"VALIDATION MODE — 5 slides\n")

    # Load checkpoint
    completed = {}
    if checkpoint_file.exists() and not validate:
        with open(checkpoint_file) as f:
            completed = json.load(f)
        print(f"Resuming: {len(completed)} done")

    pending = [
        s for s in slides if s["id"] not in completed
    ]

    print(f"\n{'='*55}")
    print(f"VLM INFERENCE: {vlm_name}")
    print(f"{'='*55}")
    print(f"Model:   {config['model']}")
    print(f"Split:   {split} ({len(slides)} slides)")
    print(f"Pending: {len(pending)}")
    print(f"{'='*55}\n")

    if not pending:
        print("All done!")
        return completed

    # Load model
    print(f"Loading {vlm_name}...")
    # Use higher memory utilization for large models
    mem_util = 0.97 if config["tensor_parallel"] == 2 else 0.85

    llm = LLM(
        model=config["model"],
        tensor_parallel_size=config["tensor_parallel"],
        gpu_memory_utilization=mem_util,
        dtype="bfloat16",
        max_model_len=config["max_model_len"],
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=config.get("enforce_eager", False),
        max_num_seqs=config.get("max_num_seqs", 256),
        disable_custom_all_reduce=config.get("enforce_eager", False)
    )
    print("Model loaded ✅\n")

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=512,
        stop=["<|im_end|>", "<|end|>", "</s>", 
              "<|endoftext|>"]
    )

   # BATCH_SIZE = 16  # Conservative for stability
    BATCH_SIZE = config.get("batch_size", 16)
    results = dict(completed)

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:
                        batch_start + BATCH_SIZE]

        print(f"Batch {batch_start//BATCH_SIZE + 1}: "
              f"slides {batch_start+1}–"
              f"{min(batch_start+BATCH_SIZE, len(pending))}"
              f"/{len(pending)}")

        inputs = []
        valid_slides = []

        for slide in batch:
            img_path = (
                Path(LECTURES_ROOT) / slide["image_path"]
            )
            if not img_path.exists():
                print(f"  SKIP {slide['id']}: "
                      f"image not found at {img_path}")
                continue
            try:
                inp = build_input(
                    config, img_path, INFERENCE_PROMPT
                )
                inputs.append(inp)
                valid_slides.append(slide)
            except Exception as e:
                print(f"  SKIP {slide['id']}: {e}")

        if not inputs:
            continue

        try:
            outputs = llm.generate(
                inputs, sampling_params
            )
        except Exception as e:
            print(f"  BATCH ERROR: {e}")
            continue

        for slide, out in zip(valid_slides, outputs):
            sid = slide["id"]
            response = out.outputs[0].text.strip()

            results[sid] = {
                "id": sid,
                "vlm": vlm_name,
                "lecture_topic": slide["lecture_topic"],
                "lecture_category": slide[
                    "lecture_category"
                ],
                "content_type": slide["content_type"],
                "modality": slide["modality"],
                "split": slide["split"],
                "image_path": slide["image_path"],
                "response": response
            }

            preview = response[:80].replace('\n', ' ')
            print(f"  ✅ {sid}: {preview}...")

            if validate:
                print(f"     FULL RESPONSE:\n"
                      f"     {response}\n")

        # Checkpoint
        if not validate:
            with open(checkpoint_file, "w") as f:
                json.dump(results, f)
            print(f"  [checkpoint: {len(results)}]\n")

    # Save final
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved {len(results)} outputs → {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        choices=list(VLM_CONFIGS.keys())
    )
    parser.add_argument(
        "--split", default="test",
        choices=["test", "val", "train", "all"]
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run 5 slides only"
    )
    args = parser.parse_args()
    run_inference(args.model, args.split, args.validate)
