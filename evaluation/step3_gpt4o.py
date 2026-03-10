#!/usr/bin/env python3
# step3_gpt4o.py
#
# Re-extracts claims from updated teacher texts using GPT-4o
# instead of Qwen2.5-7B-Instruct (step3_v4.py).
#
# Key change for NeurIPS submission:
#   - Extractor model: GPT-4o (OpenAI API)
#   - Judge model:     GPT-4o (step5_gpt4o_judge.py)
#   - Evaluated VLMs:  Qwen2-VL-7B, InternVL2-*, LLaVA-Next-34B
#   → No model-family overlap between ground truth construction
#     and any evaluated model.
#
# Run from: ~/islamm11/MedLecture/
# Usage:
#   python step3_gpt4o.py --split all
#   python step3_gpt4o.py --split test --validate
#   python step3_gpt4o.py --split all --resume   # restart after crash

import json
import argparse
import re
import os
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai package not found.\nInstall: pip install openai")

# ─────────────────────────────────────────────────────────────
DATASET_DIR   = "/home/islamm11/islamm11/MedLecture/medlecture_bench"
LECTURES_ROOT = "/home/islamm11/islamm11/MedLecture/Lectures"
OUTPUT_DIR    = "/home/islamm11/islamm11/MedLecture/medlecture_bench/claims_v5"
CLAIMS_MODEL  = "gpt-4o"
# ─────────────────────────────────────────────────────────────

MAX_RETRIES  = 5
RETRY_WAIT   = 2.0
SLIDE_DELAY  = 0.3

# ── Identical prompt content to step3_v4.py ──────────────────
# Only the delivery mechanism changes (OpenAI API vs vLLM).

PROMPT_MATHEMATICAL = """You are an expert in medical imaging education.
Extract structured pedagogical claims from a teacher's explanation of a
mathematical or technical slide.

Pay special attention to phrases like "here's the key idea",
"what's really important", "the fundamental", "remember that".

Extract:
1. KEY_CONCEPTS: Specific mathematical/technical concepts taught. Be precise. Only CENTRAL concepts.
2. RELATIONS: Cause-effect or definitional relationships. Format: "A → B"
3. SCOPE_BOUNDARY: 1-2 sentences. What should student understand? What is out of scope?
4. CONTENT_TYPE: One of: equation_derivation | concept_introduction | system_description | image_interpretation | comparison | application | overview

Respond ONLY with valid JSON, no markdown:
{
  "key_concepts": ["concept1", "concept2"],
  "relations": ["A → B"],
  "scope_boundary": "After this slide...",
  "content_type": "type",
  "concept_count": 0,
  "relation_count": 0
}"""

PROMPT_IMAGING = """You are an expert in medical imaging education.
Extract structured pedagogical claims from a teacher's explanation of a
medical imaging slide involving imaging systems, modalities, or clinical content.

Pay special attention to emphasis markers like "here's the key idea", "what's really important".

Extract:
1. KEY_CONCEPTS: Imaging concepts, physical principles, clinical ideas, or contextual concepts if they are the lesson.
2. RELATIONS: Physical, technical, or clinical relationships. Example: "higher kVp → greater photon penetration"
3. SCOPE_BOUNDARY: 1-2 sentences.
4. CONTENT_TYPE: One of: equation_derivation | concept_introduction | system_description | image_interpretation | comparison | application | overview

Respond ONLY with valid JSON, no markdown:
{
  "key_concepts": ["concept1", "concept2"],
  "relations": ["A → B"],
  "scope_boundary": "After this slide...",
  "content_type": "type",
  "concept_count": 0,
  "relation_count": 0
}"""

PROMPT_CONCEPTUAL = """You are an expert in medical imaging education.
Extract structured pedagogical claims from a teacher's motivational or
conceptual explanation. These slides teach PURPOSE, CONTEXT, or HIGH-LEVEL UNDERSTANDING.

Extract:
1. KEY_CONCEPTS: High-level concepts, motivations, clinical impact, historical context, aspirational concepts.
2. RELATIONS: Conceptual connections. Example: "medical imaging → non-invasive view of human body"
3. SCOPE_BOUNDARY: 1-2 sentences. What perspective should student gain? What technical depth is NOT expected?
4. CONTENT_TYPE: One of: overview | concept_introduction | application | comparison

Respond ONLY with valid JSON, no markdown:
{
  "key_concepts": ["concept1", "concept2"],
  "relations": ["A → B"],
  "scope_boundary": "After this slide...",
  "content_type": "type",
  "concept_count": 0,
  "relation_count": 0
}"""


def select_prompt(content_type, lecture_category):
    """Identical routing logic to step3_v4.py."""
    if lecture_category in ["fundamentals", "physics_foundation"]:
        return PROMPT_CONCEPTUAL, "conceptual"
    if content_type == "conceptual":
        return PROMPT_CONCEPTUAL, "conceptual"
    if content_type in ["mathematical", "computational"]:
        return PROMPT_MATHEMATICAL, "mathematical"
    if lecture_category in ["mathematical_foundation", "signal_theory", "systems_theory"]:
        return PROMPT_MATHEMATICAL, "mathematical"
    return PROMPT_IMAGING, "imaging"


def get_teacher_text(sid):
    """Load teacher text — identical to step3_v4.py get_updated_teacher_text()."""
    match = re.match(r'L(\d+)_S(\d+)', sid)
    if not match:
        return ""
    lec_num   = int(match.group(1))
    slide_num = int(match.group(2))
    txt_path  = (Path(LECTURES_ROOT) / f"Lecture {lec_num}"
                 / "Texts" / f"Slide{slide_num}.txt")
    return txt_path.read_text().strip() if txt_path.exists() else ""


def build_messages(system_prompt, slide, teacher_text):
    """Build OpenAI messages list. Same content as build_qwen_prompt() in step3_v4."""
    user_content = (
        f"Lecture topic: {slide['lecture_topic']}\n"
        f"Content type: {slide['content_type']}\n"
        f"Lecture category: {slide['lecture_category']}\n\n"
        f"Teacher explanation:\n{teacher_text}\n\n"
        f"Extract the pedagogical claims as instructed."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


def parse_json_response(text):
    """Identical to parse_json_response() in step3_v4.py."""
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$',     '', text)
    text = re.sub(r'^```\s*',     '', text)
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        try:
            return json.loads(text)
        except Exception:
            return None


def call_gpt4o(client, messages):
    """Call GPT-4o with exponential back-off. Returns raw text or None."""
    wait = RETRY_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model       = CLAIMS_MODEL,
                messages    = messages,
                temperature = 0.1,    # same as Qwen sampling temp in step3_v4
                max_tokens  = 800,    # same as step3_v4 sampling_params
            )
            return resp.choices[0].message.content
        except Exception as exc:
            err = str(exc)
            if attempt < MAX_RETRIES and any(
                k in err.lower()
                for k in ("rate", "limit", "timeout", "500", "502", "503")
            ):
                print(f"    ⚠  API error (attempt {attempt}/{MAX_RETRIES}): {err[:80]}")
                print(f"       Retrying in {wait:.0f}s…")
                time.sleep(wait)
                wait *= 2
            else:
                print(f"    ✗  API error (non-retryable): {err[:120]}")
                return None
    return None


def save_checkpoint(completed, checkpoint_file):
    tmp = Path(str(checkpoint_file) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(list(completed), f)
    tmp.rename(checkpoint_file)


def run_extraction(slides, output_dir, client, validate=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint_file = output / "checkpoint.json"
    completed = set()
    if checkpoint_file.exists() and not validate:
        with open(checkpoint_file) as f:
            completed = set(json.load(f))
        print(f"  Resuming: {len(completed)} already done\n")

    # Filter: skip completed, skip slides with no teacher text
    pending, skipped_no_text = [], 0
    for s in slides:
        if s["id"] in completed:
            continue
        text = get_teacher_text(s["id"])
        if not text:
            skipped_no_text += 1
            continue
        s["_teacher_text"] = text
        pending.append(s)

    print(f"  Slides to process:      {len(pending)}")
    if skipped_no_text:
        print(f"  Skipped (no text file): {skipped_no_text}")
    if not pending:
        print("  All done!")
        return

    n_ok = n_fail = 0

    for idx, slide in enumerate(pending, 1):
        sid          = slide["id"]
        teacher_text = slide["_teacher_text"]
        sys_p, ptype = select_prompt(
            slide.get("content_type", "mixed"),
            slide.get("lecture_category", "unknown")
        )
        messages = build_messages(sys_p, slide, teacher_text)

        raw = call_gpt4o(client, messages)
        if raw is None:
            print(f"  ❌ {sid}: API failed after {MAX_RETRIES} retries")
            n_fail += 1
            continue

        claims = parse_json_response(raw)
        if claims is None:
            print(f"  ❌ {sid}: JSON parse failed — raw: {raw[:80]}")
            # Save raw for debugging (same as step3_v4)
            with open(output / f"{sid}_raw.txt", "w") as f:
                f.write(raw)
            n_fail += 1
            continue

        # Attach metadata (identical fields to step3_v4)
        claims["slide_id"]       = sid
        claims["prompt_type"]    = ptype
        claims["concept_count"]  = len(claims.get("key_concepts", []))
        claims["relation_count"] = len(claims.get("relations", []))
        claims["text_version"]   = "v5_gpt4o"

        with open(output / f"{sid}.json", "w") as f:
            json.dump(claims, f, indent=2)

        completed.add(sid)
        n_ok += 1

        print(
            f"  ✅ [{idx:>4}/{len(pending)}] {sid} [{ptype}] "
            f"{claims['concept_count']}c {claims['relation_count']}r "
            f"| {claims.get('content_type','?')}"
        )

        if validate:
            print(f"       Concepts:  {claims['key_concepts'][:3]}")
            print(f"       Relations: {claims['relations'][:2]}")
            print(f"       Scope:     {claims['scope_boundary'][:120]}")
            print(f"       Orig text: {teacher_text[:150]}...")
            print()

        # Checkpoint every 20 slides (crash-safe)
        if n_ok % 20 == 0 and not validate:
            save_checkpoint(completed, checkpoint_file)
            print(f"  [checkpoint: {len(completed)} done]")

        time.sleep(SLIDE_DELAY)

    # Final checkpoint
    if not validate:
        save_checkpoint(completed, checkpoint_file)

    print(f"\n  Extracted: {n_ok}  Failed: {n_fail}")


def print_stats(output_dir):
    """Identical to print_stats() in step3_v4.py."""
    output = Path(output_dir)
    files  = list(output.glob("L*.json"))
    if not files:
        return

    concept_counts, relation_counts = [], []
    prompt_dist, content_dist = {}, {}

    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        concept_counts.append(d.get("concept_count", 0))
        relation_counts.append(d.get("relation_count", 0))
        pt = d.get("prompt_type", "?")
        ct = d.get("content_type", "?")
        prompt_dist[pt] = prompt_dist.get(pt, 0) + 1
        content_dist[ct] = content_dist.get(ct, 0) + 1

    n = len(files)
    print(f"\n{'='*55}")
    print(f"  CLAIMS V5 STATS — GPT-4o extractor ({n} slides)")
    print(f"{'='*55}")
    print(f"  Concepts/slide:  mean={sum(concept_counts)/n:.1f}  "
          f"min={min(concept_counts)}  max={max(concept_counts)}")
    print(f"  Relations/slide: mean={sum(relation_counts)/n:.1f}  "
          f"min={min(relation_counts)}  max={max(relation_counts)}")
    print(f"\n  Prompt type distribution:")
    for k, v in sorted(prompt_dist.items()):
        print(f"    {k:<15} {v:>5}  ({v/n*100:.1f}%)")
    print(f"\n  Content type detected:")
    for k, v in sorted(content_dist.items(), key=lambda x: -x[1]):
        print(f"    {k:<25} {v:>5}  ({v/n*100:.1f}%)")

    # Compare with v4 (Qwen extractor) stats
    v4_dir   = Path(DATASET_DIR) / "claims_v4"
    v4_files = list(v4_dir.glob("L*.json"))
    if v4_files:
        v4_concepts  = []
        v4_relations = []
        for f in v4_files:
            with open(f) as fp:
                d = json.load(fp)
            v4_concepts.append(d.get("concept_count", 0))
            v4_relations.append(d.get("relation_count", 0))
        print(f"\n  V4 (Qwen) vs V5 (GPT-4o) comparison:")
        print(f"    {'Metric':<25} {'V4 Qwen':>10} {'V5 GPT4o':>10}")
        print(f"    {'-'*25} {'-'*10} {'-'*10}")
        print(f"    {'Mean concepts/slide':<25} "
              f"{sum(v4_concepts)/len(v4_concepts):>10.1f} "
              f"{sum(concept_counts)/n:>10.1f}")
        print(f"    {'Mean relations/slide':<25} "
              f"{sum(v4_relations)/len(v4_relations):>10.1f} "
              f"{sum(relation_counts)/n:>10.1f}")
        print(f"    {'Total files':<25} "
              f"{len(v4_files):>10} {n:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract claims using GPT-4o (NeurIPS-clean pipeline)."
    )
    parser.add_argument("--split", default="all",
                        choices=["test", "val", "train", "all"])
    parser.add_argument("--validate", action="store_true",
                        help="Run 5 slides only to check quality.")
    parser.add_argument("--api_key", default=None,
                        help="OpenAI API key. Falls back to OPENAI_API_KEY env var.")
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "No OpenAI API key found.\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "  or:  python step3_gpt4o.py --api_key sk-..."
        )

    # Load slide metadata (identical to step3_v4.py)
    data_file = (f"{DATASET_DIR}/master_index.json"
                 if args.split == "all"
                 else f"{DATASET_DIR}/{args.split}.json")
    with open(data_file) as f:
        slides = json.load(f)

    if args.validate:
        selected, seen_types = [], set()
        for s in slides:
            key = f"{s.get('content_type','')}_{s.get('lecture_category','')}"
            if key not in seen_types and len(selected) < 5:
                seen_types.add(key)
                selected.append(s)
        slides = selected
        print(f"\n  VALIDATE MODE — {len(slides)} slides")
        for s in slides:
            txt = get_teacher_text(s["id"])
            print(f"  {s['id']}: {s.get('content_type','')} | text={len(txt)}chars")
        print()

    print(f"\n{'='*55}")
    print(f"  CLAIMS EXTRACTION V5 — GPT-4o extractor")
    print(f"{'='*55}")
    print(f"  Extractor:   {CLAIMS_MODEL} (OpenAI API)")
    print(f"  Split:       {args.split}")
    print(f"  Slides:      {len(slides)}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"  Texts from:  {LECTURES_ROOT}/Lecture N/Texts/SlideN.txt")
    print(f"  Prompts:     identical content to step3_v4.py")
    print(f"{'='*55}\n")

    # Connectivity check
    client = OpenAI(api_key=api_key)
    print(f"  Testing OpenAI API ({CLAIMS_MODEL})…")
    try:
        test = client.chat.completions.create(
            model    = CLAIMS_MODEL,
            messages = [{"role": "user", "content": "Reply with one word: ready"}],
            max_tokens = 5,
        )
        print(f"  ✅ API OK — '{test.choices[0].message.content.strip()}'\n")
    except Exception as e:
        raise SystemExit(f"  ✗  API test failed: {e}")

    run_extraction(slides, OUTPUT_DIR, client, validate=args.validate)

    if not args.validate:
        print_stats(OUTPUT_DIR)
        print(f"\n✅ Done. Next steps:")
        print(f"   python step5_gpt4o_judge.py --model all --split test")
        print(f"   python step10_gpt4o.py")
