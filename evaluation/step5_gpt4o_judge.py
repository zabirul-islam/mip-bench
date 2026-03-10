#!/usr/bin/env python3
# step5_gpt4o_judge.py
#
# Drop-in replacement for step5_rescore_v3.py
# Uses GPT-4o (OpenAI API) as judge instead of local vLLM Qwen model.
#
# Key differences from step5_rescore_v3.py:
#   - No vLLM / no GPU needed
#   - Reads claims_v3/ (falls back to claims/ like the v3 script)
#   - Saves to igs_scores_gpt4o/
#   - Saves incrementally every 10 slides (crash-safe)
#   - --resume flag to restart after a crash
#   - --compare flag to diff GPT-4o vs Qwen v3 scores
#
# Setup:
#   pip install openai
#   export OPENAI_API_KEY="sk-..."
#
# Usage (mirrors step5_rescore_v3.py exactly):
#   python step5_gpt4o_judge.py --model qwen2vl_7b --split test
#   python step5_gpt4o_judge.py --model all --split test
#   python step5_gpt4o_judge.py --model all --split test --compare
#   python step5_gpt4o_judge.py --model qwen2vl_7b --resume
#   python step5_gpt4o_judge.py --model all --judge gpt-4o-mini
#
# Exclusion file (auto-loads medlecture_bench/excluded_slides.json by default):
#   python step5_gpt4o_judge.py --model all                          # auto-loads excluded_slides.json
#   python step5_gpt4o_judge.py --model all --exclude excluded_slides.json  # filename only works too
#   python step5_gpt4o_judge.py --model all --exclude medlecture_bench/excluded_slides.json  # full path
#   python step5_gpt4o_judge.py --model all --no_exclude              # score ALL slides, no filter
#
# After this, run:
#   python step10_gpt4o.py

import json
import argparse
import re
import os
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit(
        "openai package not found.\n"
        "Install:  pip install openai\n"
    )

# ─────────────────────────────────────────────────────────────
# Paths — same base as step5_rescore_v3.py
BENCH_DIR    = "/home/islamm11/islamm11/MedLecture/medlecture_bench"
CLAIMS_V3    = f"{BENCH_DIR}/claims_v5"
CLAIMS_V1    = f"{BENCH_DIR}/claims_v4"
VLM_OUT_DIR  = f"{BENCH_DIR}/vlm_outputs"
SCORES_V3    = f"{BENCH_DIR}/igs_scores_v4"
SCORES_GPT4O = f"{BENCH_DIR}/igs_scores_gpt4o_v5"

# Default exclusion file — already exists at this path in your repo
DEFAULT_EXCLUDE = f"{BENCH_DIR}/excluded_slides.json"
# ─────────────────────────────────────────────────────────────

MODELS = [
    "qwen2vl_7b",
    "internvl2_8b",
    "internvl2_26b",
    "internvl2_40b",
    "llava_next_34b",
]

MODEL_LABELS = {
    "qwen2vl_7b":     "Qwen2-VL-7B",
    "internvl2_8b":   "InternVL2-8B",
    "internvl2_26b":  "InternVL2-26B",
    "internvl2_40b":  "InternVL2-40B",
    "llava_next_34b": "LLaVA-1.6-34B",
}

WEIGHT_CR = 0.40
WEIGHT_RV = 0.35
WEIGHT_SF = 0.25

MAX_RETRIES  = 5
RETRY_WAIT   = 2.0   # seconds, doubles each retry
SLIDE_DELAY  = 0.3   # polite pause between API calls (avoid rate limits)

# ─────────────────────────────────────────────────────────────
# Judge prompt — same rubric as step5_rescore_v3.py JUDGE_TEMPLATE
# ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert evaluator of educational content. "
    "Always respond with valid JSON only. No markdown, no preamble."
)

JUDGE_TEMPLATE = """You are scoring a student explanation against a teacher's intended lesson.

GROUND TRUTH — what the teacher intended to teach:
Key Concepts:
{concepts}

Relations:
{relations}

Instructional Scope:
{scope}

STUDENT (VLM) EXPLANATION:
{vlm_output}

Score THREE dimensions from 0.0 to 1.0:

CR (Concept Recall): What fraction of teacher's key concepts does the student correctly address?
  1.0=all | 0.7=most | 0.5=half | 0.3=few | 0.0=none

RV (Relational Validity): What fraction of teacher's relations are correctly stated?
  1.0=all correct | 0.7=most | 0.5=half | 0.3=few | 0.0=none
  Use 0.5 if no explicit relations exist.

SF (Scope Fidelity): Does the student stay within the instructional scope?
  1.0=perfectly on scope | 0.7=mostly | 0.5=some off-scope | 0.3=significant off-scope | 0.0=off

Respond ONLY with this JSON structure (no markdown, no extra text):
{{
  "concept_recall": 0.0,
  "relational_validity": 0.0,
  "scope_fidelity": 0.0,
  "cr_justification": "one sentence",
  "rv_justification": "one sentence",
  "sf_justification": "one sentence",
  "missed_concepts": ["list of missed"],
  "hallucinated_concepts": ["list of hallucinated"]
}}"""


# ─────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────

def build_messages(claims, vlm_output):
    """Build OpenAI messages list. Same content as build_judge_input() in v3 script."""
    concepts = "\n".join(
        f"  - {c}" for c in claims.get("key_concepts", [])
    ) or "  (none specified)"
    relations = "\n".join(
        f"  - {r}" for r in claims.get("relations", [])
    ) or "  (none specified)"
    scope = claims.get("scope_boundary", "Not specified")

    user_content = JUDGE_TEMPLATE.format(
        concepts=concepts,
        relations=relations,
        scope=scope,
        vlm_output=vlm_output[:1200],  # same truncation as v3 script
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


def parse_scores(text):
    """Identical to parse_scores() in step5_rescore_v3.py."""
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$',     '', text)
    text = re.sub(r'^```\s*',     '', text)
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
    try:
        return json.loads(text)
    except Exception:
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        try:
            return json.loads(text)
        except Exception:
            return None


def compute_igs(s):
    """Identical to compute_igs() in step5_rescore_v3.py."""
    cr = float(s.get("concept_recall",      0))
    rv = float(s.get("relational_validity", 0))
    sf = float(s.get("scope_fidelity",      0))
    return round(WEIGHT_CR*cr + WEIGHT_RV*rv + WEIGHT_SF*sf, 4)


def call_gpt4o(client, messages, judge_model):
    """Call OpenAI with exponential back-off. Returns raw text or None."""
    wait = RETRY_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model       = judge_model,
                messages    = messages,
                temperature = 0.0,    # deterministic scoring
                max_tokens  = 600,
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


def load_claims(sid):
    """Try claims_v3 first, fall back to claims v1. Returns (claims_dict, source_dir)."""
    for d in [CLAIMS_V3, CLAIMS_V1]:
        f = Path(d) / f"{sid}.json"
        if f.exists():
            with open(f) as fh:
                return json.load(fh), d
    return None, None


def save_json(data, path):
    """Atomic save via temp file."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


# ─────────────────────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────────────────────

def score_model(model_name, split, client, judge_model,
                resume=False, excluded_slides=None):
    """Score one VLM model. Mirrors score_model() in step5_rescore_v3.py."""
    excluded_slides = excluded_slides or set()
    scores_dir = Path(SCORES_GPT4O)
    scores_dir.mkdir(parents=True, exist_ok=True)

    vlm_file = Path(VLM_OUT_DIR) / f"{model_name}_{split}.json"
    if not vlm_file.exists():
        print(f"  ❌ VLM output not found: {vlm_file}")
        print(f"     Run step4_vlm_inference.py --model {model_name} first.")
        return None

    with open(vlm_file) as f:
        vlm_outputs = json.load(f)

    out_file = scores_dir / f"{model_name}_{split}_scores.json"

    # Resume: load existing to skip already-scored slides
    existing = {}
    if resume and out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
        print(f"  ↩  Resume mode: {len(existing)} slides already scored, skipping.")

    # Apply exclusions (same as step5_rescore_v3)
    if excluded_slides:
        vlm_outputs = {
            sid: v for sid, v in vlm_outputs.items()
            if sid not in excluded_slides
        }

    print(f"\n{'='*55}")
    print(f"  SCORING: {model_name} | {split}")
    print(f"  Judge:   {judge_model} (OpenAI API)")
    print(f"  Slides:  {len(vlm_outputs)}")
    print(f"  Claims:  claims_v5/ (fallback: claims_v4/)")
    print(f"  Output:  igs_scores_gpt4o_v5/")
    print(f"{'='*55}")

    all_scores    = dict(existing)
    n_ok = n_fail = n_skip = n_fallback = 0

    for idx, (sid, out_data) in enumerate(vlm_outputs.items(), 1):

        # Skip already scored (resume mode)
        if sid in existing:
            n_skip += 1
            continue

        claims, claims_src = load_claims(sid)
        if claims is None:
            print(f"  ⚠  {sid}: no claims file — skipping")
            n_skip += 1
            continue

        if claims_src == CLAIMS_V1:
            n_fallback += 1

        vlm_text = out_data.get("response", "")
        messages = build_messages(claims, vlm_text)

        raw = call_gpt4o(client, messages, judge_model)
        if raw is None:
            print(f"  ❌ {sid}: API failed after {MAX_RETRIES} retries")
            n_fail += 1
            continue

        scores = parse_scores(raw)
        if scores is None:
            print(f"  ❌ {sid}: JSON parse failed — raw: {raw[:80]}")
            n_fail += 1
            continue

        # Attach metadata (identical fields to v3 script)
        scores["igs"]              = compute_igs(scores)
        scores["slide_id"]         = sid
        scores["judge_model"]      = judge_model
        scores["lecture_category"] = out_data.get("lecture_category", "")
        scores["content_type"]     = out_data.get("content_type", "")
        scores["modality"]         = out_data.get("modality", "")

        all_scores[sid] = scores
        n_ok += 1

        print(
            f"  ✅ [{idx:>3}/{len(vlm_outputs)}] {sid:<16} "
            f"IGS={scores['igs']:.3f}  "
            f"CR={scores['concept_recall']:.2f}  "
            f"RV={scores['relational_validity']:.2f}  "
            f"SF={scores['scope_fidelity']:.2f}"
        )

        # Incremental save every 10 slides
        if n_ok % 10 == 0:
            save_json(all_scores, out_file)

        time.sleep(SLIDE_DELAY)

    # Final save
    save_json(all_scores, out_file)

    # Print summary — same format as step5_rescore_v3.py
    if all_scores:
        n   = len(all_scores)
        igs = sum(s["igs"]                 for s in all_scores.values()) / n
        cr  = sum(s["concept_recall"]      for s in all_scores.values()) / n
        rv  = sum(s["relational_validity"] for s in all_scores.values()) / n
        sf  = sum(s["scope_fidelity"]      for s in all_scores.values()) / n
        nz  = sum(1 for s in all_scores.values() if s["igs"] == 0.0)

        print(f"\n  V3 claims used : {n - n_fallback}")
        print(f"  V1 fallback    : {n_fallback}")
        print(f"  Skipped        : {n_skip}")
        print(f"  Parse failed   : {n_fail}")
        print(f"\n  RESULT: IGS={igs:.3f}  CR={cr:.3f}  RV={rv:.3f}  SF={sf:.3f}  "
              f"n={n}  zeros={nz} ({nz/n*100:.0f}%)")
        print(f"  Saved → {out_file}")

    return all_scores


# ─────────────────────────────────────────────────────────────
# Comparison helper
# ─────────────────────────────────────────────────────────────

def compare_qwen_vs_gpt4o(model_name, split):
    """Show Qwen v3 judge vs GPT-4o judge delta. Mirrors compare_v1_v3() in v3 script."""
    v3_file  = Path(SCORES_V3)    / f"{model_name}_{split}_scores.json"
    gpt_file = Path(SCORES_GPT4O) / f"{model_name}_{split}_scores.json"

    if not v3_file.exists():
        print(f"  ⚠  Qwen v3 scores not found — run step5_rescore_v3.py first for comparison")
        return
    if not gpt_file.exists():
        print(f"  ⚠  GPT-4o scores not found: {gpt_file}")
        return

    v3  = json.load(open(v3_file))
    gpt = json.load(open(gpt_file))
    shared = sorted(set(v3) & set(gpt))

    mean = lambda vals: sum(vals) / len(vals) if vals else 0.0

    print(f"\n{'='*55}")
    print(f"  QWEN v3 vs GPT-4o COMPARISON — {MODEL_LABELS.get(model_name, model_name)}")
    print(f"  {len(shared)} shared slides")
    print(f"{'='*55}")
    print(f"  {'Metric':<8} {'Qwen':>8} {'GPT-4o':>8} {'Δ':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for metric, key in [
        ("IGS", "igs"),
        ("CR",  "concept_recall"),
        ("RV",  "relational_validity"),
        ("SF",  "scope_fidelity"),
    ]:
        q_m = mean([v3[s][key]  for s in shared])
        g_m = mean([gpt[s][key] for s in shared])
        delta = g_m - q_m
        flag  = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")
        print(f"  {metric:<8} {q_m:>8.3f} {g_m:>8.3f} {delta:>+8.3f} {flag}")

    # Zero-IGS
    q_zero   = sum(1 for s in shared if v3[s]["igs"]  == 0.0)
    gpt_zero = sum(1 for s in shared if gpt[s]["igs"] == 0.0)
    print(f"\n  Zero-IGS:  Qwen={q_zero}  GPT-4o={gpt_zero}  Δ={gpt_zero-q_zero:+d}")

    # Most changed slides
    deltas   = {s: gpt[s]["igs"] - v3[s]["igs"] for s in shared}
    improved = sorted(deltas.items(), key=lambda x: -x[1])[:5]
    degraded = sorted(deltas.items(), key=lambda x:  x[1])[:5]

    print(f"\n  Top 5 higher with GPT-4o judge:")
    for sid, d in improved:
        if d > 0.01:
            cat = gpt[sid].get("lecture_category", "")
            print(f"    {sid:<16} Qwen={v3[sid]['igs']:.3f} → GPT4o={gpt[sid]['igs']:.3f} "
                  f"({d:+.3f})  [{cat}]")

    print(f"\n  Top 5 lower with GPT-4o judge:")
    for sid, d in degraded:
        if d < -0.01:
            cat = gpt[sid].get("lecture_category", "")
            print(f"    {sid:<16} Qwen={v3[sid]['igs']:.3f} → GPT4o={gpt[sid]['igs']:.3f} "
                  f"({d:+.3f})  [{cat}]")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score VLM outputs with GPT-4o as judge."
    )
    parser.add_argument(
        "--model", default="all",
        help="VLM model name or 'all'. Same values as step5_rescore_v3.py."
    )
    parser.add_argument(
        "--split", default="test", choices=["test", "val", "train"],
    )
    parser.add_argument(
        "--judge", default="gpt-4o",
        help="OpenAI model (default: gpt-4o). Also: gpt-4o-mini, gpt-4-turbo"
    )
    parser.add_argument(
        "--api_key", default=None,
        help="OpenAI API key. Falls back to OPENAI_API_KEY env var."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-scored slides (restart after crash)."
    )
    parser.add_argument(
        "--exclude", default=None,
        help=(
            "Path to JSON file with slide IDs to exclude. "
            "Defaults to medlecture_bench/excluded_slides.json if it exists. "
            "Pass the filename only (e.g. excluded_slides.json) and the script "
            "will find it inside medlecture_bench/ automatically."
        )
    )
    parser.add_argument(
        "--no_exclude", action="store_true",
        help="Disable exclusions entirely (score all slides, no filtering)."
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="After scoring, print Qwen v3 vs GPT-4o comparison."
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "No OpenAI API key found.\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "  or:  python step5_gpt4o_judge.py --api_key sk-..."
        )

    # ── Resolve exclusion file ───────────────────────────────────────────
    # Search order:
    #   1. --no_exclude flag → skip all exclusions
    #   2. --exclude <path>  → try as-is, then inside BENCH_DIR/
    #   3. no flag           → auto-use DEFAULT_EXCLUDE if it exists
    excluded_slides = set()

    if args.no_exclude:
        print("  --no_exclude set: scoring all slides, no filtering.")

    else:
        exclude_path = None

        if args.exclude:
            candidate = Path(args.exclude)
            if candidate.exists():
                # Absolute path or path relative to cwd — found directly
                exclude_path = candidate
            else:
                # Try inside medlecture_bench/ (user just typed the filename)
                fallback = Path(BENCH_DIR) / candidate.name
                if fallback.exists():
                    exclude_path = fallback
                    print(f"  ℹ  --exclude '{args.exclude}' not found in cwd, "
                          f"using {exclude_path} instead.")
                else:
                    raise SystemExit(
                        f"  ✗  Exclusion file not found:\n"
                        f"       tried: {candidate.resolve()}\n"
                        f"       tried: {fallback.resolve()}\n"
                        f"     Check the path and try again."
                    )
        else:
            # No --exclude flag: auto-load the default file if it exists
            default = Path(DEFAULT_EXCLUDE)
            if default.exists():
                exclude_path = default
                print(f"  ℹ  No --exclude flag given — auto-loading default: {exclude_path}")

        if exclude_path is not None:
            excluded_slides = set(json.load(open(exclude_path)))
            print(f"  ✅ Excluding {len(excluded_slides)} slides from: {exclude_path}")
        else:
            print("  ℹ  No exclusion file found or specified — scoring all slides.")

    # Init OpenAI client
    client = OpenAI(api_key=api_key)

    # Connectivity check
    print(f"\n  Testing OpenAI API ({args.judge})…")
    try:
        test = client.chat.completions.create(
            model    = args.judge,
            messages = [{"role": "user", "content": "Reply with one word: ready"}],
            max_tokens = 5,
        )
        print(f"  ✅ API OK — response: '{test.choices[0].message.content.strip()}'\n")
    except Exception as e:
        raise SystemExit(f"  ✗  API test failed: {e}")

    # Score
    models_to_run = MODELS if args.model == "all" else [args.model]

    for model_name in models_to_run:
        score_model(
            model_name      = model_name,
            split           = args.split,
            client          = client,
            judge_model     = args.judge,
            resume          = args.resume,
            excluded_slides = excluded_slides,
        )
        if args.compare:
            compare_qwen_vs_gpt4o(model_name, args.split)

    print(f"\n{'='*55}")
    print(f"  ALL DONE")
    print(f"  Scores saved to : {SCORES_GPT4O}/")
    print(f"\n  Next step:")
    print(f"    python step10_gpt4o.py")
    print(f"{'='*55}")
