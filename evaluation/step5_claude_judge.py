#!/usr/bin/env python3
# step5_claude_judge.py
#
# Drop-in twin of step5_gpt4o_judge.py that uses the Anthropic Claude API
# as a second LLM judge. Outputs go to igs_scores_claude_v5/.
#
# Purpose: multi-judge ensemble (Claude + GPT-4o) — neutralises the
# single-judge bias concern. Same rubric, same ordinal anchors, same
# ground-truth-conditioned prompt as the GPT-4o judge.
#
# Setup:
#   pip install anthropic
#   export ANTHROPIC_API_KEY="sk-ant-..."
#
# Usage (mirrors step5_gpt4o_judge.py):
#   python step5_claude_judge.py --model qwen2vl_7b --split test
#   python step5_claude_judge.py --model all --split test
#   python step5_claude_judge.py --model all --split test --compare
#   python step5_claude_judge.py --model qwen2vl_7b --resume
#   python step5_claude_judge.py --model all --judge claude-opus-4-6
#
# After scoring, run:
#   python step16_multi_judge_agreement.py
#
# NOTE: uses a non-evaluated model family (Anthropic Claude) — same
# motivation as using GPT-4o: claim vocabulary remains independent of
# the evaluated VLM families.

import argparse
import json
import os
import re
import time
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError:
    raise SystemExit(
        "anthropic package not found.\n"
        "Install:  pip install anthropic\n"
    )

# ─────────────────────────────────────────────────────────────
# Paths — same base as step5_gpt4o_judge.py on the H100 box
BENCH_DIR      = "/home/islamm11/islamm11/MedLecture/medlecture_bench"
CLAIMS_V5      = f"{BENCH_DIR}/claims_v5"
CLAIMS_V4      = f"{BENCH_DIR}/claims_v4"
VLM_OUT_DIR    = f"{BENCH_DIR}/vlm_outputs"
SCORES_GPT4O   = f"{BENCH_DIR}/igs_scores_gpt4o_v5"
SCORES_CLAUDE  = f"{BENCH_DIR}/igs_scores_claude_v5"

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

MAX_RETRIES = 5
RETRY_WAIT  = 2.0
SLIDE_DELAY = 0.3

# Identical rubric to step5_gpt4o_judge.py — deliberate
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


def build_prompt(claims, vlm_output):
    concepts  = "\n".join(f"  - {c}" for c in claims.get("key_concepts", [])) \
                or "  (none specified)"
    relations = "\n".join(f"  - {r}" for r in claims.get("relations", []))  \
                or "  (none specified)"
    scope     = claims.get("scope_boundary", "Not specified")
    return JUDGE_TEMPLATE.format(
        concepts=concepts,
        relations=relations,
        scope=scope,
        vlm_output=vlm_output[:1200],
    )


def parse_scores(text):
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$',     '', text)
    text = re.sub(r'^```\s*',     '', text)
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        try:
            return json.loads(text)
        except Exception:
            return None


def normalize_scores(scores):
    if not isinstance(scores, dict):
        return None

    # Accept a few plausible alias keys from imperfect judge outputs
    alias_map = {
        "cr": "concept_recall",
        "rv": "relational_validity",
        "sf": "scope_fidelity",
        "scope": "scope_fidelity",
        "scope_score": "scope_fidelity",
        "concepts_recall": "concept_recall",
        "relation_validity": "relational_validity",
    }
    for old_key, new_key in alias_map.items():
        if new_key not in scores and old_key in scores:
            scores[new_key] = scores[old_key]

    scores.setdefault("concept_recall", 0.0)
    scores.setdefault("relational_validity", 0.0)
    scores.setdefault("scope_fidelity", 0.0)
    scores.setdefault("cr_justification", "")
    scores.setdefault("rv_justification", "")
    scores.setdefault("sf_justification", "")
    scores.setdefault("missed_concepts", [])
    scores.setdefault("hallucinated_concepts", [])

    try:
        scores["concept_recall"] = float(scores["concept_recall"])
        scores["relational_validity"] = float(scores["relational_validity"])
        scores["scope_fidelity"] = float(scores["scope_fidelity"])
    except Exception:
        return None

    # Optional clipping for safety
    scores["concept_recall"] = max(0.0, min(1.0, scores["concept_recall"]))
    scores["relational_validity"] = max(0.0, min(1.0, scores["relational_validity"]))
    scores["scope_fidelity"] = max(0.0, min(1.0, scores["scope_fidelity"]))

    if not isinstance(scores["missed_concepts"], list):
        scores["missed_concepts"] = [str(scores["missed_concepts"])]
    if not isinstance(scores["hallucinated_concepts"], list):
        scores["hallucinated_concepts"] = [str(scores["hallucinated_concepts"])]

    return scores


def compute_igs(s):
    cr = float(s.get("concept_recall",      0))
    rv = float(s.get("relational_validity", 0))
    sf = float(s.get("scope_fidelity",      0))
    return round(WEIGHT_CR * cr + WEIGHT_RV * rv + WEIGHT_SF * sf, 4)


def call_claude(client, user_prompt, judge_model, max_tokens=600):
    wait = RETRY_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model       = judge_model,
                max_tokens  = max_tokens,
                temperature = 0.0,
                system      = JUDGE_SYSTEM,
                messages    = [{"role": "user", "content": user_prompt}],
            )
            parts = [
                b.text for b in resp.content
                if getattr(b, "type", None) == "text"
            ]
            return "\n".join(parts) if parts else None
        except Exception as exc:
            err = str(exc)
            if attempt < MAX_RETRIES and any(
                k in err.lower()
                for k in ("rate", "limit", "timeout", "overload", "500", "502", "503", "529")
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
    for d in [CLAIMS_V5, CLAIMS_V4]:
        f = Path(d) / f"{sid}.json"
        if f.exists():
            with open(f) as fh:
                return json.load(fh), d
    return None, None


def save_json(data, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


def score_model(model_name, split, client, judge_model,
                resume=False, excluded_slides=None):
    excluded_slides = excluded_slides or set()
    scores_dir = Path(SCORES_CLAUDE)
    scores_dir.mkdir(parents=True, exist_ok=True)

    vlm_file = Path(VLM_OUT_DIR) / f"{model_name}_{split}.json"
    if not vlm_file.exists():
        print(f"  ❌ VLM output not found: {vlm_file}")
        return None

    with open(vlm_file) as f:
        vlm_outputs = json.load(f)

    out_file = scores_dir / f"{model_name}_{split}_scores.json"

    existing = {}
    if resume and out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
        print(f"  ↩  Resume: {len(existing)} slides already scored, skipping.")

    if excluded_slides:
        vlm_outputs = {
            sid: v for sid, v in vlm_outputs.items()
            if sid not in excluded_slides
        }

    print(f"\n{'='*55}")
    print(f"  SCORING: {model_name} | {split}")
    print(f"  Judge:   {judge_model} (Anthropic API)")
    print(f"  Slides:  {len(vlm_outputs)}")
    print(f"  Claims:  claims_v5/ (fallback: claims_v4/)")
    print(f"  Output:  igs_scores_claude_v5/")
    print(f"{'='*55}")

    all_scores = dict(existing)
    n_ok = n_fail = n_skip = n_fallback = 0

    for idx, (sid, out_data) in enumerate(vlm_outputs.items(), 1):
        if sid in existing:
            n_skip += 1
            continue

        claims, claims_src = load_claims(sid)
        if claims is None:
            print(f"  ⚠  {sid}: no claims file — skipping")
            n_skip += 1
            continue

        if claims_src == CLAIMS_V4:
            n_fallback += 1

        vlm_text = out_data.get("response", "")
        prompt   = build_prompt(claims, vlm_text)

        raw = call_claude(client, prompt, judge_model)
        if raw is None:
            print(f"  ❌ {sid}: API failed after {MAX_RETRIES} retries")
            n_fail += 1
            continue

        scores = parse_scores(raw)
        if scores is None:
            print(f"  ❌ {sid}: JSON parse failed — raw: {raw[:80]}")
            n_fail += 1
            continue

        scores = normalize_scores(scores)
        if scores is None:
            print(f"  ❌ {sid}: invalid or non-numeric score fields — raw: {raw[:200]}")
            n_fail += 1
            continue

        scores["igs"]              = compute_igs(scores)
        scores["slide_id"]         = sid
        scores["judge_model"]      = judge_model
        scores["judge_family"]     = "anthropic"
        scores["lecture_category"] = out_data.get("lecture_category", "")
        scores["content_type"]     = out_data.get("content_type", "")
        scores["modality"]         = out_data.get("modality", "")

        all_scores[sid] = scores
        n_ok += 1

        print(
            f"  ✅ [{idx:>3}/{len(vlm_outputs)}] {sid:<16} "
            f"IGS={scores.get('igs', 0.0):.3f}  "
            f"CR={scores.get('concept_recall', 0.0):.2f}  "
            f"RV={scores.get('relational_validity', 0.0):.2f}  "
            f"SF={scores.get('scope_fidelity', 0.0):.2f}"
        )

        if n_ok % 10 == 0:
            save_json(all_scores, out_file)

        time.sleep(SLIDE_DELAY)

    save_json(all_scores, out_file)

    if all_scores:
        n   = len(all_scores)
        igs = sum(float(s.get("igs", 0.0)) for s in all_scores.values()) / n
        cr  = sum(float(s.get("concept_recall", 0.0)) for s in all_scores.values()) / n
        rv  = sum(float(s.get("relational_validity", 0.0)) for s in all_scores.values()) / n
        sf  = sum(float(s.get("scope_fidelity", 0.0)) for s in all_scores.values()) / n
        nz  = sum(1 for s in all_scores.values() if float(s.get("igs", 0.0)) == 0.0)

        print(f"\n  V5 claims used : {n - n_fallback}")
        print(f"  V4 fallback    : {n_fallback}")
        print(f"  Skipped        : {n_skip}")
        print(f"  Parse failed   : {n_fail}")
        print(f"\n  RESULT: IGS={igs:.3f}  CR={cr:.3f}  RV={rv:.3f}  SF={sf:.3f}  "
              f"n={n}  zeros={nz} ({nz/n*100:.0f}%)")
        print(f"  Saved → {out_file}")

    return all_scores


def compare_gpt4o_vs_claude(model_name, split):
    gpt_file    = Path(SCORES_GPT4O)  / f"{model_name}_{split}_scores.json"
    claude_file = Path(SCORES_CLAUDE) / f"{model_name}_{split}_scores.json"

    if not gpt_file.exists():
        print(f"  ⚠  GPT-4o scores not found: {gpt_file}")
        return
    if not claude_file.exists():
        print(f"  ⚠  Claude scores not found: {claude_file}")
        return

    gpt = json.load(open(gpt_file))
    cla = json.load(open(claude_file))
    shared = sorted(set(gpt) & set(cla))
    mean = lambda vals: sum(vals) / len(vals) if vals else 0.0

    print(f"\n{'='*55}")
    print(f"  GPT-4o vs CLAUDE — {MODEL_LABELS.get(model_name, model_name)}")
    print(f"  {len(shared)} shared slides")
    print(f"{'='*55}")
    print(f"  {'Metric':<8} {'GPT4o':>8} {'Claude':>8} {'Δ':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for metric, key in [
        ("IGS", "igs"),
        ("CR",  "concept_recall"),
        ("RV",  "relational_validity"),
        ("SF",  "scope_fidelity"),
    ]:
        g_m = mean([float(gpt[s].get(key, 0.0)) for s in shared])
        c_m = mean([float(cla[s].get(key, 0.0)) for s in shared])
        delta = c_m - g_m
        flag  = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")
        print(f"  {metric:<8} {g_m:>8.3f} {c_m:>8.3f} {delta:>+8.3f} {flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score VLM outputs with Anthropic Claude as judge."
    )
    parser.add_argument("--model", default="all")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument(
        "--judge", default="claude-opus-4-6",
        help="Anthropic model ID. Default: claude-opus-4-6. "
             "Alt: claude-sonnet-4-6, claude-haiku-4-5-20251001"
    )
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--no_exclude", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "No Anthropic API key found.\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    excluded_slides = set()
    if args.no_exclude:
        print("  --no_exclude set: scoring all slides.")
    else:
        exclude_path = None
        if args.exclude:
            candidate = Path(args.exclude)
            if candidate.exists():
                exclude_path = candidate
            else:
                fallback = Path(BENCH_DIR) / candidate.name
                if fallback.exists():
                    exclude_path = fallback
                else:
                    raise SystemExit(f"Exclusion file not found: {args.exclude}")
        else:
            default = Path(DEFAULT_EXCLUDE)
            if default.exists():
                exclude_path = default
                print(f"  ℹ  Auto-loading default exclusion: {exclude_path}")

        if exclude_path is not None:
            excluded_slides = set(json.load(open(exclude_path)))
            print(f"  ✅ Excluding {len(excluded_slides)} slides.")

    client = Anthropic(api_key=api_key)

    print(f"\n  Testing Anthropic API ({args.judge})…")
    try:
        test = client.messages.create(
            model       = args.judge,
            max_tokens  = 10,
            messages    = [{"role": "user", "content": "Reply with one word: ready"}],
        )
        reply = "".join(
            b.text for b in test.content
            if getattr(b, "type", None) == "text"
        ).strip()
        print(f"  ✅ API OK — response: '{reply}'\n")
    except Exception as e:
        raise SystemExit(f"  ✗  API test failed: {e}")

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
            compare_gpt4o_vs_claude(model_name, args.split)

    print(f"\n{'='*55}")
    print(f"  ALL DONE  →  {SCORES_CLAUDE}/")
    print(f"  Next: python step16_multi_judge_agreement.py")
    print(f"{'='*55}")
