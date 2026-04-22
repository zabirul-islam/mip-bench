#!/usr/bin/env python3
"""
step18_transfer_probe.py

Pipeline Transfer Probe: run the MIP-Bench extraction pipeline
UNCHANGED on a non-medical lecture corpus. Existence proof for
generalizability. No human labels required. No IGS reported.

Input expectation: a directory with paired (slide PNG, narration TXT)
  <probe_dir>/slides/Slide01.png
  <probe_dir>/slides/Slide02.png
  ...
  <probe_dir>/texts/Slide01.txt
  <probe_dir>/texts/Slide02.txt
  ...

Output:
  <probe_dir>/claims/Slide01.json   (same schema as claims_v5)
  <probe_dir>/transfer_probe_stats.json  (counts + failure modes)

Usage:
  export OPENAI_API_KEY=sk-...
  python step18_transfer_probe.py --probe_dir /path/to/mit_ocw_fourier

Good public probe corpora:
  - MIT OCW 18.S096 Introduction to Fourier Analysis (slides + transcripts)
  - Stanford CS229 Lecture Notes (slides, transcripts via YouTube caption)
  - 3Blue1Brown essence-of-* series (visuals + transcripts on YouTube)

We ship this as a reviewer-facing demonstration, not as a second
benchmark. No IGS numbers are reported for the probe corpus.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("pip install openai")


EXTRACT_SYSTEM = (
    "You are an expert at extracting instructional intent from "
    "aligned lecture narration. Always respond with valid JSON only."
)

EXTRACT_TEMPLATE = """You will be given the instructor's narration for a single lecture slide.
Extract a structured representation of what the slide is designed to teach.

Return STRICT JSON with this exact shape:
{{
  "key_concepts": ["concept 1", "concept 2", "..."],
  "relations":    ["A -> B", "C -> D", "..."],
  "scope_boundary": "1-2 sentence description of what this slide covers and does NOT cover"
}}

Rules:
- "key_concepts": 3-8 central concepts the slide teaches. Exclude peripheral mentions.
- "relations":   directed A -> B strings capturing the intended conceptual links.
                 If there are no explicit relations, return [].
- "scope_boundary": short natural-language description grounded in the narration.

NARRATION:
{narration}
"""


def call(client, narration, model="gpt-4o", tries=4):
    wait = 2
    for att in range(1, tries + 1):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=700,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user",   "content":
                        EXTRACT_TEMPLATE.format(narration=narration[:4000])},
                ],
            )
            return r.choices[0].message.content
        except Exception as e:
            if att == tries: raise
            time.sleep(wait); wait *= 2


def parse(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1:
        text = text[a:b + 1]
    try:
        return json.loads(text)
    except Exception:
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(text)


def validate(d):
    """Schema check. Return ('ok', d) or (reason, None)."""
    if not isinstance(d, dict):
        return "not-a-dict", None
    kc = d.get("key_concepts")
    rel = d.get("relations")
    sb = d.get("scope_boundary")
    if not isinstance(kc, list) or not kc:
        return "empty-concepts", None
    if not isinstance(rel, list):
        rel = []
    if not isinstance(sb, str) or len(sb.strip()) < 10:
        return "missing-scope", None
    return "ok", {"key_concepts": kc, "relations": rel, "scope_boundary": sb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    probe = Path(args.probe_dir)
    texts_dir  = probe / "texts"
    claims_dir = probe / "claims"
    claims_dir.mkdir(exist_ok=True, parents=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    ok = 0; empty = 0; fail = 0
    n_concepts, n_relations, scope_cov = [], [], 0
    per_slide = {}

    text_files = sorted(texts_dir.glob("*.txt"))
    print(f"\n  Probe corpus: {probe}")
    print(f"  Slides found: {len(text_files)}")
    print(f"  Using model:  {args.model}\n")

    for tf in text_files:
        sid = tf.stem
        narration = tf.read_text(encoding="utf-8", errors="ignore").strip()
        if not narration or len(narration) < 40:
            print(f"  ⊘ {sid}: narration too short, skip")
            empty += 1; per_slide[sid] = {"status": "empty-narration"}
            continue
        try:
            raw = call(client, narration, model=args.model)
            parsed = parse(raw)
            status, d = validate(parsed)
        except Exception as e:
            status = f"error:{type(e).__name__}"; d = None

        if status == "ok":
            json.dump(d, open(claims_dir / f"{sid}.json", "w"),
                      indent=2, ensure_ascii=False)
            ok += 1
            n_concepts.append(len(d["key_concepts"]))
            n_relations.append(len(d["relations"]))
            if d["scope_boundary"].strip():
                scope_cov += 1
            per_slide[sid] = {
                "status": "ok",
                "n_concepts": len(d["key_concepts"]),
                "n_relations": len(d["relations"]),
            }
            print(f"  ✓ {sid}: {len(d['key_concepts'])} concepts, "
                  f"{len(d['relations'])} relations")
        else:
            fail += 1
            per_slide[sid] = {"status": status}
            print(f"  ✗ {sid}: {status}")

    stats = {
        "total": len(text_files),
        "ok": ok, "empty": empty, "failed": fail,
        "ok_rate": ok / max(1, len(text_files)),
        "mean_concepts_per_slide":  float(sum(n_concepts) / max(1, len(n_concepts))),
        "mean_relations_per_slide": float(sum(n_relations) / max(1, len(n_relations))),
        "scope_coverage":           scope_cov / max(1, ok),
        "per_slide":                per_slide,
    }
    json.dump(stats, open(probe / "transfer_probe_stats.json", "w"), indent=2)

    print(f"\n  RESULT: ok={ok}/{len(text_files)} "
          f"({stats['ok_rate']*100:.0f}%)   "
          f"mean concepts={stats['mean_concepts_per_slide']:.1f}   "
          f"mean relations={stats['mean_relations_per_slide']:.1f}   "
          f"scope coverage={stats['scope_coverage']*100:.0f}%")
    print(f"  Saved stats → {probe}/transfer_probe_stats.json")


if __name__ == "__main__":
    main()
