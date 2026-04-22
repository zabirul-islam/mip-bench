#!/usr/bin/env python3
"""
step17_judge_bias_ablation.py

Judge-family bias check.

Worry: because the IGS judge is GPT-4o and the extractor is GPT-4o,
reviewers may argue the benchmark implicitly favours GPT-family
models. We cannot include GPT-4V as an evaluated model (closed
weights outside our release policy), but we *can* test the weaker
sufficient condition: the GPT-4o judge should not systematically
reward responses that paraphrase the GPT-4o-authored concept graph
more than responses that paraphrase the instructor narration.

Design:
  For each slide, compute:
    - concept_overlap(ŷ, C)     : Jaccard(tokens in ŷ,
                                           tokens in GPT-4o C)
    - narration_overlap(ŷ, N)   : Jaccard(tokens in ŷ,
                                           tokens in narration)
  Then fit OLS:
      IGS ~ β0 + β_C · concept_overlap + β_N · narration_overlap
  The worry is that β_C >> β_N. We report both β's with 95% CIs and
  partial R² contributions. A small β_C - β_N (or β_N >= β_C) means
  the judge is not preferentially rewarding extractor-family
  vocabulary.

Usage:
  python step17_judge_bias_ablation.py
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


MODELS = [
    "qwen2vl_7b",
    "internvl2_8b",
    "internvl2_26b",
    "internvl2_40b",
    "llava_next_34b",
]

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']+")

def toks(s):
    return set(t.lower() for t in _TOKEN_RE.findall(s or ""))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench_dir",
        default="/sessions/peaceful-hopeful-hypatia/mnt/MedLecture/medlecture_bench",
    )
    ap.add_argument(
        "--vlm_dir",
        default="/sessions/peaceful-hopeful-hypatia/mnt/MedLecture/medlecture_bench/vlm_outputs",
    )
    ap.add_argument(
        "--scores_dir",
        default="/sessions/peaceful-hopeful-hypatia/mnt/MedLecture/medlecture_bench/igs_scores_gpt4o_v5",
    )
    ap.add_argument(
        "--claims_dir",
        default="/sessions/peaceful-hopeful-hypatia/mnt/MedLecture/medlecture_bench/claims_v5",
    )
    ap.add_argument(
        "--test_json",
        default="/sessions/peaceful-hopeful-hypatia/mnt/MedLecture/medlecture_bench/test.json",
    )
    args = ap.parse_args()

    # Load teacher narration per slide
    test = json.load(open(args.test_json))
    narration = {t["id"]: t.get("teacher_text", "") for t in test}

    # Load claims per slide → concept string
    claims_tokens = {}
    claims_dir = Path(args.claims_dir)
    skipped = 0
    for cf in claims_dir.glob("*.json"):
        try:
            d = json.load(open(cf))
        except Exception:
            skipped += 1
            continue
        if isinstance(d, list):
            # some older claim files are lists of strings
            concepts = [str(x) for x in d]
        elif isinstance(d, dict):
            concepts = d.get("key_concepts", []) or []
        else:
            skipped += 1
            continue
        if not isinstance(concepts, list):
            concepts = []
        claims_tokens[cf.stem] = toks(" ".join(str(c) for c in concepts))
    if skipped:
        print(f"  ℹ  {skipped} malformed claims files skipped")

    print(f"\n{'='*78}")
    print("  JUDGE-BIAS ABLATION  (per model, OLS: IGS ~ concept + narration)")
    print(f"{'='*78}\n")

    all_rows = []
    per_model = {}

    for m in MODELS:
        vlm_path = Path(args.vlm_dir) / f"{m}_test.json"
        sc_path  = Path(args.scores_dir) / f"{m}_test_scores.json"
        if not vlm_path.exists() or not sc_path.exists():
            print(f"  ⚠  missing files for {m}"); continue
        vlm = json.load(open(vlm_path))
        sc  = json.load(open(sc_path))

        X, y = [], []
        for sid, s in sc.items():
            resp = (vlm.get(sid) or {}).get("response", "")
            c_tok = claims_tokens.get(sid)
            n_tok = toks(narration.get(sid, ""))
            r_tok = toks(resp)
            if not r_tok or c_tok is None:
                continue
            c_ov = jaccard(r_tok, c_tok)
            n_ov = jaccard(r_tok, n_tok)
            X.append([1.0, c_ov, n_ov])
            y.append(float(s["igs"]))

        X = np.asarray(X); y = np.asarray(y)
        if len(y) < 30:
            print(f"  ⚠  {m}: too few rows ({len(y)})"); continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        resid = y - y_hat
        sigma2 = (resid @ resid) / max(1, len(y) - 3)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
        ci = [(float(b - 1.96*s), float(b + 1.96*s))
              for b, s in zip(beta, se)]

        ss_tot = ((y - y.mean())**2).sum()
        ss_res = (resid**2).sum()
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        per_model[m] = {
            "n": len(y),
            "beta_intercept": float(beta[0]),
            "beta_concept":   float(beta[1]),
            "beta_narration": float(beta[2]),
            "ci_concept":     ci[1],
            "ci_narration":   ci[2],
            "r2":             float(r2),
        }
        all_rows.append({
            "model": m,
            "n": len(y),
            "β_concept":  beta[1],
            "β_narration": beta[2],
            "Δβ":          beta[1] - beta[2],
            "R²":          r2,
        })

        print(f"  {m:<18}  n={len(y):>3}   "
              f"β_concept = {beta[1]:+.3f}  [{ci[1][0]:+.2f}, {ci[1][1]:+.2f}]   "
              f"β_narration = {beta[2]:+.3f}  [{ci[2][0]:+.2f}, {ci[2][1]:+.2f}]   "
              f"Δ = {beta[1]-beta[2]:+.3f}   R²={r2:.3f}")

    print(f"\n{'='*78}")
    print("  INTERPRETATION")
    print(f"{'='*78}")
    print("  If β_concept ≫ β_narration for every model, the judge is rewarding")
    print("  GPT-4o-extractor vocabulary preferentially. In practice we expect")
    print("  Δ small or near zero; β_narration often comparable because the")
    print("  judge scores against the concept graph AND can identify concepts")
    print("  via paraphrase.")

    Path("/sessions/peaceful-hopeful-hypatia/mnt/ALIVE-slide/judge_bias_ablation.json"
         ).parent.mkdir(exist_ok=True, parents=True)
    json.dump(per_model,
              open("/sessions/peaceful-hopeful-hypatia/mnt/ALIVE-slide/judge_bias_ablation.json",
                   "w"),
              indent=2)
    print("\n  Saved → judge_bias_ablation.json")


if __name__ == "__main__":
    main()
