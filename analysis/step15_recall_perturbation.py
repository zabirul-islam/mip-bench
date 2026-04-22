#!/usr/bin/env python3
"""
step15_recall_perturbation.py

Reviewer defense: show that IGS model rankings are robust to extraction
recall degradation. Synthetically drop a fraction of gold concepts from
each slide's claims, re-run the judge logic on the *already scored*
missed_concepts / hallucinated_concepts fields, and show ranking stability.

Method:
  Judge already recorded per-slide CR/RV/SF. We do NOT re-call the API.
  Instead, we simulate "lower-recall extraction" by:
    (a) randomly pruning a fraction p of the gold key_concepts,
    (b) recomputing CR as the fraction of *remaining* gold concepts the
        VLM recovered (using the judge's missed_concepts list),
    (c) RV and SF stay fixed (they do not depend on concept count
        enumeration in the same way; lower-bound estimate).
  Final IGS = 0.40*CR' + 0.35*RV + 0.25*SF.

For each drop rate p in {0.10, 0.20, 0.30}, we bootstrap over 500 random
prunings per slide and report:
  - mean IGS per model
  - Spearman rank correlation of model ranking vs unperturbed
  - max rank flip

Output: medlecture_bench/recall_perturbation_results.json + stdout table.

Usage:
  python step15_recall_perturbation.py
  python step15_recall_perturbation.py --bench_dir /path/to/medlecture_bench
"""

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import spearmanr

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

W_CR, W_RV, W_SF = 0.40, 0.35, 0.25
DROP_RATES = [0.10, 0.20, 0.30]
N_BOOT = 500
SEED = 42


_CLAIMS_CACHE = {}

def load_claims(claims_dir, sid):
    key = (str(claims_dir), sid)
    if key in _CLAIMS_CACHE:
        return _CLAIMS_CACHE[key]
    f = Path(claims_dir) / f"{sid}.json"
    out = json.load(open(f)) if f.exists() else None
    _CLAIMS_CACHE[key] = out
    return out


def compute_perturbed_cr(gold_concepts, missed_concepts, drop_rate, rng):
    """CR after randomly dropping `drop_rate` fraction of gold concepts.

    Idea: if extractor had captured fewer gold concepts, the VLM would be
    scored only against the retained subset. Concepts missed by the VLM
    that fall into the dropped subset no longer hurt CR.
    """
    if not gold_concepts:
        return None
    n_drop = int(round(len(gold_concepts) * drop_rate))
    if n_drop >= len(gold_concepts):
        n_drop = len(gold_concepts) - 1
    if n_drop < 0:
        n_drop = 0
    dropped = set(rng.sample(gold_concepts, n_drop))
    retained = [c for c in gold_concepts if c not in dropped]
    if not retained:
        return None
    # Judge missed_concepts is a list of human-readable strings.
    # Count how many of the retained golds were missed by VLM.
    missed_lower = {str(m).strip().lower() for m in (missed_concepts or [])}
    n_missed_retained = sum(
        1 for c in retained
        if str(c).strip().lower() in missed_lower
    )
    # CR = fraction of retained concepts the VLM addressed.
    cr_frac = 1.0 - n_missed_retained / len(retained)
    # Snap to judge's ordinal anchors {0, 0.3, 0.5, 0.7, 1.0}
    anchors = [0.0, 0.3, 0.5, 0.7, 1.0]
    return min(anchors, key=lambda a: abs(a - cr_frac))


def load_scores(scores_dir, model):
    f = Path(scores_dir) / f"{model}_test_scores.json"
    if not f.exists():
        raise FileNotFoundError(f)
    return json.load(open(f))


def baseline_mean_igs(scores):
    vals = [s["igs"] for s in scores.values()]
    return mean(vals)


def perturb_one(scores, claims_dir, drop_rate, rng):
    """Return mean IGS after one perturbation realization."""
    per_slide = []
    for sid, s in scores.items():
        claims = load_claims(claims_dir, sid)
        if claims is None:
            per_slide.append(s["igs"])
            continue
        gold = claims.get("key_concepts", [])
        cr_prime = compute_perturbed_cr(
            gold_concepts=gold,
            missed_concepts=s.get("missed_concepts", []),
            drop_rate=drop_rate,
            rng=rng,
        )
        if cr_prime is None:
            per_slide.append(s["igs"])
            continue
        igs_prime = W_CR * cr_prime + W_RV * s["relational_validity"] \
            + W_SF * s["scope_fidelity"]
        per_slide.append(igs_prime)
    return mean(per_slide)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench_dir",
        default="/sessions/peaceful-hopeful-hypatia/mnt/MedLecture/medlecture_bench",
    )
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument(
        "--output",
        default="/sessions/peaceful-hopeful-hypatia/mnt/MedLecture/medlecture_bench/recall_perturbation_results.json",
    )
    args = ap.parse_args()

    bench = Path(args.bench_dir)
    scores_dir = bench / "igs_scores_gpt4o_v5"
    claims_dir = bench / "claims_v5"

    # Load baseline
    baseline = {}
    raw_scores = {}
    for m in MODELS:
        sc = load_scores(scores_dir, m)
        raw_scores[m] = sc
        baseline[m] = baseline_mean_igs(sc)

    base_rank = sorted(MODELS, key=lambda m: -baseline[m])

    print(f"\n{'='*70}")
    print(f"  BASELINE IGS")
    print(f"{'='*70}")
    for m in base_rank:
        print(f"  {MODEL_LABELS[m]:<20} IGS = {baseline[m]:.4f}")

    results = {"baseline": {m: baseline[m] for m in MODELS},
               "perturbations": {}}

    rng = random.Random(SEED)

    for p in DROP_RATES:
        print(f"\n{'='*70}")
        print(f"  DROP RATE: {p*100:.0f}%  (N_BOOT = {args.n_boot})")
        print(f"{'='*70}")

        # bootstrap: for each boot, get mean IGS per model
        per_boot = {m: [] for m in MODELS}
        spearmans = []
        rank_flips = []

        for b in range(args.n_boot):
            boot_rng = random.Random(rng.randint(0, 1_000_000_000))
            igs_b = {}
            for m in MODELS:
                igs_b[m] = perturb_one(raw_scores[m], claims_dir, p, boot_rng)
                per_boot[m].append(igs_b[m])
            pert_rank = sorted(MODELS, key=lambda m: -igs_b[m])
            rho, _ = spearmanr(
                [base_rank.index(m) for m in MODELS],
                [pert_rank.index(m) for m in MODELS],
            )
            spearmans.append(rho)
            max_flip = max(
                abs(base_rank.index(m) - pert_rank.index(m)) for m in MODELS
            )
            rank_flips.append(max_flip)

        print(f"  {'Model':<20} {'Baseline':>10} {'Pert μ':>10} {'Pert σ':>8} {'Δ':>8}")
        p_row = {}
        for m in base_rank:
            mu = float(np.mean(per_boot[m]))
            sd = float(np.std(per_boot[m]))
            delta = mu - baseline[m]
            p_row[m] = {"mean": mu, "std": sd, "delta": delta}
            print(f"  {MODEL_LABELS[m]:<20} {baseline[m]:>10.4f} "
                  f"{mu:>10.4f} {sd:>8.4f} {delta:>+8.4f}")

        rho_mean = float(np.mean(spearmans))
        rho_min = float(np.min(spearmans))
        max_flip_overall = int(max(rank_flips))

        print(f"\n  Rank Spearman ρ : mean={rho_mean:.4f}, min={rho_min:.4f}")
        print(f"  Max rank flip   : {max_flip_overall}")

        results["perturbations"][str(p)] = {
            "models": p_row,
            "rank_spearman_mean": rho_mean,
            "rank_spearman_min": rho_min,
            "max_rank_flip": max_flip_overall,
            "n_boot": args.n_boot,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"\n{'='*70}")
    print(f"  Saved → {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
