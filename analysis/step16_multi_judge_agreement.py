#!/usr/bin/env python3
"""
step16_multi_judge_agreement.py

Multi-judge ensemble analysis: GPT-4o vs Claude as IGS judge.

For each model m and each dimension d in {CR, RV, SF, IGS}:
  - Pearson r between per-slide (GPT-4o_m,d, Claude_m,d)
  - Spearman ρ between per-slide rankings
  - Cohen's quadratic-weighted κ on ordinal dim scores
  - Mean absolute deviation between judges
  - Ensemble mean IGS per model and bootstrap 95% CI

Also reports:
  - Per-model ranking under each judge individually
  - Ranking agreement (top-K overlap, Kendall τ)

Usage:
  python step16_multi_judge_agreement.py
  python step16_multi_judge_agreement.py --bench_dir /path/to/medlecture_bench
"""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

try:
    from sklearn.metrics import cohen_kappa_score
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

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
DIMS = [
    ("CR",  "concept_recall"),
    ("RV",  "relational_validity"),
    ("SF",  "scope_fidelity"),
    ("IGS", "igs"),
]
ANCHORS = [0.0, 0.3, 0.5, 0.7, 1.0]
W_CR, W_RV, W_SF = 0.40, 0.35, 0.25


def load(scores_dir, model):
    f = Path(scores_dir) / f"{model}_test_scores.json"
    if not f.exists():
        return None
    return json.load(open(f))


def anchor(x):
    return min(ANCHORS, key=lambda a: abs(a - float(x)))


def bootstrap_ci(vals, n=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(42)
    vals = np.asarray(vals)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(n)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1 - alpha/2)])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench_dir",
        default="/home/islamm11/islamm11/MedLecture/medlecture_bench",
    )
    ap.add_argument(
        "--output",
        default=None,
    )
    args = ap.parse_args()

    bench = Path(args.bench_dir)
    gpt_dir = bench / "igs_scores_gpt4o_v5"
    claude_dir = bench / "igs_scores_claude_v5"

    out = {"per_model": {}, "ensemble": {}, "judge_ranking": {}}

    print(f"\n{'='*78}")
    print("  MULTI-JUDGE AGREEMENT: GPT-4o vs Claude")
    print(f"{'='*78}")

    ensemble_means = {}
    gpt_means = {}
    claude_means = {}

    for m in MODELS:
        g = load(gpt_dir, m)
        c = load(claude_dir, m)
        if g is None:
            print(f"  ⚠  GPT-4o scores missing for {m}")
            continue
        if c is None:
            print(f"  ⚠  Claude scores missing for {m} — skip")
            continue
        shared = sorted(set(g) & set(c))
        if not shared:
            print(f"  ⚠  no shared slides for {m}")
            continue

        print(f"\n  {MODEL_LABELS[m]}   (n shared = {len(shared)})")
        print(f"  {'Dim':<4} {'r':>7} {'ρ':>7} {'κq':>7} {'MAD':>7} {'μ_GPT':>7} {'μ_Cla':>7}")
        print(f"  {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        model_row = {}
        for dim_name, key in DIMS:
            gv = np.array([float(g[s][key]) for s in shared])
            cv = np.array([float(c[s][key]) for s in shared])
            try:
                r, _  = pearsonr(gv, cv)
            except Exception:
                r = float("nan")
            try:
                rho, _ = spearmanr(gv, cv)
            except Exception:
                rho = float("nan")
            mad = float(np.mean(np.abs(gv - cv)))
            mu_g = float(gv.mean())
            mu_c = float(cv.mean())

            if HAVE_SK and dim_name in ("CR", "RV", "SF"):
                g_anchored = [anchor(x) for x in gv]
                c_anchored = [anchor(x) for x in cv]
                try:
                    kq = cohen_kappa_score(
                        g_anchored, c_anchored,
                        labels=ANCHORS, weights="quadratic",
                    )
                except Exception:
                    kq = float("nan")
            else:
                kq = float("nan")

            print(f"  {dim_name:<4} {r:>7.3f} {rho:>7.3f} "
                  f"{kq:>7.3f} {mad:>7.3f} {mu_g:>7.3f} {mu_c:>7.3f}")
            model_row[dim_name] = dict(
                pearson_r=float(r), spearman_rho=float(rho),
                cohen_kappa_quadratic=float(kq), mad=mad,
                mu_gpt4o=mu_g, mu_claude=mu_c,
            )

        out["per_model"][m] = model_row
        gpt_means[m]    = float(np.mean([g[s]["igs"] for s in shared]))
        claude_means[m] = float(np.mean([c[s]["igs"] for s in shared]))

        ensemble_per_slide = [
            (g[s]["igs"] + c[s]["igs"]) / 2.0 for s in shared
        ]
        mu = float(np.mean(ensemble_per_slide))
        lo, hi = bootstrap_ci(ensemble_per_slide)
        ensemble_means[m] = {"mean": mu, "ci95": [lo, hi],
                             "n": len(shared)}

    # Ranking comparison
    print(f"\n{'='*78}")
    print("  MODEL RANKINGS UNDER EACH JUDGE")
    print(f"{'='*78}")
    header = f"  {'Model':<20} {'GPT-4o IGS':>12} {'Claude IGS':>12} {'Ensemble':>10} {'95% CI':>22}"
    print(header)
    rank_g  = sorted(gpt_means,    key=lambda m: -gpt_means[m])
    for m in rank_g:
        e = ensemble_means.get(m, {})
        ci = e.get("ci95", [float("nan"), float("nan")])
        print(f"  {MODEL_LABELS[m]:<20} {gpt_means[m]:>12.4f} "
              f"{claude_means[m]:>12.4f} {e.get('mean', float('nan')):>10.4f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]")

    # Kendall τ between rankings
    models_common = [m for m in MODELS if m in gpt_means and m in claude_means]
    if len(models_common) >= 3:
        g_r = [sorted(models_common, key=lambda m: -gpt_means[m]).index(m)
               for m in models_common]
        c_r = [sorted(models_common, key=lambda m: -claude_means[m]).index(m)
               for m in models_common]
        tau, p = kendalltau(g_r, c_r)
        print(f"\n  Kendall τ between judge rankings: {tau:.4f} (p={p:.4f})")
        out["judge_ranking"] = {
            "kendall_tau": float(tau), "kendall_p": float(p),
            "gpt4o": {m: gpt_means[m] for m in models_common},
            "claude": {m: claude_means[m] for m in models_common},
        }

    out["ensemble"] = ensemble_means

    output_path = args.output or str(bench / "multi_judge_agreement.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(output_path, "w"), indent=2)
    print(f"\n  Saved → {output_path}\n")


if __name__ == "__main__":
    main()

