#!/usr/bin/env python3
"""
step10_gpt4o.py
────────────────────────────────────────────────────────────────────────────
Same as step10_final_results.py but reads from igs_scores_gpt4o/ instead
of igs_scores_v3/.

Run from: ~/islamm11/MedLecture/
Usage:
  python step10_gpt4o.py                  # GPT-4o scores only
  python step10_gpt4o.py --compare        # also show Qwen v3 vs GPT-4o diff
"""

import json, csv
from pathlib import Path
from collections import defaultdict

# ─── Paths ────────────────────────────────────────────────────
SCORES_V3     = "medlecture_bench/igs_scores_v4"
SCORES_GPT4O  = "medlecture_bench/igs_scores_gpt4o_v5"
DECISIONS_CSV = "medlecture_bench/manual_review/review_decisions.csv"
# ──────────────────────────────────────────────────────────────

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

TOTAL_TEST = 211   # test set size before exclusions


def load_decisions():
    p = Path(DECISIONS_CSV)
    if not p.exists():
        print(f"  [review_decisions.csv not found — running without exclusions]")
        return {}
    decisions = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            sid = row.get("slide_id", "").strip()
            dec = row.get("decision", "").strip()
            if sid and dec:
                decisions[sid] = dec
    return decisions


def load(scores_dir, model, split="test"):
    f = Path(scores_dir) / f"{model}_{split}_scores.json"
    return json.load(open(f)) if f.exists() else {}


def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def stats(scores, keep=None):
    sids = keep if keep is not None else list(scores.keys())
    vals = [scores[s] for s in sids if s in scores]
    if not vals:
        return None
    n = len(vals)
    return dict(
        igs    = mean([v["igs"]                 for v in vals]),
        cr     = mean([v["concept_recall"]      for v in vals]),
        rv     = mean([v["relational_validity"] for v in vals]),
        sf     = mean([v["scope_fidelity"]      for v in vals]),
        n      = n,
        n_zero = sum(1 for v in vals if v["igs"] == 0.0),
    )


def fmt(s):
    if not s:
        return "     —      —      —      —     —"
    return (f" {s['igs']:>6.3f} {s['cr']:>6.3f} {s['rv']:>6.3f} "
            f"{s['sf']:>6.3f} {s['n']:>4}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true",
                        help="Also print Qwen v3 vs GPT-4o comparison table.")
    args = parser.parse_args()

    decisions = load_decisions()
    B    = {s for s, d in decisions.items() if d.startswith("B")}
    C    = {s for s, d in decisions.items() if d.startswith("C")}
    excl = B | C

    # Check which GPT-4o score files exist
    gpt4o_ready = any(
        (Path(SCORES_GPT4O) / f"{m}_test_scores.json").exists()
        for m in MODELS
    )
    if not gpt4o_ready:
        print("\n  ✗  No GPT-4o score files found.")
        print(f"     Run first:  python step5_gpt4o_judge.py --model all --split test")
        return

    print(f"\n{'='*70}")
    print(f"  MEDLECTUREBENCH — FINAL RESULTS  (judge: GPT-4o)")
    print(f"{'='*70}")
    print(f"\n  Review decisions : {len(decisions)} slides")
    print(f"  B (narration)    : {len(B)}  → reported separately (Table 5)")
    print(f"  C (admin)        : {len(C)}  → excluded")
    print(f"  Total excluded   : {len(excl)}")

    # ─── TABLE 1: Main results ─────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  TABLE 1: MAIN RESULTS — GPT-4o JUDGE")
    print(f"  {'Model':<18} {'─── Full benchmark ───':^33}  {'─── Clean (excl B+C) ───':^33}")
    print(f"  {'Model':<18} {'IGS':>6} {'CR':>6} {'RV':>6} {'SF':>6} {'n':>4}  "
          f"{'IGS':>6} {'CR':>6} {'RV':>6} {'SF':>6} {'n':>4}")
    print(f"  {'─'*18} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*4}  "
          f"{'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*4}")

    paper_rows = []
    for model in MODELS:
        sc = load(SCORES_GPT4O, model)
        if not sc:
            continue
        s_full  = stats(sc)
        clean   = [s for s in sc if s not in excl]
        s_clean = stats(sc, clean)
        print(f"  {MODEL_LABELS[model]:<18}{fmt(s_full)}  {fmt(s_clean)}")
        if s_clean:
            paper_rows.append((model, s_clean))

    # ─── TABLE 2: By content type ──────────────────────────────
    best_m  = max(paper_rows, key=lambda x: x[1]["igs"])[0] if paper_rows else MODELS[0]
    sc_best = load(SCORES_GPT4O, best_m)

    print(f"\n{'─'*70}")
    print(f"  TABLE 2: BY CONTENT TYPE — {MODEL_LABELS[best_m]} (clean benchmark)")
    print(f"  {'Type':<20} {'IGS':>6} {'CR':>6} {'RV':>6} {'SF':>6} {'n':>4} {'zeros':>6}")
    print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*4} {'─'*6}")
    for ct in ["mathematical", "conceptual", "computational", "mixed"]:
        sids = [s for s, v in sc_best.items()
                if v.get("content_type") == ct and s not in excl]
        s = stats(sc_best, sids)
        if s:
            print(f"  {ct:<20} {s['igs']:>6.3f} {s['cr']:>6.3f} "
                  f"{s['rv']:>6.3f} {s['sf']:>6.3f} {s['n']:>4} {s['n_zero']:>6}")

    # ─── TABLE 3: By lecture category ─────────────────────────
    cats = [
        "fundamentals", "mathematical_foundation", "signal_theory",
        "computational", "imaging_modality",
    ]
    print(f"\n{'─'*70}")
    print(f"  TABLE 3: BY LECTURE CATEGORY — all models (clean benchmark)")
    print(f"  {'Category':<25}" +
          "".join(f" {MODEL_LABELS[m][:9]:>10}" for m in MODELS))
    print(f"  {'─'*25}" + "".join(f" {'─'*10}" for _ in MODELS))
    for cat in cats:
        row, n_ref = f"  {cat:<25}", 0
        for model in MODELS:
            sc   = load(SCORES_GPT4O, model)
            sids = [s for s, v in sc.items()
                    if v.get("lecture_category") == cat and s not in excl]
            if model == MODELS[0]:
                n_ref = len(sids)
            avg = mean([sc[s]["igs"] for s in sids]) if sids else None
            row += f" {avg:>10.3f}" if avg is not None else f" {'—':>10}"
        row += f"  (n≈{n_ref})"
        print(row)

    # ─── TABLE 4: SF−CR gap ────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  TABLE 4: SF−CR GAP — scope–concept gap (GPT-4o judge)")
    print(f"  {'Model':<18} {'SF':>6} {'CR':>6} {'SF−CR':>8}")
    print(f"  {'─'*18} {'─'*6} {'─'*6} {'─'*8}")
    for model, s in paper_rows:
        gap = s["sf"] - s["cr"]
        print(f"  {MODEL_LABELS[model]:<18} {s['sf']:>6.3f} {s['cr']:>6.3f} {gap:>+8.3f}")

    # ─── TABLE 5: Narration-grounded slides ────────────────────
    if B:
        print(f"\n{'─'*70}")
        print(f"  TABLE 5: NARRATION-GROUNDED ({len(B)} slides) — structural VLM limit")
        print(f"  {'Model':<18} {'IGS':>6} {'n':>4}")
        print(f"  {'─'*18} {'─'*6} {'─'*4}")
        for model in MODELS:
            sc = load(SCORES_GPT4O, model)
            s  = stats(sc, [x for x in sc if x in B])
            if s:
                print(f"  {MODEL_LABELS[model]:<18} {s['igs']:>6.3f} {s['n']:>4}")

    # ─── TABLE 6: Qwen v3 vs GPT-4o comparison ─────────────────
    if args.compare:
        v3_ready = any(
            (Path(SCORES_V3) / f"{m}_test_scores.json").exists()
            for m in MODELS
        )
        if not v3_ready:
            print(f"\n  ⚠  Qwen v3 scores not found for comparison.")
            print(f"     Run: python step5_rescore_v3.py --model all --split test")
        else:
            print(f"\n{'─'*70}")
            print(f"  TABLE 6: QWEN v3 JUDGE vs GPT-4o JUDGE (clean benchmark)")
            print(f"  {'Model':<18} {'Qwen-IGS':>10} {'GPT4-IGS':>10} {'Δ IGS':>8}  "
                  f"{'Qwen-CR':>8} {'GPT4-CR':>8} {'Δ CR':>7}")
            print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*8}  {'─'*8} {'─'*8} {'─'*7}")
            for model in MODELS:
                sc_q = load(SCORES_V3,    model)
                sc_g = load(SCORES_GPT4O, model)
                shared_clean = [
                    s for s in set(sc_q) & set(sc_g) if s not in excl
                ]
                sq = stats(sc_q, shared_clean)
                sg = stats(sc_g, shared_clean)
                if sq and sg:
                    d_igs = sg["igs"] - sq["igs"]
                    d_cr  = sg["cr"]  - sq["cr"]
                    print(f"  {MODEL_LABELS[model]:<18} "
                          f"{sq['igs']:>10.3f} {sg['igs']:>10.3f} {d_igs:>+8.3f}  "
                          f"{sq['cr']:>8.3f} {sg['cr']:>8.3f} {d_cr:>+7.3f}")

    # ─── Headline summary ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  HEADLINE NUMBERS FOR PAPER  (judge: GPT-4o)")
    print(f"{'='*70}")
    if paper_rows:
        best_r  = max(paper_rows, key=lambda x: x[1]["igs"])
        worst_r = min(paper_rows, key=lambda x: x[1]["igs"])
        n_clean = best_r[1]["n"]
        max_igs = max(r["igs"] for _, r in paper_rows)
        all_sf_gt_cr = all((r["sf"] - r["cr"]) > 0 for _, r in paper_rows)

        print(f"""
  Judge model       : GPT-4o
  Clean benchmark   : {n_clean} slides
  ({len(B)} narration-grounded + {len(C)} administrative excluded from {TOTAL_TEST} total)

  Best model  : {MODEL_LABELS[best_r[0]]:<22} IGS = {best_r[1]['igs']:.3f}
  Worst model : {MODEL_LABELS[worst_r[0]]:<22} IGS = {worst_r[1]['igs']:.3f}
  IGS range   : {worst_r[1]['igs']:.3f} – {best_r[1]['igs']:.3f}

  Finding 1: No model exceeds IGS = {max_igs:.2f}.
  Finding 2: SF−CR gap positive for ALL models: {'✓ YES' if all_sf_gt_cr else '✗ CHECK DATA'}
  Finding 3: Check scale paradox (7B vs 40B).
  Finding 4: Narration-grounded slides: IGS ≈ 0.0 (structural VLM limit).
  Finding 5: imaging_modality easiest, mathematical_foundation hardest.
""")


if __name__ == "__main__":
    main()
