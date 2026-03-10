#!/usr/bin/env python3
"""
Baseline Metrics vs IGS Correlation.
Computes BLEU-1, BLEU-4, ROUGE-L, BERTScore for VLM outputs vs teacher texts.
Correlates each metric with IGS to prove standard metrics fail.

Install deps:
  pip install sacrebleu rouge-score bert-score --break-system-packages

Run from: ~/islamm11/MedLecture/
Usage:
  python step6_baseline_metrics.py                 # all models, v3 scores
  python step6_baseline_metrics.py --no-bertscore  # skip BERTScore (slow)
  python step6_baseline_metrics.py --use-v1        # use original v1 scores
"""

import json, re, csv, argparse
from pathlib import Path
from collections import defaultdict

LECTURES_ROOT = "/home/islamm11/islamm11/MedLecture/Lectures"
SCORES_V3     = "/home/islamm11/islamm11/MedLecture/medlecture_bench/igs_scores_gpt4o_v5"
SCORES_V1     = "/home/islamm11/islamm11/MedLecture/medlecture_bench/igs_scores_gpt4o_v5"
VLM_OUT_DIR   = "/home/islamm11/islamm11/MedLecture/medlecture_bench/vlm_outputs"
DECISIONS_CSV = "/home/islamm11/islamm11/MedLecture/medlecture_bench/manual_review/review_decisions.csv"
OUT_DIR       = "/home/islamm11/islamm11/MedLecture/medlecture_bench/baseline_metrics"

MODELS = ["qwen2vl_7b","internvl2_8b","internvl2_26b","internvl2_40b","llava_next_34b"]
MODEL_LABELS = {
    "qwen2vl_7b":    "Qwen2-VL-7B",
    "internvl2_8b":  "InternVL2-8B",
    "internvl2_26b": "InternVL2-26B",
    "internvl2_40b": "InternVL2-40B",
    "llava_next_34b":"LLaVA-1.6-34B",
}

def get_teacher_text(sid):
    m = re.match(r'L(\d+)_S(\d+)', sid)
    if not m: return ""
    p = Path(LECTURES_ROOT) / f"Lecture {int(m.group(1))}" / "Texts" / f"Slide{int(m.group(2))}.txt"
    return p.read_text().strip() if p.exists() else ""

def load_decisions():
    p = Path(DECISIONS_CSV)
    if not p.exists(): return set()
    excl = set()
    with open(p) as f:
        for row in csv.DictReader(f):
            d = row.get("decision","").strip()
            if d.startswith("B") or d.startswith("C"):
                excl.add(row.get("slide_id","").strip())
    return excl

def pearson(x, y):
    n = len(x)
    if n < 3: return 0.0
    mx, my = sum(x)/n, sum(y)/n
    num = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    dx  = sum((x[i]-mx)**2 for i in range(n))**0.5
    dy  = sum((y[i]-my)**2 for i in range(n))**0.5
    return num/(dx*dy) if dx*dy > 0 else 0.0

def spearman(x, y):
    n = len(x)
    if n < 3: return 0.0
    def ranks(lst):
        si = sorted(range(n), key=lambda i: lst[i])
        r  = [0.0]*n
        for rank, idx in enumerate(si): r[idx] = rank+1
        return r
    rx, ry = ranks(x), ranks(y)
    d2 = sum((rx[i]-ry[i])**2 for i in range(n))
    return 1 - 6*d2/(n*(n**2-1))

def compute_bleu(hyp, ref):
    try:
        import sacrebleu
        b1 = sacrebleu.sentence_bleu(hyp, [ref], smooth_method='exp', smooth_value=0.1)
        b4 = sacrebleu.sentence_bleu(hyp, [ref], smooth_method='exp', smooth_value=0.1)
        return float(b1.score)/100, float(b4.score)/100
    except: return 0.0, 0.0

def compute_rouge(hyp, ref):
    try:
        from rouge_score import rouge_scorer
        s = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        return s.score(ref, hyp)['rougeL'].fmeasure
    except: return 0.0

def mean(lst): return sum(lst)/len(lst) if lst else 0.0

def main(models, split, use_bertscore, scores_dir):
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    excl = load_decisions()

    print(f"\n{'='*65}")
    print(f"  BASELINE METRICS vs IGS CORRELATION")
    print(f"  Split: {split}  |  Scores: {Path(scores_dir).name}")
    print(f"  Excluded (B+C): {len(excl)} slides")
    print(f"{'='*65}")

    if use_bertscore:
        try:
            from bert_score import score as bscore
            print(f"  BERTScore: ✅ available")
        except ImportError:
            print(f"  BERTScore: ❌ not installed, skipping")
            use_bertscore = False

    all_summary = {}

    for model in models:
        print(f"\n{'─'*55}")
        print(f"  {MODEL_LABELS[model]}")

        sf = Path(scores_dir) / f"{model}_{split}_scores.json"
        if not sf.exists():
            sf = Path(SCORES_V1) / f"{model}_{split}_scores.json"
            if not sf.exists():
                print(f"  ❌ No scores found — skip"); continue
            print(f"  (using v1 fallback)")
        igs_scores = json.load(open(sf))

        vf = Path(VLM_OUT_DIR) / f"{model}_{split}.json"
        if not vf.exists():
            print(f"  ❌ No VLM outputs — skip"); continue
        vlm_data = json.load(open(vf))

        results, hyps, refs, sids_ord = {}, [], [], []

        for sid in sorted(igs_scores.keys()):
            if sid in excl: continue
            igs = igs_scores[sid].get("igs", 0)
            if igs == 0.0: continue
            hyp = vlm_data.get(sid,{}).get("response","").strip()
            ref = get_teacher_text(sid)
            if not hyp or not ref or len(ref) < 30: continue
            b1, b4 = compute_bleu(hyp, ref)
            rl     = compute_rouge(hyp, ref)
            results[sid] = {
                "igs": igs, "bleu1": b1, "bleu4": b4, "rougeL": rl,
                "content_type":     igs_scores[sid].get("content_type",""),
                "lecture_category": igs_scores[sid].get("lecture_category",""),
            }
            hyps.append(hyp); refs.append(ref); sids_ord.append(sid)

        n = len(results)
        print(f"  Valid slides: {n}")
        if n < 10: print(f"  ⚠️  Too few — skip"); continue

        if use_bertscore and hyps:
            print(f"  Computing BERTScore...")
            try:
                P, R, F1 = bscore(hyps, refs, lang="en",
                                   rescale_with_baseline=True, verbose=False)
                for i, sid in enumerate(sids_ord):
                    if sid in results: results[sid]["bertscore"] = float(F1[i])
            except Exception as e:
                print(f"  BERTScore error: {e}")

        igs_v  = [results[s]["igs"]    for s in results]
        b1_v   = [results[s]["bleu1"]  for s in results]
        b4_v   = [results[s]["bleu4"]  for s in results]
        rl_v   = [results[s]["rougeL"] for s in results]

        metrics = {
            "BLEU-1":  (b1_v,  pearson(b1_v, igs_v),  spearman(b1_v, igs_v)),
            "BLEU-4":  (b4_v,  pearson(b4_v, igs_v),  spearman(b4_v, igs_v)),
            "ROUGE-L": (rl_v,  pearson(rl_v, igs_v),  spearman(rl_v, igs_v)),
        }
        if use_bertscore and any("bertscore" in results[s] for s in results):
            bs_v = [results[s].get("bertscore",0) for s in results]
            metrics["BERTScore"] = (bs_v, pearson(bs_v,igs_v), spearman(bs_v,igs_v))

        print(f"  Mean IGS: {mean(igs_v):.3f}")
        print(f"\n  Correlation with IGS:")
        print(f"  {'Metric':<10} {'Mean':>6} {'Pearson r':>10} {'Spearman ρ':>11}  Verdict")
        print(f"  {'─'*10} {'─'*6} {'─'*10} {'─'*11}  {'─'*20}")
        for name, (vals, pr, sr) in metrics.items():
            flag = "✅ cannot detect IGF" if abs(pr)<0.35 else "⚠️  partial" if abs(pr)<0.60 else "❌ correlated"
            print(f"  {name:<10} {mean(vals):>6.3f} {pr:>+10.3f} {sr:>+11.3f}  {flag}")

        # Paper examples: high ROUGE-L, low IGS
        candidates = sorted(
            [(s,r) for s,r in results.items() if r["rougeL"]>0.20 and r["igs"]<0.20],
            key=lambda x: x[1]["rougeL"]-x[1]["igs"], reverse=True
        )
        if candidates:
            print(f"\n  ★ Figure 1 candidates (high ROUGE-L, low IGS):")
            print(f"    {'Slide':<15} {'ROUGE-L':>8} {'IGS':>6}  {'Gap':>5}  Category")
            for sid, r in candidates[:5]:
                print(f"    {sid:<15} {r['rougeL']:>8.3f} {r['igs']:>6.3f}  "
                      f"{r['rougeL']-r['igs']:>5.2f}  {r['lecture_category']}")

        json.dump(results, open(Path(OUT_DIR)/f"{model}_{split}_baseline.json","w"), indent=2)

        summ = {"n": n, "mean_igs": mean(igs_v)}
        for name,(vals,pr,sr) in metrics.items():
            k = name.lower().replace("-","").replace(" ","")
            summ[f"mean_{k}"] = mean(vals)
            summ[f"pearson_{k}"] = pr
            summ[f"spearman_{k}"] = sr
        all_summary[model] = summ

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY TABLE — CORRELATION WITH IGS (Pearson r)")
    print(f"  Key threshold: |r| < 0.35 = metric CANNOT detect instructional failure")
    print(f"{'='*65}")
    print(f"  {'Model':<18} {'IGS':>6} {'BLEU-1':>8} {'ROUGE-L':>9} {'BERTSc':>8}")
    for model in models:
        s = all_summary.get(model)
        if not s: continue
        bs = f"{s.get('pearson_bertscore',0):>8.3f}" if "pearson_bertscore" in s else f"{'N/A':>8}"
        print(f"  {MODEL_LABELS[model]:<18} {s['mean_igs']:>6.3f} "
              f"{s.get('pearson_bleu1',0):>8.3f} {s.get('pearson_rougel',0):>9.3f} {bs}")

    json.dump(all_summary, open(Path(OUT_DIR)/f"summary_{split}.json","w"), indent=2)
    print(f"\n  ✅ Saved to {OUT_DIR}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--use-v1", action="store_true")
    args = parser.parse_args()

    missing = []
    try: import sacrebleu
    except ImportError: missing.append("sacrebleu")
    try: from rouge_score import rouge_scorer
    except ImportError: missing.append("rouge-score")
    if missing:
        print(f"❌ Missing: pip install {' '.join(missing)} --break-system-packages")
        exit(1)

    models = MODELS if args.model == "all" else [args.model]
    scores_dir = SCORES_V1 if args.use_v1 else SCORES_V3
    main(models, args.split, not args.no_bertscore, scores_dir)
