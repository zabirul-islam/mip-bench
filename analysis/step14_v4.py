import json, numpy as np
from pathlib import Path
from collections import Counter

claims_dir = Path("medlecture_bench/claims_v5")
scores_dir = Path("medlecture_bench/igs_scores_gpt4o_v5")

MODELS = ["qwen2vl_7b","internvl2_8b","internvl2_26b","internvl2_40b","llava_next_34b"]

# ── Load teacher concept sets ─────────────────────────────────────────────────
teacher = {}
for f in sorted(claims_dir.glob("*.json")):
    d = json.load(open(f))
    if not isinstance(d, dict): continue
    sid = d.get('slide_id') or f.stem
    kc  = [c for c in (d.get('key_concepts') or []) if isinstance(c, str)]
    teacher[sid] = set(kc)

global_freq = Counter(c for kc in teacher.values() for c in kc)
print(f"Teacher graphs loaded: {len(teacher)} slides")
print(f"Top 5 concepts by frequency: {global_freq.most_common(5)}")

# ── Peek at score file structure to find correct keys ─────────────────────────
print("\n=== SCORE FILE KEY INSPECTION ===")
for m in MODELS:
    sf = scores_dir / f"{m}_test_scores.json"
    if not sf.exists():
        print(f"  {m}: NOT FOUND")
        continue
    data = json.load(open(sf))
    first_sid = next(iter(data))
    first_val = data[first_sid]
    print(f"  {m}: {len(data)} slides | first entry keys: {list(first_val.keys()) if isinstance(first_val, dict) else type(first_val)}")
    if isinstance(first_val, dict):
        print(f"    sample values: { {k: first_val[k] for k in list(first_val.keys())[:6]} }")

# ── Find correct IGS/CR/RV/SF keys ───────────────────────────────────────────
def get_score(entry, key):
    """Try multiple possible key names."""
    for k in [key, key.upper(), key.lower(),
              f"human_{key}", f"score_{key}",
              'instructional_grounding_score' if key == 'igs' else None]:
        if k and k in entry:
            return float(entry[k])
    # fallback: if entry IS a float/int
    if isinstance(entry, (int, float)):
        return float(entry)
    return None

print("\n=== PER-MODEL SCORE SUMMARY ===")
model_data = {}
for m in MODELS:
    sf = scores_dir / f"{m}_test_scores.json"
    if not sf.exists(): continue
    data = json.load(open(sf))

    # Auto-detect keys from first entry
    first = next(iter(data.values()))
    if not isinstance(first, dict):
        print(f"  {m}: scores are not dicts — value type: {type(first)}")
        continue

    keys = list(first.keys())
    # Common patterns
    igs_key = next((k for k in keys if 'igs' in k.lower() or 'grounding' in k.lower()), None)
    cr_key  = next((k for k in keys if k.lower() in ['cr','concept_recall','cr_score']), None)
    rv_key  = next((k for k in keys if k.lower() in ['rv','relational_validity','rv_score']), None)
    sf_key  = next((k for k in keys if k.lower() in ['sf','scope_fidelity','sf_score']), None)

    print(f"\n  {m}:")
    print(f"    Keys: {keys}")
    print(f"    Detected: igs={igs_key} cr={cr_key} rv={rv_key} sf={sf_key}")

    if not igs_key:
        print(f"    WARNING: could not find IGS key — skipping")
        continue

    igs_vals, cr_vals, rv_vals, sf_vals = [], [], [], []
    for sid, s in data.items():
        igs_vals.append(float(s.get(igs_key, 0) or 0))
        cr_vals.append(float(s.get(cr_key,  0) or 0) if cr_key else 0)
        rv_vals.append(float(s.get(rv_key,  0) or 0) if rv_key else 0)
        sf_vals.append(float(s.get(sf_key,  0) or 0) if sf_key else 0)

    zeros = sum(1 for v in igs_vals if v == 0)
    print(f"    n={len(igs_vals)} | IGS={np.mean(igs_vals):.3f} CR={np.mean(cr_vals):.3f} "
          f"RV={np.mean(rv_vals):.3f} SF={np.mean(sf_vals):.3f} zeros={zeros}")

    model_data[m] = {
        'igs_key': igs_key, 'cr_key': cr_key, 'rv_key': rv_key, 'sf_key': sf_key,
        'raw': data
    }

# ── Inter-model IGS correlation ───────────────────────────────────────────────
if len(model_data) >= 2:
    print("\n=== INTER-MODEL IGS CORRELATION ===")
    m_igs = {}
    for m, d in model_data.items():
        igs_k = d['igs_key']
        m_igs[m] = {sid: float(s.get(igs_k, 0) or 0) for sid, s in d['raw'].items()}

    avail = list(m_igs.keys())
    shared = set.intersection(*[set(v.keys()) for v in m_igs.values()])
    print(f"Shared test slides: {len(shared)}")

    for i, m1 in enumerate(avail):
        for m2 in avail[i+1:]:
            v1 = [m_igs[m1][s] for s in shared]
            v2 = [m_igs[m2][s] for s in shared]
            r  = np.corrcoef(v1, v2)[0,1]
            print(f"  {m1} vs {m2}: r={r:.3f}")

# ── Centrality analysis ───────────────────────────────────────────────────────
    print("\n=== CENTRALITY-IGS ANALYSIS ===")
    print("Do models miss high-centrality (cross-cutting) concepts?")
    for m, d in model_data.items():
        cr_k = d['cr_key']
        if not cr_k: continue
        low_cent, high_cent = [], []
        for sid, s in d['raw'].items():
            if sid not in teacher or not teacher[sid]: continue
            avg_c = np.mean([global_freq[c] for c in teacher[sid]])
            cr    = float(s.get(cr_k, 0) or 0)
            if cr == 0:   low_cent.append(avg_c)
            elif cr >= 0.8: high_cent.append(avg_c)
        lc = np.mean(low_cent)  if low_cent  else 0
        hc = np.mean(high_cent) if high_cent else 0
        flag = "MISSES high-centrality ⚠" if lc > hc else "recalls high-centrality ✓"
        print(f"  {m}: CR=0→centrality={lc:.2f} | CR≥0.8→centrality={hc:.2f} | {flag}")
        print(f"    (n_zero={len(low_cent)}, n_high={len(high_cent)})")

print("\nDone.")
