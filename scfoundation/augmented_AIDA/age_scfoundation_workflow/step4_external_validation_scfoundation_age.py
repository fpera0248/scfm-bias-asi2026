#!/usr/bin/env python3
"""STEP 4 -- External Validation Disease + Cell-Type Classification (AGE, scFoundation)"""

import copy, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

BASE        = Path("/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/age_scfoundation_workflow")
LABELED_DIR = BASE / "step3b_labeled"
OUTDIR      = BASE / "step4_external_validation_scfoundation_age"
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "AIDA_Age_Pilot"
PILOTS = {
    "Proportional_2498":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_2498_labeled_scfoundation.h5ad",
    "Downsampled_230Each":        LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_230Each_labeled_scfoundation.h5ad",
    "BalancedUpsampled_747Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_747Each_labeled_scfoundation.h5ad",
    "BalancedAugmented_747Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_747Each_labeled_scfoundation.h5ad",
}
VALIDATION_FILE = BASE / "ILD_Age_External_Validation_10500_scfoundation.h5ad"
CSV_FILE = OUTDIR / "step4_external_validation_results_age_scfoundation.csv"
TXT_FILE = OUTDIR / "step4_external_validation_log_age_scfoundation.txt"

EMB_KEY        = "X_scfoundation"
DISEASE_COL    = "disease"
CELLTYPE_COL   = "cell_type"
AGE_BIN_COL        = "age_bin_10yr"
DISEASE_GROUPS = {"10_19", "20_29", "30_39", "40_49", "50_59", "60_69", "70_79"}
RANDOM_STATE   = 42
EOS_NEIGHBORS  = 5
EOS_MULTIPLIER = 0.50
AR_BINS        = 30

MODEL_TEMPLATES = {
    "LogReg":       LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", n_jobs=None),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"),
}
STRATEGIES = {
    "Baseline": dict(use_ar=False, use_eos=False),
    "AR":       dict(use_ar=True,  use_eos=False),
    "EOS":      dict(use_ar=False, use_eos=True),
    "AR+EOS":   dict(use_ar=True,  use_eos=True),
}

log_fh = open(TXT_FILE, "w")
def log(msg):
    ts = time.strftime("%H:%M:%S"); line = f"[{ts}] {msg}"
    print(line, flush=True); print(line, file=log_fh, flush=True)

def canonicalize_age_bin(series): return series.astype(str).str.strip().str.lower()
def make_binary_disease(x):
    if pd.isna(x): return np.nan
    return "normal" if str(x).lower().strip() == "normal" else "disease"

def load_adata(name, path):
    if not path.exists(): raise FileNotFoundError(f"Not found: {path}")
    ad = sc.read_h5ad(path)
    if EMB_KEY not in ad.obsm: raise RuntimeError(f"{name}: missing {EMB_KEY}")
    X           = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    age_bin     = canonicalize_age_bin(ad.obs[AGE_BIN_COL])
    disease_bin = ad.obs[DISEASE_COL].apply(make_binary_disease)
    ct          = ad.obs[CELLTYPE_COL].astype(str)
    log(f"  {name}: {ad.n_obs:,} cells, {age_bin.nunique()} age groups, {ct.nunique()} cell types")
    return X, age_bin.to_numpy(), disease_bin.to_numpy(), ct.to_numpy()

def calculate_ar_weights(X_train, bins=AR_BINS):
    ALPHA = 0.0001; n, d = X_train.shape
    dim_weights = np.empty((n, d), dtype=np.float64)
    for k in range(d):
        vals = X_train[:, k]
        hist, edges = np.histogram(vals, bins=bins, density=True)
        bin_idx = np.clip(np.digitize(vals, edges) - 1, 0, bins - 1)
        prob_s  = (1.0 - ALPHA) * hist[bin_idx] * (edges[1]-edges[0]) + ALPHA / n
        dim_weights[:, k] = 1.0 / prob_s
    weights = dim_weights.max(axis=1); weights /= weights.mean()
    return weights.astype(np.float32)

def apply_eos(X_train, y_train, g_train, target_group):
    mask_min = (g_train == target_group)
    X_min, y_min = X_train[mask_min], y_train[mask_min]
    if X_min.shape[0] < 10: return X_train, y_train, g_train
    nn = NearestNeighbors(n_neighbors=EOS_NEIGHBORS).fit(X_train)
    _, neighbors = nn.kneighbors(X_min)
    rng = np.random.default_rng(RANDOM_STATE); synth = []
    for _ in range(int(len(X_min) * EOS_MULTIPLIER)):
        i0 = rng.integers(0, len(X_min))
        enemy = [j for j in neighbors[i0] if g_train[j] != target_group]
        if not enemy: continue
        j = rng.choice(enemy)
        synth.append(X_min[i0] + rng.uniform(0., 1.) * (X_min[i0] - X_train[j]))
    if not synth: return X_train, y_train, g_train
    synth = np.vstack(synth)
    return (np.vstack([X_train, synth]),
            np.concatenate([y_train, rng.choice(y_min, len(synth))]),
            np.concatenate([g_train, np.repeat(target_group, len(synth))]))

def evaluate_external(model_template, X_train, y_train, g_train, X_test, y_test, g_test, use_ar=False, use_eos=False):
    mask = pd.Series(y_train).notna().to_numpy() & ~np.isnan(X_train).any(axis=1)
    X_tr, y_tr, g_tr = X_train[mask], y_train[mask], g_train[mask]
    if len(np.unique(y_tr)) < 2: return None
    scaler = StandardScaler(); X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_test)
    if use_eos:
        minority = pd.Series(g_tr).value_counts().idxmin()
        X_tr, y_tr, g_tr = apply_eos(X_tr, y_tr, g_tr, minority)
    weights = calculate_ar_weights(X_tr) if use_ar else None
    model = copy.deepcopy(model_template)
    if weights is not None:
        try: model.fit(X_tr, y_tr, sample_weight=weights)
        except: model.fit(X_tr, y_tr)
    else: model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_test, preds); f1 = f1_score(y_test, preds, average="macro")
    per_group = {grp: accuracy_score(y_test[g_test==grp], preds[g_test==grp])
                 for grp in np.unique(g_test) if (g_test==grp).sum() > 0}
    worst_grp = min(per_group, key=per_group.get) if per_group else "n/a"
    return dict(accuracy=acc, macro_f1=f1, per_group=per_group,
                worst_group=worst_grp, worst_acc=per_group.get(worst_grp, np.nan),
                n_train=len(y_tr), n_test=len(y_test))

def main():
    warnings.filterwarnings("ignore")
    log("="*70); log("STEP 4 -- External Validation Classification (AGE, scFoundation)"); log("="*70)
    log("\nLoading validation set ...")
    X_val, age_val, dis_val, ct_val = load_adata("Validation", VALIDATION_FILE)
    disease_mask = pd.Series(age_val).isin(DISEASE_GROUPS).to_numpy()
    X_val_dis = X_val[disease_mask]; age_val_dis = age_val[disease_mask]; dis_val_dis = dis_val[disease_mask]
    log(f"  Disease subset: {X_val_dis.shape[0]:,} cells  Cell-type: {X_val.shape[0]:,} cells")
    rows = []
    for pname, ppath in PILOTS.items():
        log(f"\n{'='*70}\nPilot: {pname}")
        try: X_pilot, age_pilot, dis_pilot, ct_pilot = load_adata(pname, ppath)
        except (FileNotFoundError, RuntimeError) as err: log(f"  SKIP: {err}"); continue
        pilot_disease_mask = pd.Series(age_pilot).isin(DISEASE_GROUPS).to_numpy()
        X_pilot_dis = X_pilot[pilot_disease_mask]; age_pilot_dis = age_pilot[pilot_disease_mask]; dis_pilot_dis = dis_pilot[pilot_disease_mask]
        for model_name, model_tmpl in MODEL_TEMPLATES.items():
            for strat_name, flags in STRATEGIES.items():
                log(f"  {model_name:12s} | {strat_name:8s}")
                dres = evaluate_external(model_tmpl, X_pilot_dis, dis_pilot_dis, age_pilot_dis,
                                         X_val_dis, dis_val_dis, age_val_dis, use_ar=flags["use_ar"], use_eos=flags["use_eos"])
                cres = evaluate_external(model_tmpl, X_pilot, ct_pilot, age_pilot,
                                         X_val, ct_val, age_val, use_ar=flags["use_ar"], use_eos=flags["use_eos"])
                row = {"pilot": pname, "model": model_name, "strategy": strat_name}
                if dres:
                    row.update({"disease_accuracy": dres["accuracy"], "disease_macro_f1": dres["macro_f1"],
                                "disease_worst_age": dres["worst_group"], "disease_worst_acc": dres["worst_acc"],
                                "disease_n_train": dres["n_train"], "disease_n_test": dres["n_test"]})
                    for grp, a in dres["per_group"].items(): row[f"disease_acc_{grp}"] = a
                    log(f"    disease: acc={dres['accuracy']:.3f} f1={dres['macro_f1']:.3f} worst={dres['worst_group']}:{dres['worst_acc']:.3f}")
                if cres:
                    row.update({"celltype_accuracy": cres["accuracy"], "celltype_macro_f1": cres["macro_f1"],
                                "celltype_worst_age": cres["worst_group"], "celltype_worst_acc": cres["worst_acc"]})
                    log(f"    celltype: acc={cres['accuracy']:.3f} f1={cres['macro_f1']:.3f}")
                rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(CSV_FILE, index=False)
    log(f"\n{'='*70}\nSTEP 4 COMPLETE (AGE, scFoundation)")
    summary_cols = ["pilot","model","strategy","disease_accuracy","disease_macro_f1","celltype_accuracy","celltype_macro_f1"]
    print("\n" + df[[c for c in summary_cols if c in df.columns]].to_string(index=False))
    log_fh.close()

if __name__ == "__main__":
    main()
