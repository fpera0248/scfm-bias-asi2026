#!/usr/bin/env python3
"""
STEP 4a — Downstream Disease Prediction (SEX) with AR + EOS
Geneformer V2-316M workflow

Changes vs ethnicity version:
  [SEX 1] BASE path     -> age_Geneformer_workflow
  [SEX 2] OUTPUT_BASE   -> ILD_Age_Pilot
  [SEX 3] DATASETS      -> sex filenames (1413Each, 586Each, 1999)
  [SEX 4] GROUP_COL     -> sex instead of self_reported_ethnicity
  [SEX 5] Output files  -> *_sex_geneformer suffix
"""

import copy
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

BASE   = Path("/oscar/home/fperalta/data/fperalta/Geneformer/augmented/age_Geneformer_workflow")
OUTDIR = BASE / "step4a_downstream_sex"
OUTDIR.mkdir(exist_ok=True)

OUTFILE_TXT = OUTDIR / "step4a_downstream_results_age_AR_EOS_geneformer.txt"
OUTFILE_CSV = OUTDIR / "step4a_downstream_results_age_AR_EOS_geneformer.csv"

OUTPUT_BASE = "ILD_Age_Pilot"                                    # [SEX 2]
LABELED_DIR = BASE / "step3b_labeled"

DATASETS = {                                                     # [SEX 3]
    "BalancedAugmented_1262Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_1262Each_labeled_geneformer.h5ad",
    "Proportional_2495":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_2495_labeled_geneformer.h5ad",
    "BalancedUpsampled_1262Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_1262Each_labeled_geneformer.h5ad",
    "Downsampled_25Each":        LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_25Each_labeled_geneformer.h5ad",
}

MIN_CELLS_FOR_EVAL = 80
EMB_KEY      = "X_geneformer"
DISEASE_COL  = "disease"
CELLTYPE_COL = "cell_type"

AGE_BIN_COL_CANDIDATES = ["age_bin_10yr", "Sex", "SEX", "gender", "Gender"]  # [SEX 4]

RANDOM_STATE   = 42
TEST_SIZE      = 0.20
EOS_NEIGHBORS  = 5
EOS_MULTIPLIER = 0.50
AR_BINS        = 30

MODEL_TEMPLATES = {
    "LogReg":       LogisticRegression(max_iter=2000, solver="lbfgs",
                                       class_weight="balanced", n_jobs=None),
    "LinearSVM":    LinearSVC(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
}

STRATEGIES = {
    "Baseline": dict(use_ar=False, use_eos=False),
    "AR":       dict(use_ar=True,  use_eos=False),
    "EOS":      dict(use_ar=False, use_eos=True),
    "AR+EOS":   dict(use_ar=True,  use_eos=True),
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def detect_age_col(obs, fname):                                  # [SEX 4]
    for c in AGE_BIN_COL_CANDIDATES:
        if c in obs.columns:
            return c
    raise RuntimeError(f"No age column found in {fname}.")

def canonicalize_age_bin(series):
    return series.astype(str).str.strip().str.lower()

def make_binary_disease(x):
    if pd.isna(x):
        return np.nan
    return "normal" if str(x).lower().strip() == "normal" else "disease"

def extract_features(adata, label_col, grp_col):
    if EMB_KEY not in adata.obsm:
        raise RuntimeError(f"Missing embedding '{EMB_KEY}'")
    X = np.array(adata.obsm[EMB_KEY], dtype=np.float32)
    y = adata.obs[label_col]
    g = canonicalize_age_bin(adata.obs[grp_col])
    valid = (~pd.isna(y)) & (~pd.isna(g)) & (~np.isnan(X).any(axis=1)) & (np.abs(X).sum(axis=1) > 0)
    return X[valid].astype(np.float32), y[valid].astype(str).to_numpy(), g[valid].to_numpy()

def random_baseline(y):
    rng   = np.random.default_rng(RANDOM_STATE)
    preds = rng.choice(np.unique(y), size=len(y))
    return accuracy_score(y, preds), f1_score(y, preds, average="macro")

def safe_stratify(y):
    counts = pd.Series(y).value_counts()
    if (counts < 2).any():
        return None
    return y

def calculate_adaptive_weights_pdf(X_train, bins=AR_BINS):
    ALPHA = 0.0001
    n, d  = X_train.shape
    dim_weights = np.empty((n, d), dtype=np.float64)
    for k in range(d):
        vals        = X_train[:, k]
        hist, edges = np.histogram(vals, bins=bins, density=True)
        bin_idx     = np.clip(np.digitize(vals, edges) - 1, 0, bins - 1)
        bin_width   = edges[1] - edges[0]
        prob        = hist[bin_idx] * bin_width
        prob_s      = (1.0 - ALPHA) * prob + ALPHA / n
        dim_weights[:, k] = 1.0 / prob_s
    weights  = dim_weights.max(axis=1)
    weights /= weights.mean()
    return weights.astype(np.float32)

def apply_eos(X_train, y_train, g_train, target_group):
    mask_min = (g_train == target_group)
    X_min, y_min = X_train[mask_min], y_train[mask_min]
    if X_min.shape[0] < 10:
        return X_train, y_train, g_train
    nn = NearestNeighbors(n_neighbors=EOS_NEIGHBORS).fit(X_train)
    _, neighbors = nn.kneighbors(X_min)
    rng   = np.random.default_rng(RANDOM_STATE)
    n_new = int(len(X_min) * EOS_MULTIPLIER)
    synth = []
    for _ in range(n_new):
        i0    = rng.integers(0, len(X_min))
        enemy = [j for j in neighbors[i0] if g_train[j] != target_group]
        if not enemy:
            continue
        j = rng.choice(enemy)
        R = rng.uniform(0.0, 1.0)
        synth.append(X_min[i0] + R * (X_min[i0] - X_train[j]))
    if not synth:
        return X_train, y_train, g_train
    synth = np.vstack(synth)
    y_new = rng.choice(y_min, size=len(synth))
    g_new = np.repeat(target_group, len(synth))
    return np.vstack([X_train, synth]), np.concatenate([y_train, y_new]), np.concatenate([g_train, g_new])

def evaluate(model_template, X, y, g, use_ar=False, use_eos=False):
    stratify = safe_stratify(y)
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, g, test_size=TEST_SIZE, stratify=stratify, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    if use_eos:
        minority = pd.Series(g_tr).value_counts().idxmin()
        X_tr, y_tr, g_tr = apply_eos(X_tr, y_tr, g_tr, minority)
    weights = calculate_adaptive_weights_pdf(X_tr) if use_ar else None
    model = copy.deepcopy(model_template)
    if weights is not None:
        try:
            model.fit(X_tr, y_tr, sample_weight=weights)
        except TypeError:
            model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    f1    = f1_score(y_te, preds, average="macro")
    per_group = {grp: accuracy_score(y_te[g_te==grp], preds[g_te==grp])
                 for grp in np.unique(g_te) if (g_te==grp).sum() > 0}
    worst_grp = min(per_group, key=per_group.get) if per_group else "n/a"
    worst_acc = per_group.get(worst_grp, np.nan)
    return acc, f1, worst_grp, worst_acc, per_group

def main():
    warnings.filterwarnings("ignore")
    rows = []
    with open(OUTFILE_TXT, "w") as fh:
        fh.write("STEP 4a -- Downstream Disease Prediction (SEX) with AR + EOS\nGeneformer V2-316M\n" + "="*80 + "\n")
        for dname, fpath in DATASETS.items():
            if not fpath.exists():
                log(f"  Skipping {dname} -- file not found: {fpath}")
                continue
            log(f"Loading {dname}")
            adata = sc.read_h5ad(fpath)
            adata.obs_names_make_unique()
            grp_col = detect_age_col(adata.obs, fpath)           # [SEX 4]
            X, y_raw, g = extract_features(adata, DISEASE_COL, grp_col)
            y_bin = np.array([make_binary_disease(v) for v in y_raw])
            y_bin = pd.Series(y_bin).map({"normal": 0, "disease": 1})
            keep  = ~pd.isna(y_bin)
            X_d, g_d, y_d = X[keep.to_numpy()].astype(np.float32), g[keep.to_numpy()], y_bin[keep].astype(int).to_numpy()
            X_ct, y_ct, g_ct = extract_features(adata, CELLTYPE_COL, grp_col)
            rand_acc, rand_f1 = random_baseline(y_d)
            fh.write(f"\n{'='*80}\nDataset: {dname}\n  Cells: {len(y_d):,}\n  Random baseline: acc={rand_acc:.3f}  f1={rand_f1:.3f}\n")
            for model_name, model_tmpl in MODEL_TEMPLATES.items():
                for strat_name, flags in STRATEGIES.items():
                    log(f"  Running {model_name} | {strat_name}")
                    acc, f1, worst_grp, worst_acc, per_group = evaluate(
                        model_tmpl, X_d, y_d, g_d, use_ar=flags["use_ar"], use_eos=flags["use_eos"])
                    ct_acc, ct_f1, ct_worst_grp, ct_worst_acc, ct_per_group = evaluate(
                        model_tmpl, X_ct, y_ct, g_ct, use_ar=flags["use_ar"], use_eos=flags["use_eos"])
                    row = {
                        "dataset": dname, "model": model_name, "strategy": strat_name,
                        "disease_accuracy": acc, "disease_macro_f1": f1,
                        "disease_worst_age": worst_grp, "disease_worst_age_acc": worst_acc,
                        "celltype_accuracy": ct_acc, "celltype_macro_f1": ct_f1,
                        "celltype_worst_age": ct_worst_grp, "celltype_worst_age_acc": ct_worst_acc,
                        "random_baseline_acc": rand_acc, "random_baseline_f1": rand_f1, "n_cells": len(y_d),
                    }
                    for grp, a in per_group.items():
                        row[f"per_age_disease_acc_{grp}"] = a
                    rows.append(row)
                    fh.write(f"  {model_name:12s} | {strat_name:8s} | disease acc={acc:.3f} f1={f1:.3f} | worst={worst_grp}:{worst_acc:.3f} | ct acc={ct_acc:.3f}\n")
                    log(f"    -> disease acc={acc:.3f} f1={f1:.3f} | worst={worst_grp}:{worst_acc:.3f} | ct acc={ct_acc:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUTFILE_CSV, index=False)
    log("STEP 4a COMPLETE")
    log(f"   Results -> {OUTFILE_CSV}")
    print("\n" + df[["dataset","model","strategy","disease_accuracy","disease_macro_f1","disease_worst_age_acc","celltype_accuracy"]].to_string(index=False))

if __name__ == "__main__":
    main()
