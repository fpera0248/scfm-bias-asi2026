#!/usr/bin/env python3
"""
STEP 5 — Fairness Stress Test (SEX, scGPT)

Changes vs ethnicity scGPT version:
  [SEX 1] BASE/OUTDIR  -> sex_scGPT_workflow
  [SEX 2] GROUP_KEY    -> "sex"
  [SEX 3] DATASETS     -> sex filenames
"""

import scanpy as sc
import numpy as np
import pandas as pd
import pathlib, time, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, silhouette_score

warnings.filterwarnings("ignore")

BASE = pathlib.Path("/oscar/home/fperalta/data/fperalta/scGPT/sex_scGPT_workflow")
LABELED_DIR = BASE / "step3b_labeled"
OUTDIR      = BASE / "step5_outputs_scgpt_sex"
OUTDIR.mkdir(exist_ok=True)

EMB_KEY      = "X_scGPT"
GROUP_KEY    = "sex"
DISEASE_COL  = "disease"
SOURCE_COL   = "source"
OUTPUT_BASE  = "ILD_Sex_Pilot"

DATASETS = {
    "BalancedAugmented_1413Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_1413Each_labeled_scgpt.h5ad",
    "Proportional_1999":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_1999_labeled_scgpt.h5ad",
    "BalancedUpsampled_1413Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_1413Each_labeled_scgpt.h5ad",
    "Downsampled_586Each":        LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_586Each_labeled_scgpt.h5ad",
}

RANDOM_STATE = 42
TEST_SIZE    = 0.20
MIN_REQUIRED = 10
LC_CV_FOLDS  = 5
LC_N_POINTS  = 8

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def canonicalize(series): return series.astype(str).str.strip().str.lower()
def to_binary_disease(x):
    if pd.isna(x): return np.nan
    return "normal" if str(x).lower().strip() == "normal" else "disease"

def load_dataset(path):
    ad = sc.read_h5ad(path)
    if not ad.obs_names.is_unique: ad.obs_names_make_unique()
    ad.obs[GROUP_KEY]        = canonicalize(ad.obs[GROUP_KEY])
    ad.obs["disease_binary"] = ad.obs[DISEASE_COL].apply(to_binary_disease)
    ad.obs["is_synthetic"]   = (ad.obs[SOURCE_COL] == "synthetic") if SOURCE_COL in ad.obs.columns else False
    return ad

def extract_scaled(ad):
    X   = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    y   = ad.obs["disease_binary"]; g = ad.obs[GROUP_KEY]; syn = ad.obs["is_synthetic"]
    bad = np.isnan(X).any(axis=1) | (np.abs(X).sum(axis=1) == 0)
    if bad.any():
        X, y, g, syn = X[~bad], y[~bad], g[~bad], syn[~bad]
    valid = pd.notna(y) & pd.notna(g)
    Xv    = X[valid.values]
    return StandardScaler().fit_transform(Xv), np.asarray(y[valid]).astype(str), np.asarray(g[valid]).astype(str), np.asarray(syn[valid]).astype(bool)

def safe_silhouette(X, y, syn):
    mask = (~syn) & (pd.Series(y).notna().values)
    if mask.sum() < 50 or len(np.unique(np.asarray(y)[mask])) < 2: return np.nan
    return round(float(silhouette_score(X[mask], np.asarray(y)[mask], sample_size=min(5000, mask.sum()))), 4)

def evaluate_classifier(X, y, g, seed=RANDOM_STATE):
    if len(np.unique(y)) < 2 or X.shape[0] < 40: return None
    counts   = pd.Series(y).value_counts()
    stratify = y if (counts >= 2).all() else None
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(X, y, g, test_size=TEST_SIZE, stratify=stratify, random_state=seed)
    clf = LogisticRegression(max_iter=8000, solver="saga", class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    per_grp = {grp: accuracy_score(y_te[g_te==grp], preds[g_te==grp]) for grp in np.unique(g_te) if (g_te==grp).sum() >= 5}
    worst   = min(per_grp, key=per_grp.get) if per_grp else "NA"
    return {
        "accuracy": round(float(accuracy_score(y_te, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_te, preds)), 4),
        "macro_f1": round(float(f1_score(y_te, preds, average="macro")), 4),
        "worst_sex": str(worst), "worst_sex_accuracy": round(float(per_grp.get(worst, np.nan)), 4) if worst != "NA" else np.nan,
        "per_sex_accuracy": {k: round(float(v), 4) for k, v in per_grp.items()},
        "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0]),
    }

def strict_sex_balance_indices(ad, min_required=MIN_REQUIRED, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    obs = ad.obs.copy()
    obs = obs[pd.notna(obs["disease_binary"])]
    tab = obs.groupby([GROUP_KEY, "disease_binary"], observed=False).size().reset_index(name="count")
    valid_groups = [grp for grp in tab[GROUP_KEY].unique()
                    if tab[tab[GROUP_KEY]==grp]["disease_binary"].nunique()==2
                    and tab[tab[GROUP_KEY]==grp]["count"].min() > 0]
    if len(valid_groups) < 2:
        raise RuntimeError(f"Not enough sex groups with both disease classes. Found: {valid_groups}")
    obs2   = obs[obs[GROUP_KEY].isin(valid_groups)].copy()
    groups = obs2.groupby([GROUP_KEY, "disease_binary"], observed=False)
    min_g  = int(groups.size().min())
    if min_g < min_required:
        raise RuntimeError(f"min_group_size too small: {min_g}")
    selected = []
    for (_, _), idx in groups.groups.items():
        selected.extend(rng.choice(list(idx), size=min_g, replace=False))
    return np.array(selected), min_g, valid_groups

def compute_learning_curves(X, y, dataset_name):
    counts = pd.Series(y).value_counts()
    if counts.min() < LC_CV_FOLDS * 2: return pd.DataFrame()
    train_sizes = np.linspace(0.10, 1.0, LC_N_POINTS)
    results = []
    for model_name, estimator in [
        ("LogReg",       LogisticRegression(max_iter=5000, solver="saga", n_jobs=-1)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)),
    ]:
        try:
            ts, tr_scores, val_scores = learning_curve(
                estimator=estimator, X=X, y=y, train_sizes=train_sizes,
                cv=LC_CV_FOLDS, scoring="f1_macro", n_jobs=-1, random_state=RANDOM_STATE)
            for i, n in enumerate(ts):
                results.append({"dataset": dataset_name, "model": model_name, "n_train": int(n),
                                 "train_f1_mean": round(float(tr_scores[i].mean()), 4),
                                 "train_f1_std":  round(float(tr_scores[i].std()),  4),
                                 "val_f1_mean":   round(float(val_scores[i].mean()), 4),
                                 "val_f1_std":    round(float(val_scores[i].std()),  4)})
        except Exception as e:
            log(f"  WARNING: LC failed for {model_name}: {e}")
    return pd.DataFrame(results)

t_start = time.time()
report_lines = ["STEP 5 -- FAIRNESS STRESS TEST (SEX, scGPT)\n" + "="*70 + "\n"]
all_lc_rows  = []
summary_rows = []

for ds_name, path in DATASETS.items():
    if not path.exists():
        log(f"  Skipping {ds_name}"); continue
    log(f"\n{'='*60}\nDataset: {ds_name}")
    ad = load_dataset(path)
    log(f"  Cells: {ad.n_obs:,}  Sex dist: {ad.obs[GROUP_KEY].value_counts().to_dict()}")
    X, y, g, syn = extract_scaled(ad)
    sil      = safe_silhouette(X, y, syn)
    cls_full = evaluate_classifier(X, y, g)
    if cls_full is None:
        log(f"  Degenerate split -- skipping"); continue

    try:
        sel_idx, min_g, valid_groups = strict_sex_balance_indices(ad)
        ad_strict  = ad[sel_idx].copy()
        Xs, ys, gs, syns = extract_scaled(ad_strict)
        sil_strict = safe_silhouette(Xs, ys, syns)
        cls_strict = evaluate_classifier(Xs, ys, gs)
    except RuntimeError as e:
        log(f"  Strict balance failed: {e}")
        cls_strict = None; sil_strict = np.nan; min_g = 0; valid_groups = []

    lc_df = compute_learning_curves(X, y, ds_name)
    all_lc_rows.append(lc_df)

    report_lines += [f"\nDATASET: {ds_name}\n" + "-"*60,
                     f"Cells: {len(y):,}  Synthetic: {syn.sum():,}  Silhouette: {sil}",
                     f"Accuracy: {cls_full['accuracy']}  F1: {cls_full['macro_f1']}  Worst sex: {cls_full['worst_sex']} ({cls_full['worst_sex_accuracy']})"]
    if cls_strict:
        report_lines.append(f"Strict balanced (min={min_g}, groups={valid_groups}): acc={cls_strict['accuracy']} f1={cls_strict['macro_f1']}")

    summary_rows.append({
        "dataset": ds_name, "n_labeled": len(y), "n_synthetic": int(syn.sum()),
        "silhouette": sil, "accuracy": cls_full["accuracy"],
        "balanced_accuracy": cls_full["balanced_accuracy"], "macro_f1": cls_full["macro_f1"],
        "worst_sex": cls_full["worst_sex"], "worst_sex_accuracy": cls_full["worst_sex_accuracy"],
        "strict_accuracy":  cls_strict["accuracy"]          if cls_strict else np.nan,
        "strict_macro_f1":  cls_strict["macro_f1"]          if cls_strict else np.nan,
        "strict_worst_sex": cls_strict["worst_sex"]         if cls_strict else "NA",
        "strict_worst_acc": cls_strict["worst_sex_accuracy"] if cls_strict else np.nan,
    })

runtime_min = round((time.time() - t_start) / 60, 2)
report_lines.append(f"\n{'='*70}\nSTEP 5 COMPLETE (runtime: {runtime_min} min)\n")
(OUTDIR / "step5_fairness_stress_test_scgpt_sex.txt").write_text("\n".join(report_lines))
pd.DataFrame(summary_rows).to_csv(OUTDIR / "step5_summary_scgpt_sex.csv", index=False)
lc_nonempty = [df for df in all_lc_rows if not df.empty]
if lc_nonempty:
    pd.concat(lc_nonempty, ignore_index=True).to_csv(OUTDIR / "step5_learning_curves_scgpt_sex.csv", index=False)
log("Done.")
