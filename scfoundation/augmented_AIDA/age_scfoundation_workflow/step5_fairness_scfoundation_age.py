#!/usr/bin/env python3
"""
STEP 5 — Fairness Stress Test under Strict Age × Disease Balance (AGE)
scFoundation workflow

Changes vs sex version:
  [AGE 1] BASE/OUTDIR  -> age_scfoundation_workflow
  [AGE 2] GROUP_KEY    -> "age_bin_10yr"
  [AGE 3] DATASETS     -> age filenames (747Each, 230Each, 2495)
  [AGE 4] Output files -> *_age_geneformer suffix
  [AGE 5] Log strings updated for age context
"""

import scanpy as sc
import numpy as np
import pandas as pd
import pathlib
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, silhouette_score

warnings.filterwarnings("ignore")

BASE = pathlib.Path(
    "/data/scfoundation/augmented_AIDA/age_scfoundation_workflow"
)

LABELED_DIR = BASE / "step3b_labeled"
OUTDIR      = BASE / "step5_outputs_age_scfoundation"              # [AGE 4]
OUTDIR.mkdir(exist_ok=True)

EMB_KEY      = "X_scfoundation"
GROUP_KEY    = "age_bin_10yr"                                    # [AGE 2]
DISEASE_COL  = "disease"
CELLTYPE_COL = "cell_type"
SOURCE_COL   = "source"

OUTPUT_BASE = "AIDA_Age_Pilot"                                    # [AGE 1]

AGE_COL_CANDIDATES = ["age_bin_10yr", "age_bin", "age_group", "development_stage"]

DATASETS = {                                                     # [AGE 3]
    "BalancedAugmented_747Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_747Each_labeled_scfoundation.h5ad",
    "Proportional_2498":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_2498_labeled_scfoundation.h5ad",
    "BalancedUpsampled_747Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_747Each_labeled_scfoundation.h5ad",
    "Downsampled_230Each":         LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_230Each_labeled_scfoundation.h5ad",
}

RANDOM_STATE = 42
TEST_SIZE    = 0.20
MIN_REQUIRED = 10
LC_CV_FOLDS  = 5
LC_N_POINTS  = 8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def detect_age_col(obs):                                         # [AGE 2]
    for c in AGE_COL_CANDIDATES:
        if c in obs.columns: return c
    raise RuntimeError(f"No age column found. Available: {list(obs.columns)}")

def canonicalize_age(series):
    return series.astype(str).str.strip().str.lower()

def to_binary_disease(x):
    if pd.isna(x):
        return np.nan
    return "normal" if str(x).lower().strip() == "normal" else "disease"

def load_dataset(path):
    log(f"  Loading: {path.name}")
    ad = sc.read_h5ad(path)
    if not ad.obs_names.is_unique:
        ad.obs_names_make_unique()
    age_col = detect_age_col(ad.obs)
    ad.obs[GROUP_KEY] = canonicalize_age(ad.obs[age_col])
    if DISEASE_COL not in ad.obs.columns:
        raise RuntimeError(f"Missing '{DISEASE_COL}' column in {path.name}")
    ad.obs["disease_binary"] = ad.obs[DISEASE_COL].apply(to_binary_disease)
    if SOURCE_COL in ad.obs.columns:
        ad.obs["is_synthetic"] = (ad.obs[SOURCE_COL] == "synthetic")
    else:
        ad.obs["is_synthetic"] = False
    return ad

def extract_scaled(ad):
    if EMB_KEY not in ad.obsm:
        raise RuntimeError(f"Missing embedding '{EMB_KEY}'")
    X   = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    y   = ad.obs["disease_binary"]
    g   = ad.obs[GROUP_KEY]
    syn = ad.obs["is_synthetic"]

    nan_rows  = np.isnan(X).any(axis=1)
    zero_rows = (np.abs(X).sum(axis=1) == 0)
    bad_rows  = nan_rows | zero_rows
    if bad_rows.any():
        log(f"  Dropping {bad_rows.sum()} cells (NaN or zero embeddings)")
        keep = ~bad_rows
        X, y, g, syn = X[keep], y[keep], g[keep], syn[keep]

    valid = pd.notna(y) & pd.notna(g)
    if valid.sum() < 20:
        raise RuntimeError(f"Too few valid cells after filtering: {valid.sum()}")

    Xv   = X[valid.values]
    yv   = np.asarray(y[valid]).astype(str)
    gv   = np.asarray(g[valid]).astype(str)
    synv = np.asarray(syn[valid]).astype(bool)
    Xs   = StandardScaler().fit_transform(Xv)
    return Xs, yv, gv, synv

def safe_silhouette(X, y, syn):
    mask = (~syn) & (pd.Series(y).notna().values)
    if mask.sum() < 50:
        return np.nan
    yv = np.asarray(y)[mask]
    if len(np.unique(yv)) < 2:
        return np.nan
    return round(float(silhouette_score(X[mask], yv, sample_size=min(5000, mask.sum()))), 4)

def evaluate_classifier(X, y, g, seed=RANDOM_STATE):
    if len(np.unique(y)) < 2 or X.shape[0] < 40:
        return None
    counts   = pd.Series(y).value_counts()
    stratify = y if (counts >= 2).all() else None
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, g, test_size=TEST_SIZE, stratify=stratify, random_state=seed)
    clf = LogisticRegression(max_iter=8000, solver="saga", class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    per_grp = {grp: accuracy_score(y_te[g_te==grp], preds[g_te==grp])
               for grp in np.unique(g_te) if (g_te==grp).sum() >= 5}
    worst_grp = min(per_grp, key=per_grp.get) if per_grp else "NA"
    worst_acc = per_grp.get(worst_grp, np.nan)
    return {
        "accuracy":              round(float(accuracy_score(y_te, preds)), 4),
        "balanced_accuracy":     round(float(balanced_accuracy_score(y_te, preds)), 4),
        "macro_f1":              round(float(f1_score(y_te, preds, average="macro")), 4),
        "worst_age_bin":         str(worst_grp),                # [AGE 2]
        "worst_age_bin_accuracy": round(float(worst_acc), 4) if pd.notna(worst_acc) else np.nan,
        "per_age_accuracy":      {k: round(float(v), 4) for k, v in per_grp.items()},
        "n_train":               int(X_tr.shape[0]),
        "n_test":                int(X_te.shape[0]),
    }

def strict_age_balance_indices(ad, min_required=MIN_REQUIRED, seed=RANDOM_STATE):  # [AGE 2]
    rng = np.random.default_rng(seed)
    obs = ad.obs.copy()
    obs = obs[pd.notna(obs["disease_binary"])]
    tab = obs.groupby([GROUP_KEY, "disease_binary"], observed=False).size().reset_index(name="count")
    valid_groups = []
    for grp in tab[GROUP_KEY].unique():
        sub = tab[tab[GROUP_KEY] == grp]
        if sub["disease_binary"].nunique() == 2 and sub["count"].min() > 0:
            valid_groups.append(grp)
    if len(valid_groups) < 2:
        raise RuntimeError(f"Not enough age bins with both disease classes. Found: {valid_groups}")
    obs2   = obs[obs[GROUP_KEY].isin(valid_groups)].copy()
    groups = obs2.groupby([GROUP_KEY, "disease_binary"], observed=False)
    min_g  = int(groups.size().min())
    if min_g < min_required:
        raise RuntimeError(
            f"min_group_size too small: {min_g} (< {min_required}). "
            f"Group breakdown: {groups.size().to_dict()}")
    selected = []
    for (_, _), idx in groups.groups.items():
        chosen = rng.choice(list(idx), size=min_g, replace=False)
        selected.extend(chosen)
    return np.array(selected), min_g, valid_groups

def compute_learning_curves(X, y, dataset_name):
    log(f"  Computing learning curves for {dataset_name}...")
    counts = pd.Series(y).value_counts()
    if counts.min() < LC_CV_FOLDS * 2:
        log(f"  Skipping LC — too few cells per class ({counts.min()} < {LC_CV_FOLDS * 2})")
        return pd.DataFrame()
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
                results.append({
                    "dataset": dataset_name, "model": model_name, "n_train": int(n),
                    "train_f1_mean": round(float(tr_scores[i].mean()), 4),
                    "train_f1_std":  round(float(tr_scores[i].std()),  4),
                    "val_f1_mean":   round(float(val_scores[i].mean()), 4),
                    "val_f1_std":    round(float(val_scores[i].std()),  4),
                })
        except Exception as e:
            log(f"  WARNING: Learning curve failed for {model_name}: {e}")
    return pd.DataFrame(results)

def plot_learning_curves(lc_df, outdir):
    datasets = lc_df["dataset"].unique()
    models   = lc_df["model"].unique()
    colors   = {"LogReg": "#2196F3", "RandomForest": "#FF5722"}
    n_panels = len(datasets)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, dname in zip(axes, datasets):
        sub = lc_df[lc_df["dataset"] == dname]
        for mname in models:
            msub = sub[sub["model"] == mname]
            if msub.empty:
                continue
            col = colors.get(mname, "gray")
            ax.plot(msub["n_train"], msub["val_f1_mean"], "-o", color=col,
                    label=f"{mname} (val)", linewidth=2)
            ax.fill_between(msub["n_train"],
                msub["val_f1_mean"] - msub["val_f1_std"],
                msub["val_f1_mean"] + msub["val_f1_std"], alpha=0.15, color=col)
            ax.plot(msub["n_train"], msub["train_f1_mean"], "--", color=col,
                    alpha=0.4, label=f"{mname} (train)")
        ax.set_title(dname.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("Training set size (cells)")
        ax.set_ylabel("Macro-F1")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Learning Curves — Age Fairness (Pilot, Geneformer)", fontsize=10, y=1.02)  # [AGE 5]
    plt.tight_layout()
    out = outdir / "step5_learning_curves_age_scfoundation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure saved -> {out.name}")
    return out


t_start = time.time()
report_lines = ["STEP 5 -- FAIRNESS STRESS TEST (AGE, Geneformer)\n" + "=" * 70 + "\n"]  # [AGE 5]
all_lc_rows  = []
summary_rows = []

for ds_name, path in DATASETS.items():
    if not path.exists():
        log(f"  Skipping {ds_name} -- file not found: {path.name}")
        continue

    log(f"\n{'='*60}")
    log(f"Dataset: {ds_name}")

    ad = load_dataset(path)
    log(f"  Cells: {ad.n_obs:,}  |  Age dist: {ad.obs[GROUP_KEY].value_counts().sort_index().to_dict()}")

    X, y, g, syn = extract_scaled(ad)
    log(f"  Labeled cells: {len(y):,}")

    sil      = safe_silhouette(X, y, syn)
    cls_full = evaluate_classifier(X, y, g)
    if cls_full is None:
        log(f"  Degenerate split -- skipping {ds_name}")
        continue

    try:
        sel_idx, min_g, valid_groups = strict_age_balance_indices(ad)
        ad_strict = ad[sel_idx].copy()
        Xs, ys, gs, syns = extract_scaled(ad_strict)
        sil_strict = safe_silhouette(Xs, ys, syns)
        cls_strict = evaluate_classifier(Xs, ys, gs)
    except RuntimeError as e:
        log(f"  Strict balance failed: {e}")
        cls_strict = None; sil_strict = np.nan; min_g = 0; valid_groups = []

    lc_df = compute_learning_curves(X, y, ds_name)
    all_lc_rows.append(lc_df)

    report_lines.append(f"\nDATASET: {ds_name}\n" + "-" * 60)
    report_lines.append(f"Cells (labeled): {len(y):,}  |  Synthetic: {syn.sum():,}")
    report_lines.append(f"Silhouette (real only): {sil}")
    report_lines.append(f"\nFull evaluation (LogReg):")
    report_lines.append(f"  Accuracy:          {cls_full['accuracy']}")
    report_lines.append(f"  Balanced accuracy: {cls_full['balanced_accuracy']}")
    report_lines.append(f"  Macro-F1:          {cls_full['macro_f1']}")
    report_lines.append(f"  Worst age bin:     {cls_full['worst_age_bin']} ({cls_full['worst_age_bin_accuracy']})")  # [AGE 2]
    report_lines.append("  Per-age accuracy:")
    for k, v in sorted(cls_full["per_age_accuracy"].items()):
        report_lines.append(f"    {k}: {v}")

    if cls_strict:
        report_lines.append(f"\nStrict balanced (min per group={min_g}, groups={valid_groups}):")
        report_lines.append(f"  Cells: {len(ys):,}  |  Silhouette: {sil_strict}")
        report_lines.append(f"  Accuracy:          {cls_strict['accuracy']}")
        report_lines.append(f"  Balanced accuracy: {cls_strict['balanced_accuracy']}")
        report_lines.append(f"  Macro-F1:          {cls_strict['macro_f1']}")
        report_lines.append(f"  Worst age bin:     {cls_strict['worst_age_bin']} ({cls_strict['worst_age_bin_accuracy']})")
        report_lines.append("  Per-age accuracy:")
        for k, v in sorted(cls_strict["per_age_accuracy"].items()):
            report_lines.append(f"    {k}: {v}")

    if not lc_df.empty:
        report_lines.append("\nLearning curve summary (val Macro-F1 min/max):")
        for mname in lc_df["model"].unique():
            msub = lc_df[lc_df["model"] == mname].sort_values("n_train")
            if len(msub) >= 2:
                f1_min = msub.iloc[0]["val_f1_mean"]; f1_max = msub.iloc[-1]["val_f1_mean"]
                n_min  = msub.iloc[0]["n_train"];     n_max  = msub.iloc[-1]["n_train"]
                verdict = "STILL RISING (data-limited)" if f1_max > f1_min + 0.02 else "plateauing"
                report_lines.append(f"  {mname}: F1 {f1_min:.3f} @ N={n_min} -> {f1_max:.3f} @ N={n_max}  [{verdict}]")

    summary_rows.append({
        "dataset": ds_name, "n_labeled": len(y), "n_synthetic": int(syn.sum()),
        "silhouette": sil,
        "accuracy": cls_full["accuracy"], "balanced_accuracy": cls_full["balanced_accuracy"],
        "macro_f1": cls_full["macro_f1"],
        "worst_age_bin": cls_full["worst_age_bin"],                          # [AGE 2]
        "worst_age_bin_accuracy": cls_full["worst_age_bin_accuracy"],
        "strict_accuracy":      cls_strict["accuracy"]              if cls_strict else np.nan,
        "strict_macro_f1":      cls_strict["macro_f1"]              if cls_strict else np.nan,
        "strict_worst_age_bin": cls_strict["worst_age_bin"]         if cls_strict else "NA",
        "strict_worst_acc":     cls_strict["worst_age_bin_accuracy"] if cls_strict else np.nan,
    })

runtime_min = round((time.time() - t_start) / 60, 2)
report_lines.append(f"\n{'='*70}\nSTEP 5 COMPLETE (runtime: {runtime_min} min)\n")

out_txt = OUTDIR / "step5_fairness_stress_test_age_scfoundation.txt"
out_txt.write_text("\n".join(report_lines))
log(f"\nReport  -> {out_txt}")

out_csv = OUTDIR / "step5_summary_age_scfoundation.csv"
pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
log(f"Summary -> {out_csv}")

if all_lc_rows:
    lc_nonempty = [df for df in all_lc_rows if not df.empty]
    if lc_nonempty:
        lc_all = pd.concat(lc_nonempty, ignore_index=True)
        out_lc_csv = OUTDIR / "step5_learning_curves_age_scfoundation.csv"
        lc_all.to_csv(out_lc_csv, index=False)
        log(f"LC CSV  -> {out_lc_csv}")
        plot_learning_curves(lc_all, OUTDIR)

log("\nDone.")
