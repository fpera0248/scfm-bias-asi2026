#!/usr/bin/env python3
"""
STEP 4 -- External Validation Disease + Cell-Type Classification (CRC ETHNICITY, scGPT)

Changes vs Geneformer ethnicity version:
  [SCGPT 1] BASE path       -> scGPT/ethnicity_scGPT_workflow
  [SCGPT 2] EMB_KEY         -> X_scGPT
  [SCGPT 3] LABELED_DIR     -> step3b_labeled/
  [SCGPT 4] PILOTS          -> _labeled_scgpt.h5ad filenames
  [SCGPT 5] VALIDATION_FILE -> ILD_Ethnicity_External_Validation_12500_scgpt.h5ad
  [SCGPT 6] OUTDIR          -> step4_external_validation_scgpt
"""

import copy
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

BASE        = Path("/data/scGPT/augmented_CRC/ethnicity_scGPT_workflow")
LABELED_DIR = BASE / "step3b_labeled"                            # [SCGPT 3]
OUTDIR      = BASE / "step4_external_validation_scgpt"           # [SCGPT 6]
OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "CRC_Eth_Pilot"

PILOTS = {                                                       # [SCGPT 4]
    "Proportional_2497":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_2497_labeled_scgpt.h5ad",
    "Downsampled_48Each":         LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_48Each_labeled_scgpt.h5ad",
    "BalancedUpsampled_1880Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_1880Each_labeled_scgpt.h5ad",
    "BalancedAugmented_1880Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_1880Each_labeled_scgpt.h5ad",
}

VALIDATION_FILE = BASE / "CRC_Eth_External_Validation_8572_scgpt.h5ad"  # [SCGPT 5]

CSV_FILE = OUTDIR / "step4_external_validation_results_scgpt.csv"
TXT_FILE = OUTDIR / "step4_external_validation_log_scgpt.txt"

EMB_KEY       = "X_scGPT"                                       # [SCGPT 2]
DISEASE_COL   = "disease"
CELLTYPE_COL  = "cell_type"
ETHNICITY_COL = "self_reported_ethnicity"

DISEASE_GROUPS = {"african american", "asian", "european american", "hispanic or latin"}

RANDOM_STATE   = 42
EOS_NEIGHBORS  = 5
EOS_MULTIPLIER = 0.50
AR_BINS        = 30

MODEL_TEMPLATES = {
    "LogReg":       LogisticRegression(max_iter=2000, solver="lbfgs",
                                       class_weight="balanced", n_jobs=None),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
}

STRATEGIES = {
    "Baseline": dict(use_ar=False, use_eos=False),
    "AR":       dict(use_ar=True,  use_eos=False),
    "EOS":      dict(use_ar=False, use_eos=True),
    "AR+EOS":   dict(use_ar=True,  use_eos=True),
}

log_fh = open(TXT_FILE, "w")

def log(msg: str):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    print(line, file=log_fh, flush=True)

def canonicalize_ethnicity(series):
    return series.astype(str).str.strip().str.lower()

def make_binary_disease(x):
    if pd.isna(x):
        return np.nan
    return "normal" if str(x).lower().strip() == "normal" else "disease"

def load_pilot(name, path):
    if not path.exists():
        raise FileNotFoundError(f"Pilot file not found: {path}")
    ad = sc.read_h5ad(path)
    if EMB_KEY not in ad.obsm:
        raise RuntimeError(f"{name}: missing {EMB_KEY} in obsm")
    X           = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    eth         = canonicalize_ethnicity(ad.obs[ETHNICITY_COL])
    disease_bin = ad.obs[DISEASE_COL].apply(make_binary_disease)
    ct          = ad.obs[CELLTYPE_COL].astype(str)
    log(f"  {name}: {ad.n_obs:,} cells, {eth.nunique()} ethnicities, {ct.nunique()} cell types")
    return X, eth.to_numpy(), disease_bin.to_numpy(), ct.to_numpy()

def load_validation(path):
    if not path.exists():
        raise FileNotFoundError(f"Validation file not found: {path}")
    ad = sc.read_h5ad(path)
    if EMB_KEY not in ad.obsm:
        raise RuntimeError(f"Validation: missing {EMB_KEY} in obsm")
    X           = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    eth         = canonicalize_ethnicity(ad.obs[ETHNICITY_COL])
    disease_bin = ad.obs[DISEASE_COL].apply(make_binary_disease)
    ct          = ad.obs[CELLTYPE_COL].astype(str)
    log(f"Validation: {ad.n_obs:,} cells, {eth.nunique()} ethnicities, {ct.nunique()} cell types")
    log(f"  Ethnicity dist: {pd.Series(eth).value_counts().to_dict()}")
    return X, eth.to_numpy(), disease_bin.to_numpy(), ct.to_numpy()

def calculate_ar_weights(X_train, bins=AR_BINS):
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
        log(f"    EOS: minority '{target_group}' has only {X_min.shape[0]} cells -- skipping.")
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
    return (
        np.vstack([X_train, synth]),
        np.concatenate([y_train, y_new]),
        np.concatenate([g_train, g_new]),
    )

def evaluate_external(model_template, X_train, y_train, g_train,
                      X_test, y_test, g_test, use_ar=False, use_eos=False):
    mask = pd.Series(y_train).notna().to_numpy() & ~np.isnan(X_train).any(axis=1)
    X_tr, y_tr, g_tr = X_train[mask], y_train[mask], g_train[mask]
    if len(np.unique(y_tr)) < 2:
        return None
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_test)
    if use_eos:
        minority = pd.Series(g_tr).value_counts().idxmin()
        X_tr, y_tr, g_tr = apply_eos(X_tr, y_tr, g_tr, minority)
    weights = calculate_ar_weights(X_tr) if use_ar else None
    model   = copy.deepcopy(model_template)
    if weights is not None:
        try:
            model.fit(X_tr, y_tr, sample_weight=weights)
        except TypeError:
            model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average="macro")
    per_group = {}
    for grp in np.unique(g_test):
        m = g_test == grp
        if m.sum() > 0:
            per_group[grp] = accuracy_score(y_test[m], preds[m])
    if per_group:
        worst_grp = min(per_group, key=per_group.get)
        worst_acc = per_group[worst_grp]
    else:
        worst_grp, worst_acc = "n/a", np.nan
    return dict(accuracy=acc, macro_f1=f1, per_group=per_group,
                worst_group=worst_grp, worst_acc=worst_acc,
                n_train=len(y_tr), n_test=len(y_test))


def main():
    warnings.filterwarnings("ignore")

    log("=" * 70)
    log("STEP 4 -- External Validation Classification (CRC ETHNICITY, scGPT)")
    log("=" * 70)

    log("\nLoading validation set ...")
    X_val, eth_val, dis_val, ct_val = load_validation(VALIDATION_FILE)

    disease_mask = pd.Series(eth_val).isin(DISEASE_GROUPS).to_numpy()
    X_val_dis   = X_val[disease_mask]
    eth_val_dis = eth_val[disease_mask]
    dis_val_dis = dis_val[disease_mask]
    log(f"  Disease task validation subset: {X_val_dis.shape[0]:,} cells ({sorted(set(eth_val_dis))})")
    log(f"    Disease dist: {pd.Series(dis_val_dis).value_counts().to_dict()}")
    log(f"  Cell-type task validation: {X_val.shape[0]:,} cells, {len(np.unique(ct_val))} types")

    rows = []

    for pname, ppath in PILOTS.items():
        log(f"\n{'='*70}")
        log(f"Pilot: {pname}")
        try:
            X_pilot, eth_pilot, dis_pilot, ct_pilot = load_pilot(pname, ppath)
        except (FileNotFoundError, RuntimeError) as err:
            log(f"  SKIP: {err}")
            continue

        pilot_disease_mask = pd.Series(eth_pilot).isin(DISEASE_GROUPS).to_numpy()
        X_pilot_dis   = X_pilot[pilot_disease_mask]
        eth_pilot_dis = eth_pilot[pilot_disease_mask]
        dis_pilot_dis = dis_pilot[pilot_disease_mask]
        log(f"  Disease training subset: {X_pilot_dis.shape[0]:,} cells (filtered to {sorted(DISEASE_GROUPS)})")

        for model_name, model_tmpl in MODEL_TEMPLATES.items():
            for strat_name, flags in STRATEGIES.items():
                log(f"  {model_name:12s} | {strat_name:8s}")

                dres = evaluate_external(
                    model_tmpl,
                    X_pilot_dis, dis_pilot_dis, eth_pilot_dis,
                    X_val_dis,   dis_val_dis,   eth_val_dis,
                    use_ar=flags["use_ar"], use_eos=flags["use_eos"],
                )

                cres = evaluate_external(
                    model_tmpl,
                    X_pilot, ct_pilot, eth_pilot,
                    X_val,   ct_val,   eth_val,
                    use_ar=flags["use_ar"], use_eos=flags["use_eos"],
                )

                row = {"pilot": pname, "model": model_name, "strategy": strat_name}

                if dres is not None:
                    row.update({
                        "disease_accuracy":  dres["accuracy"],
                        "disease_macro_f1":  dres["macro_f1"],
                        "disease_worst_eth": dres["worst_group"],
                        "disease_worst_acc": dres["worst_acc"],
                        "disease_n_train":   dres["n_train"],
                        "disease_n_test":    dres["n_test"],
                    })
                    for eth, a in dres["per_group"].items():
                        row[f"disease_acc_{eth}"] = a
                    log(f"    disease: acc={dres['accuracy']:.3f} f1={dres['macro_f1']:.3f} worst={dres['worst_group']}:{dres['worst_acc']:.3f}")
                else:
                    log(f"    disease: SKIPPED (single-class training)")

                if cres is not None:
                    row.update({
                        "celltype_accuracy":  cres["accuracy"],
                        "celltype_macro_f1":  cres["macro_f1"],
                        "celltype_worst_eth": cres["worst_group"],
                        "celltype_worst_acc": cres["worst_acc"],
                        "celltype_n_train":   cres["n_train"],
                        "celltype_n_test":    cres["n_test"],
                    })
                    for eth, a in cres["per_group"].items():
                        row[f"celltype_acc_{eth}"] = a
                    log(f"    celltype: acc={cres['accuracy']:.3f} f1={cres['macro_f1']:.3f} worst={cres['worst_group']}:{cres['worst_acc']:.3f}")
                else:
                    log(f"    celltype: SKIPPED (single-class training)")

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_FILE, index=False)

    log(f"\n{'='*70}")
    log(f"STEP 4 COMPLETE (CRC ETHNICITY, scGPT)")
    log(f"  Results -> {CSV_FILE.name}")
    log(f"  Log     -> {TXT_FILE.name}")

    summary_cols = ["pilot", "model", "strategy",
                    "disease_accuracy", "disease_macro_f1",
                    "celltype_accuracy", "celltype_macro_f1"]
    available = [c for c in summary_cols if c in df.columns]
    print("\n" + df[available].to_string(index=False))

    log_fh.close()


if __name__ == "__main__":
    main()
