#!/usr/bin/env python3
"""STEP 7 — Representation Quality & Fairness Diagnostics (SEX, scFoundation)"""

import pathlib, time, warnings
import numpy as np, pandas as pd, scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = pathlib.Path("/oscar/home/fperalta/data/fperalta/scfoundation/augmentedv4/sex_scfoundation_workflow")
LABELED_DIR = BASE / "step3b_labeled"
OUTDIR = BASE / "step7_representation_diagnostics_scfoundation_sex"; OUTDIR.mkdir(exist_ok=True)
OUTPUT_BASE = "ILD_Sex_Pilot"
FILES = {
    "Proportional_1999":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_1999_labeled_scfoundation.h5ad",
    "BalancedAugmented_1413Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_1413Each_labeled_scfoundation.h5ad",
    "BalancedUpsampled_1413Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_1413Each_labeled_scfoundation.h5ad",
    "Downsampled_586Each":        LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_586Each_labeled_scfoundation.h5ad",
}
EMB_KEY = "X_scfoundation"; SEX_KEY = "sex"; CELL_KEY = "cell_type"
SOURCE_COL = "source"; KNN_K = 15; MIN_CT_SIZE = 20; RANDOM_STATE = 42

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def load_dataset(label, path):
    ad = sc.read_h5ad(path)
    if not ad.obs_names.is_unique: ad.obs_names_make_unique()
    ad.obs[SEX_KEY] = ad.obs[SEX_KEY].astype(str).str.strip().str.lower()
    ad.obs["is_synthetic"] = (ad.obs[SOURCE_COL] == "synthetic") if SOURCE_COL in ad.obs.columns else False
    emb = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    bad_mask = np.isnan(emb).any(axis=1) | (np.abs(emb).sum(axis=1) == 0)
    if bad_mask.any(): ad = ad[~bad_mask].copy()
    log(f"  {label}: {ad.n_obs:,} cells | {ad.obs[SEX_KEY].value_counts().to_dict()}")
    return ad
def celltype_purity(ad):
    emb = np.array(ad.obsm[EMB_KEY], dtype=np.float32); cell = ad.obs[CELL_KEY].astype(str).values
    nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(emb); neigh = nn.kneighbors(return_distance=False)[:, 1:]
    purity = np.mean(cell[neigh] == cell[:, None], axis=1)
    df = pd.DataFrame({SEX_KEY: ad.obs[SEX_KEY].astype(str).values, "purity": purity})
    return df.groupby(SEX_KEY, observed=False)["purity"].mean()
def within_celltype_mixing(ad):
    results = []
    for ct, sub_idx in ad.obs.groupby(CELL_KEY, observed=False).groups.items():
        if len(sub_idx) < MIN_CT_SIZE: continue
        pos = ad.obs.index.get_indexer_for(sub_idx)
        emb = np.array(ad.obsm[EMB_KEY], dtype=np.float32)[pos]; sex = ad.obs.iloc[pos][SEX_KEY].astype(str).values
        nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(emb); neigh = nn.kneighbors(return_distance=False)[:, 1:]
        mixing = np.mean(sex[neigh] != sex[:, None], axis=1)
        df = pd.DataFrame({SEX_KEY: sex, "mixing": mixing})
        results.append(df.groupby(SEX_KEY, observed=False)["mixing"].mean())
    if not results: return pd.Series(dtype=float)
    return pd.concat(results, axis=1).mean(axis=1)
def celltype_linear_probe(ad):
    X = np.array(ad.obsm[EMB_KEY], dtype=np.float32); y = ad.obs[CELL_KEY].astype(str).values; g = ad.obs[SEX_KEY].astype(str).values
    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=3000, solver="saga", class_weight="balanced", n_jobs=-1)
    clf.fit(X, y); preds = clf.predict(X)
    df = pd.DataFrame({SEX_KEY: g, "y_true": y, "y_pred": preds})
    return pd.Series({grp: round(float(f1_score(sub["y_true"], sub["y_pred"], average="macro")), 4)
                      for grp in np.unique(g) for sub in [df[df[SEX_KEY] == grp]] if sub["y_true"].nunique() >= 2})

def main():
    log("="*70); log("STEP 7 -- Representation Quality & Fairness Diagnostics (SEX, scFoundation)"); log("="*70)
    datasets = {}
    for label, path in FILES.items():
        if not path.exists(): log(f"  Skipping {label}"); continue
        datasets[label] = load_dataset(label, path)
    if "Proportional_1999" not in datasets: raise RuntimeError("Proportional_1999 required as reference baseline.")
    all_results = []
    for label, ad in datasets.items():
        log(f"\n>> Computing diagnostics for {label}")
        purity = celltype_purity(ad); mixing = within_celltype_mixing(ad); probe = celltype_linear_probe(ad)
        df = pd.concat([purity, mixing, probe], axis=1)
        df.columns = ["celltype_purity", "within_ct_sex_mixing", "celltype_macroF1"]
        df = df.reset_index().rename(columns={"index": SEX_KEY}); df["dataset"] = label; all_results.append(df)
        log(f"  Purity: {purity.to_dict()}"); log(f"  Mixing: {mixing.to_dict()}"); log(f"  ProbeF1: {probe.to_dict()}")
    df_all = pd.concat(all_results, ignore_index=True)
    pivot = df_all.pivot_table(index=SEX_KEY, columns="dataset",
        values=["celltype_purity", "within_ct_sex_mixing", "celltype_macroF1"], observed=False)
    ref = "Proportional_1999"
    for metric in ["celltype_purity", "within_ct_sex_mixing", "celltype_macroF1"]:
        for ds in [d for d in datasets if d != ref]:
            col_ds = (metric, ds); col_ref = (metric, ref)
            if col_ds in pivot.columns and col_ref in pivot.columns:
                pivot[(metric, f"delta_{ds}_vs_prop")] = pivot[col_ds] - pivot[col_ref]
    df_all.to_csv(OUTDIR / "step7_per_sex_diagnostics_scfoundation.csv", index=False)
    delta_cols = [c for c in pivot.columns if "delta" in str(c[1])]
    lines = ["STEP 7 -- REPRESENTATION QUALITY & FAIRNESS DIAGNOSTICS (SEX, scFoundation)", "="*70, "",
             "NOTE: With 2 sex groups, random expected kNN mixing is ~0.50.",
             "", "="*70, "FULL PIVOT TABLE", "="*70, pivot.to_string(),
             "", "="*70, "DELTA SUMMARY (vs Proportional_1999)", "="*70,
             pivot[delta_cols].sort_index().to_string() if delta_cols else "No delta columns produced."]
    (OUTDIR / "step7_summary_scfoundation_sex.txt").write_text("\n".join(lines))
    log("STEP 7 COMPLETE")

if __name__ == "__main__":
    main()
