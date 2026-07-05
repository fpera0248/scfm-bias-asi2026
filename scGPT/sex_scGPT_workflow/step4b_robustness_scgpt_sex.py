#!/usr/bin/env python3
"""
STEP 4b — Robustness Stress Test (SEX, scGPT)

Changes vs ethnicity scGPT version:
  [SEX 1] BASE/OUTDIR  -> sex_scGPT_workflow
  [SEX 2] EMB_KEY      -> X_scGPT
  [SEX 3] DATASETS     -> sex filenames
  [SEX 4] GROUP_COL    -> "sex"
  [SEX 5] KNOWN_GROUPS -> ["female", "male"]
"""

import gc, hashlib, pathlib, logging, time, warnings
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline

BASE = pathlib.Path("/data/scGPT/sex_scGPT_workflow")

OUTPUT_BASE = "ILD_Sex_Pilot"
LABELED_DIR = BASE / "step3b_labeled"

DATASETS = {
    "BalancedAugmented_1413Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_1413Each_labeled_scgpt.h5ad",
    "Proportional_1999":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_1999_labeled_scgpt.h5ad",
    "BalancedUpsampled_1413Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_1413Each_labeled_scgpt.h5ad",
    "Downsampled_586Each":        LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_586Each_labeled_scgpt.h5ad",
}

OUTDIR   = BASE / "step4b_model_robustness_tests_sex_scgpt"
OUTDIR.mkdir(exist_ok=True)
CSV_FILE = OUTDIR / "step4b_results_sex_labeled_scgpt.csv"
TXT_FILE = OUTDIR / "step4b_full_log_sex_labeled_scgpt.txt"

logger = logging.getLogger("STEP4B_SEX_SCGPT")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
fh  = logging.FileHandler(TXT_FILE); fh.setFormatter(fmt); logger.addHandler(fh)
sh  = logging.StreamHandler();       sh.setFormatter(fmt); logger.addHandler(sh)

warnings.filterwarnings("always")
logger.info("========== STEP 4b START (SEX, scGPT) ==========")

EMB_KEY   = "X_scGPT"
LABEL_KEY = "disease"
SEX_COL_CANDIDATES = ["sex", "Sex", "SEX", "gender", "Gender"]

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
N_REPEATS     = 3
EOS_NEIGHBORS = 10
EOS_MULT      = 0.50
AR_BINS       = 30
KNOWN_GROUPS  = ["female", "male"]

rng = np.random.default_rng(RANDOM_STATE)

def detect_sex_col(obs, fname):
    for c in SEX_COL_CANDIDATES:
        if c in obs.columns: return c
    raise RuntimeError(f"No sex column in {fname}.")

def canonicalize(series): return series.astype(str).str.strip().str.lower()
def to_binary_disease(x):
    if pd.isna(x): return np.nan
    return "normal" if str(x).lower().strip() == "normal" else "disease"

def hash_rows(X):
    return np.array([hashlib.md5(row.tobytes()).hexdigest() for row in X])

def calculate_ar_weights(X, bins=AR_BINS):
    ALPHA = 0.0001
    n, d  = X.shape
    dim_weights = np.zeros((n, d), dtype=np.float64)
    for k in range(d):
        vals = X[:, k]
        hist, edges = np.histogram(vals, bins=bins, density=True)
        bin_idx   = np.clip(np.digitize(vals, edges) - 1, 0, len(hist) - 1)
        prob_smooth = (1.0 - ALPHA) * hist[bin_idx] * (edges[1]-edges[0]) + ALPHA / n
        dim_weights[:, k] = 1.0 / prob_smooth
    weights  = dim_weights.max(axis=1)
    weights /= weights.mean()
    return weights.astype(np.float32)

def apply_eos_adv(X, y, g, target):
    mask  = (g == target)
    X_min = X[mask]; y_min = y[mask]; X_oth = X[~mask]
    if X_min.shape[0] < 10: return X, y, g
    nn = NearestNeighbors(n_neighbors=EOS_NEIGHBORS).fit(X_oth)
    _, neigh = nn.kneighbors(X_min)
    synth = []
    for _ in range(int(len(X_min) * EOS_MULT)):
        i = rng.integers(len(X_min)); j = rng.choice(neigh[i])
        synth.append(X_min[i] + rng.uniform(0., 1.) * (X_min[i] - X_oth[j]))
    if not synth: return X, y, g
    synth = np.vstack(synth)
    return (np.vstack([X, synth]),
            np.concatenate([y, rng.choice(y_min, len(synth))]),
            np.concatenate([g, np.repeat(target, len(synth))]))

def group_metrics(y_true, y_pred, groups):
    df    = pd.DataFrame({"y": y_true, "p": y_pred, "g": groups})
    accs  = df.groupby("g").apply(lambda d: accuracy_score(d["y"], d["p"]))
    worst = accs.idxmin()
    return accs.to_dict(), worst, float(accs.min())

def supports_sample_weight(model):
    try:    return "sample_weight" in model.fit.__code__.co_varnames
    except: return False

def make_models():
    return {
        "LogReg":     LogisticRegression(max_iter=5000, solver="saga", n_jobs=-1),
        "LinearSVM":  LinearSVC(max_iter=5000),
        "RBF_Approx": make_pipeline(RBFSampler(gamma=1.0, n_components=500, random_state=RANDOM_STATE), LinearSVC(max_iter=5000)),
        "MLP":        MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300, random_state=RANDOM_STATE),
    }

STRATS = {
    "Baseline":   dict(use_ar=False, eos=None),
    "AR":         dict(use_ar=True,  eos=None),
    "EOS_adv":    dict(use_ar=False, eos="adv"),
    "AR+EOS_adv": dict(use_ar=True,  eos="adv"),
}

rows = []
global_start = time.time()

for dname, path in DATASETS.items():
    if not path.exists():
        logger.warning(f"Skipping {dname}"); continue
    logger.info(f"\nLoading dataset: {dname}")
    ad = sc.read_h5ad(path)
    grp_col = detect_sex_col(ad.obs, path.name)
    ad.obs["_sex"]          = canonicalize(ad.obs[grp_col])
    ad.obs["disease_binary"] = ad.obs[LABEL_KEY].apply(to_binary_disease)
    ad = ad[~ad.obs["disease_binary"].isna()].copy()

    X0 = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    y0 = ad.obs["disease_binary"].to_numpy()
    g0 = ad.obs["_sex"].to_numpy()

    hashes = hash_rows(X0)
    _, uniq = np.unique(hashes, return_index=True)
    dup_removed = len(hashes) - len(uniq)
    if dup_removed / len(hashes) > 0.20:
        logger.warning(f"  Dedup fraction too high -- skipping dedup."); dup_removed = 0
    else:
        X0, y0, g0 = X0[uniq], y0[uniq], g0[uniq]

    bad_mask = np.isnan(X0).any(axis=1) | (np.abs(X0).sum(axis=1) == 0)
    if bad_mask.any():
        X0, y0, g0 = X0[~bad_mask], y0[~bad_mask], g0[~bad_mask]

    logger.info(f"  Cells: {len(y0):,}  Sex dist: {pd.Series(g0).value_counts().to_dict()}")

    for rep in range(N_REPEATS):
        uniq_counts = pd.Series(y0).value_counts()
        stratify    = y0 if (uniq_counts >= 2).all() else None
        Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
            X0, y0, g0, stratify=stratify, test_size=TEST_SIZE, random_state=RANDOM_STATE + rep)
        scaler = StandardScaler()
        Xtr    = scaler.fit_transform(Xtr)
        Xte    = scaler.transform(Xte)

        for strat, flags in STRATS.items():
            Xs, ys, gs = Xtr, ytr, gtr
            if flags["eos"] == "adv":
                minority = pd.Series(gs).value_counts().idxmin()
                Xs, ys, gs = apply_eos_adv(Xs, ys, gs, minority)
            weights = calculate_ar_weights(Xs) if flags["use_ar"] else None

            for mname, model in make_models().items():
                t0 = time.time()
                if weights is not None and supports_sample_weight(model):
                    model.fit(Xs, ys, sample_weight=weights); ar_applied = True
                else:
                    model.fit(Xs, ys); ar_applied = False if weights is not None else None
                pred = model.predict(Xte)
                acc  = accuracy_score(yte, pred)
                f1   = f1_score(yte, pred, average="macro")
                per, worst, wacc = group_metrics(yte, pred, gte)
                row = dict(dataset=dname, repeat=rep, model=mname, strategy=strat,
                           accuracy=float(acc), macro_f1=float(f1),
                           worst_sex=worst, worst_sex_acc=float(wacc),
                           dedup_removed=dup_removed, n_unique=int(X0.shape[0]),
                           n_train=int(Xtr.shape[0]), n_test=int(Xte.shape[0]),
                           ar_requested=bool(flags["use_ar"]), ar_applied=ar_applied,
                           runtime_sec=time.time() - t0)
                for grp in KNOWN_GROUPS:
                    row[f"per_sex_acc_{grp}"] = float(per.get(grp, np.nan))
                rows.append(row)
                logger.info(f"  {dname} | rep={rep} | {strat:12s} | {mname:12s} | acc={acc:.4f} f1={f1:.4f} worst={worst}:{wacc:.4f}")
                gc.collect()

pd.DataFrame(rows).to_csv(CSV_FILE, index=False)
total_min = (time.time() - global_start) / 60
logger.info(f"========== STEP 4b COMPLETE (SEX, scGPT) ==========  Runtime: {total_min:.2f} min")
df = pd.DataFrame(rows)
print("\n" + df.groupby(["dataset","model","strategy"])[["accuracy","macro_f1","worst_sex_acc"]].mean().round(4).reset_index().to_string(index=False))
