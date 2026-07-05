#!/usr/bin/env python3
"""STEP 4b — Robustness Stress Test (AGE, scFoundation)"""
import gc, hashlib, pathlib, logging, time, warnings
import numpy as np, pandas as pd, scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline

BASE = pathlib.Path("/data/scfoundation/augmented_AIDA/age_scfoundation_workflow")
OUTPUT_BASE = "AIDA_Age_Pilot"; LABELED_DIR = BASE / "step3b_labeled"
DATASETS = {
    "BalancedAugmented_747Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_747Each_labeled_scfoundation.h5ad",
    "Proportional_2498":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_2498_labeled_scfoundation.h5ad",
    "BalancedUpsampled_747Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_747Each_labeled_scfoundation.h5ad",
    "Downsampled_230Each":         LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_230Each_labeled_scfoundation.h5ad",
}
OUTDIR = BASE / "step4b_model_robustness_tests_age_scfoundation"; OUTDIR.mkdir(exist_ok=True)
CSV_FILE = OUTDIR / "step4b_results_age_labeled_scfoundation.csv"
TXT_FILE = OUTDIR / "step4b_full_log_age_labeled_scfoundation.txt"

logger = logging.getLogger("STEP4B_AGE_SCF"); logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
for h in [logging.FileHandler(TXT_FILE), logging.StreamHandler()]:
    h.setFormatter(fmt); logger.addHandler(h)

warnings.filterwarnings("always")
logger.info("========== STEP 4b START (AGE, scFoundation) ==========")

EMB_KEY = "X_scfoundation"; LABEL_KEY = "disease"
AGE_COL_CANDIDATES = ["age_bin_10yr", "age_bin", "age_group", "development_stage"]
RANDOM_STATE = 42; TEST_SIZE = 0.20; N_REPEATS = 3
EOS_NEIGHBORS = 10; EOS_MULT = 0.50; AR_BINS = 30
KNOWN_GROUPS = ["10_19", "20_29", "30_39", "40_49", "50_59", "60_69", "70_79"]
rng = np.random.default_rng(RANDOM_STATE)

def detect_age_col(obs, fname):
    for c in AGE_COL_CANDIDATES:
        if c in obs.columns: return c
    raise RuntimeError(f"No age column found in {fname}.")

def to_binary(x):
    if pd.isna(x): return np.nan
    return "normal" if str(x).lower().strip() == "normal" else "disease"

def hash_rows(X): return np.array([hashlib.md5(row.tobytes()).hexdigest() for row in X])

def ar_weights(X, bins=AR_BINS):
    ALPHA = 0.0001; n, d = X.shape; w = np.zeros((n, d), dtype=np.float64)
    for k in range(d):
        hist, edges = np.histogram(X[:, k], bins=bins, density=True)
        bi = np.clip(np.digitize(X[:, k], edges) - 1, 0, len(hist)-1)
        w[:, k] = 1.0 / ((1-ALPHA)*hist[bi]*(edges[1]-edges[0]) + ALPHA/n)
    weights = w.max(axis=1); weights /= weights.mean(); return weights.astype(np.float32)

def apply_eos_adv(X, y, g, target):
    m = g == target; X_min = X[m]; y_min = y[m]; X_oth = X[~m]
    if X_min.shape[0] < 10: logger.warning(f"  EOS: '{target}' too small -- skipping."); return X, y, g
    nn = NearestNeighbors(n_neighbors=EOS_NEIGHBORS).fit(X_oth); _, neigh = nn.kneighbors(X_min)
    synth = []
    for _ in range(int(len(X_min) * EOS_MULT)):
        i = rng.integers(len(X_min)); j = rng.choice(neigh[i]); R = rng.uniform(0.0, 1.0)
        synth.append(X_min[i] + R * (X_min[i] - X_oth[j]))
    if not synth: return X, y, g
    synth = np.vstack(synth)
    return (np.vstack([X, synth]), np.concatenate([y, rng.choice(y_min, len(synth))]), np.concatenate([g, np.repeat(target, len(synth))]))

def group_metrics(y_true, y_pred, groups):
    df = pd.DataFrame({"y": y_true, "p": y_pred, "g": groups})
    accs = df.groupby("g").apply(lambda d: accuracy_score(d["y"], d["p"]))
    worst = accs.idxmin(); return accs.to_dict(), worst, float(accs.min())

def supports_sw(model):
    try: return "sample_weight" in model.fit.__code__.co_varnames
    except: return False

def make_models():
    return {"LogReg": LogisticRegression(max_iter=5000, solver="saga", n_jobs=-1),
            "LinearSVM": LinearSVC(max_iter=5000),
            "RBF_Approx": make_pipeline(RBFSampler(gamma=1.0, n_components=500, random_state=RANDOM_STATE), LinearSVC(max_iter=5000)),
            "MLP": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300, random_state=RANDOM_STATE)}

STRATS = {"Baseline": dict(use_ar=False, eos=None), "AR": dict(use_ar=True, eos=None),
          "EOS_adv": dict(use_ar=False, eos="adv"), "AR+EOS_adv": dict(use_ar=True, eos="adv")}

rows = []; global_start = time.time()
for dname, path in DATASETS.items():
    if not path.exists(): logger.warning(f"Skipping {dname} -- not found"); continue
    logger.info(f"\nLoading: {dname}"); ad = sc.read_h5ad(path)
    age_col = detect_age_col(ad.obs, path.name)
    ad.obs["_age"] = ad.obs[age_col].astype(str).str.strip().str.lower()
    ad.obs["disease_binary"] = ad.obs[LABEL_KEY].apply(to_binary)
    ad = ad[~ad.obs["disease_binary"].isna()].copy()
    X0 = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    y0 = ad.obs["disease_binary"].to_numpy(); g0 = ad.obs["_age"].to_numpy()
    hashes = hash_rows(X0); _, uniq = np.unique(hashes, return_index=True)
    dup_frac = (len(hashes) - len(uniq)) / len(hashes)
    if dup_frac > 0.20: logger.warning(f"  Dedup would remove {dup_frac:.1%} -- skipping dedup."); dup_removed = 0
    else: X0, y0, g0 = X0[uniq], y0[uniq], g0[uniq]; dup_removed = len(hashes) - len(uniq)
    bad = np.isnan(X0).any(axis=1) | (np.abs(X0).sum(axis=1) == 0)
    if bad.any(): X0, y0, g0 = X0[~bad], y0[~bad], g0[~bad]
    logger.info(f"  Cells: {len(y0):,}  dups removed: {dup_removed}")
    for rep in range(N_REPEATS):
        counts = pd.Series(y0).value_counts(); stratify = y0 if (counts >= 2).all() else None
        Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X0, y0, g0, test_size=TEST_SIZE, stratify=stratify, random_state=RANDOM_STATE+rep)
        sc_ = StandardScaler(); Xtr = sc_.fit_transform(Xtr); Xte = sc_.transform(Xte)
        for strat, flags in STRATS.items():
            Xs, ys, gs = Xtr, ytr, gtr
            if flags["eos"] == "adv": Xs, ys, gs = apply_eos_adv(Xs, ys, gs, pd.Series(gs).value_counts().idxmin())
            weights = ar_weights(Xs) if flags["use_ar"] else None
            for mname, model in make_models().items():
                t0 = time.time()
                if weights is not None and supports_sw(model): model.fit(Xs, ys, sample_weight=weights); ar_applied = True
                else: model.fit(Xs, ys); ar_applied = False if weights is not None else None
                pred = model.predict(Xte); acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, average="macro")
                per, worst, wacc = group_metrics(yte, pred, gte)
                row = dict(dataset=dname, repeat=rep, model=mname, strategy=strat, accuracy=float(acc), macro_f1=float(f1),
                           worst_age_bin=worst, worst_age_bin_acc=float(wacc), dedup_removed=dup_removed,
                           n_unique=int(X0.shape[0]), n_train=int(Xtr.shape[0]), n_test=int(Xte.shape[0]),
                           ar_requested=bool(flags["use_ar"]), ar_applied=ar_applied, runtime_sec=time.time()-t0)
                for grp in KNOWN_GROUPS: row[f"per_age_acc_{grp}"] = float(per.get(grp, np.nan))
                rows.append(row)
                logger.info(f"  {dname}|rep={rep}|{strat}|{mname}|acc={acc:.4f}|worst={worst}:{wacc:.4f}")
                gc.collect()

pd.DataFrame(rows).to_csv(CSV_FILE, index=False)
logger.info(f"========== STEP 4b COMPLETE (AGE, scFoundation) ==========")
logger.info(f"CSV -> {CSV_FILE}  Runtime -> {(time.time()-global_start)/60:.2f} min")
df = pd.DataFrame(rows)
summary = df.groupby(["dataset","model","strategy"])[["accuracy","macro_f1","worst_age_bin_acc"]].mean().round(4).reset_index()
print("\n" + summary.to_string(index=False))
