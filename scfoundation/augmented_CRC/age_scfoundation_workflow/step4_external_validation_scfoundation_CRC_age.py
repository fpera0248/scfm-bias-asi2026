#!/usr/bin/env python3
"""STEP 4 — External Validation Disease + Cell-Type Classification (CRC AGE, scFoundation)"""
import copy, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

BASE        = Path("/oscar/home/fperalta/data/fperalta/scfoundation/augmented_CRC/age_scfoundation_workflow")
LABELED_DIR = BASE / "step3b_labeled"
OUTDIR      = BASE / "step4_external_validation_scfoundation_age"; OUTDIR.mkdir(exist_ok=True)

OUTPUT_BASE = "CRC_Age_Pilot"
PILOTS = {
    "Proportional_2498":          LABELED_DIR / f"{OUTPUT_BASE}_Proportional_2498_labeled_scfoundation.h5ad",
    "Downsampled_124Each":         LABELED_DIR / f"{OUTPUT_BASE}_Downsampled_124Each_labeled_scfoundation.h5ad",
    "BalancedUpsampled_650Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedUpsampled_650Each_labeled_scfoundation.h5ad",
    "BalancedAugmented_650Each": LABELED_DIR / f"{OUTPUT_BASE}_BalancedAugmented_650Each_labeled_scfoundation.h5ad",
}
VALIDATION_FILE = BASE / "step2a_embeddings" / "CRC_Age_External_Validation_9402_scfoundation.h5ad"
if not VALIDATION_FILE.exists():
    VALIDATION_FILE = BASE / "CRC_Age_External_Validation_9402_scfoundation.h5ad"
CSV_FILE = OUTDIR / "step4_external_validation_results_age_scfoundation.csv"
TXT_FILE = OUTDIR / "step4_external_validation_log_age_scfoundation.txt"

EMB_KEY      = "X_scfoundation"
DISEASE_COL  = "disease"; CELLTYPE_COL = "cell_type"
AGE_COL_CANDIDATES = ["age_bin_10yr", "age_bin", "age_group", "development_stage"]
RANDOM_STATE = 42; EOS_NEIGHBORS = 5; EOS_MULTIPLIER = 0.50; AR_BINS = 30

MODEL_TEMPLATES = {
    "LogReg":       LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", n_jobs=None),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"),
}
STRATEGIES = {
    "Baseline": dict(use_ar=False, use_eos=False), "AR": dict(use_ar=True, use_eos=False),
    "EOS": dict(use_ar=False, use_eos=True), "AR+EOS": dict(use_ar=True, use_eos=True),
}

log_fh = open(TXT_FILE, "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); print(line, file=log_fh, flush=True)

def detect_age_col(obs):
    for c in AGE_COL_CANDIDATES:
        if c in obs.columns: return c
    raise RuntimeError(f"No age column. Available: {list(obs.columns)}")

def make_binary(x):
    if pd.isna(x): return np.nan
    return "normal" if str(x).lower().strip() == "normal" else "disease"

def load_ds(name, path):
    if not path.exists(): raise FileNotFoundError(str(path))
    ad = sc.read_h5ad(path)
    if EMB_KEY not in ad.obsm: raise RuntimeError(f"Missing {EMB_KEY}")
    X = np.array(ad.obsm[EMB_KEY], dtype=np.float32)
    age_col = detect_age_col(ad.obs)
    age = ad.obs[age_col].astype(str).str.strip().str.lower().to_numpy()
    dis = ad.obs[DISEASE_COL].apply(make_binary).to_numpy()
    ct  = ad.obs[CELLTYPE_COL].astype(str).to_numpy()
    log(f"  {name}: {ad.n_obs:,} cells, {pd.Series(age).nunique()} age bins")
    return X, age, dis, ct

def calculate_ar_weights(X_train, bins=AR_BINS):
    ALPHA = 0.0001; n, d = X_train.shape; dim_weights = np.empty((n, d), dtype=np.float64)
    for k in range(d):
        vals = X_train[:, k]; hist, edges = np.histogram(vals, bins=bins, density=True)
        bin_idx = np.clip(np.digitize(vals, edges) - 1, 0, bins - 1)
        prob_s  = (1.0 - ALPHA) * hist[bin_idx] * (edges[1]-edges[0]) + ALPHA / n
        dim_weights[:, k] = 1.0 / prob_s
    weights = dim_weights.max(axis=1); weights /= weights.mean(); return weights.astype(np.float32)

def apply_eos(X_train, y_train, g_train, target_group):
    mask_min = (g_train == target_group); X_min, y_min = X_train[mask_min], y_train[mask_min]
    if X_min.shape[0] < 10: log(f"    EOS: minority '{target_group}' too small -- skipping."); return X_train, y_train, g_train
    nn = NearestNeighbors(n_neighbors=EOS_NEIGHBORS).fit(X_train); _, neighbors = nn.kneighbors(X_min)
    rng = np.random.default_rng(RANDOM_STATE); synth = []
    for _ in range(int(len(X_min) * EOS_MULTIPLIER)):
        i0 = rng.integers(0, len(X_min)); enemy = [j for j in neighbors[i0] if g_train[j] != target_group]
        if not enemy: continue
        j = rng.choice(enemy); synth.append(X_min[i0] + rng.uniform(0.0, 1.0) * (X_min[i0] - X_train[j]))
    if not synth: return X_train, y_train, g_train
    synth = np.vstack(synth)
    return (np.vstack([X_train, synth]), np.concatenate([y_train, rng.choice(y_min, size=len(synth))]),
            np.concatenate([g_train, np.repeat(target_group, len(synth))]))

def evaluate_external(model_template, X_train, y_train, g_train, X_test, y_test, g_test, use_ar=False, use_eos=False):
    mask = pd.Series(y_train).notna().to_numpy() & ~np.isnan(X_train).any(axis=1)
    X_tr, y_tr, g_tr = X_train[mask], y_train[mask], g_train[mask]
    if len(np.unique(y_tr)) < 2: return None
    scaler = StandardScaler(); X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_test)
    if use_eos: X_tr, y_tr, g_tr = apply_eos(X_tr, y_tr, g_tr, pd.Series(g_tr).value_counts().idxmin())
    weights = calculate_ar_weights(X_tr) if use_ar else None
    model = copy.deepcopy(model_template)
    if weights is not None:
        try: model.fit(X_tr, y_tr, sample_weight=weights)
        except TypeError: model.fit(X_tr, y_tr)
    else: model.fit(X_tr, y_tr)
    preds = model.predict(X_te); acc = accuracy_score(y_test, preds); f1 = f1_score(y_test, preds, average="macro")
    per_group = {grp: accuracy_score(y_test[g_test==grp], preds[g_test==grp]) for grp in np.unique(g_test) if (g_test==grp).sum() > 0}
    worst_grp = min(per_group, key=per_group.get) if per_group else "n/a"
    return dict(accuracy=acc, macro_f1=f1, per_group=per_group, worst_group=worst_grp,
                worst_acc=per_group.get(worst_grp, np.nan), n_train=len(y_tr), n_test=len(y_test))

def main():
    warnings.filterwarnings("ignore")
    log("=" * 70); log("STEP 4 -- External Validation Classification (CRC AGE, scFoundation)"); log("=" * 70)
    log("\nLoading validation set ...")
    X_val, age_val, dis_val, ct_val = load_ds("ExternalValidation", VALIDATION_FILE)
    valid_dis = ~pd.isna(pd.Series(dis_val))
    X_val_dis = X_val[valid_dis.values]; age_val_dis = age_val[valid_dis.values]; dis_val_dis = dis_val[valid_dis.values]
    log(f"  Disease task validation: {X_val_dis.shape[0]:,} cells")
    log(f"  Cell-type task validation: {X_val.shape[0]:,} cells, {len(np.unique(ct_val))} types")
    rows = []
    for pname, ppath in PILOTS.items():
        log(f"\n{'='*70}"); log(f"Pilot: {pname}")
        try: X_pilot, age_pilot, dis_pilot, ct_pilot = load_ds(pname, ppath)
        except (FileNotFoundError, RuntimeError) as err: log(f"  SKIP: {err}"); continue
        valid_pilot = ~pd.isna(pd.Series(dis_pilot))
        X_pilot_dis = X_pilot[valid_pilot.values]; age_pilot_dis = age_pilot[valid_pilot.values]; dis_pilot_dis = dis_pilot[valid_pilot.values]
        for model_name, model_tmpl in MODEL_TEMPLATES.items():
            for strat_name, flags in STRATEGIES.items():
                log(f"  {model_name:12s} | {strat_name:8s}")
                dres = evaluate_external(model_tmpl, X_pilot_dis, dis_pilot_dis, age_pilot_dis, X_val_dis, dis_val_dis, age_val_dis, **flags)
                cres = evaluate_external(model_tmpl, X_pilot, ct_pilot, age_pilot, X_val, ct_val, age_val, **flags)
                row = {"pilot": pname, "model": model_name, "strategy": strat_name}
                if dres is not None:
                    row.update({"disease_accuracy": dres["accuracy"], "disease_macro_f1": dres["macro_f1"],
                                "disease_worst_age_bin": dres["worst_group"], "disease_worst_acc": dres["worst_acc"],
                                "disease_n_train": dres["n_train"], "disease_n_test": dres["n_test"]})
                    for grp, a in dres["per_group"].items(): row[f"disease_acc_{grp}"] = a
                    log(f"    disease: acc={dres['accuracy']:.3f} worst={dres['worst_group']}:{dres['worst_acc']:.3f}")
                if cres is not None:
                    row.update({"celltype_accuracy": cres["accuracy"], "celltype_macro_f1": cres["macro_f1"],
                                "celltype_worst_age_bin": cres["worst_group"], "celltype_worst_acc": cres["worst_acc"]})
                    for grp, a in cres["per_group"].items(): row[f"celltype_acc_{grp}"] = a
                    log(f"    celltype: acc={cres['accuracy']:.3f}")
                rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(CSV_FILE, index=False)
    log(f"STEP 4 COMPLETE (CRC AGE, scFoundation)  Results -> {CSV_FILE.name}")
    available = [c for c in ["pilot","model","strategy","disease_accuracy","disease_macro_f1","celltype_accuracy","celltype_macro_f1"] if c in df.columns]
    print("\n" + df[available].to_string(index=False)); log_fh.close()

if __name__ == "__main__":
    main()
