#!/usr/bin/env python3
"""
Check pipeline completion status across all CRC workflows.
Run on Oscar: python3 check_CRC_pipeline_status.py
"""
import pathlib
import pandas as pd

BASE = pathlib.Path("/oscar/home/fperalta/data/fperalta")

WORKFLOWS = {
    # Geneformer
    "GF_AGE":   BASE / "Geneformer/augmented_CRC/age_Geneformer_workflow",
    "GF_ETH":   BASE / "Geneformer/augmented_CRC/ethnicity_Geneformer_workflow",
    # scGPT
    "SCGPT_AGE": BASE / "scGPT/augmented_CRC/age_scGPT_workflow",
    "SCGPT_ETH": BASE / "scGPT/augmented_CRC/ethnicity_scGPT_workflow",
    # scFoundation
    "SCF_AGE":  BASE / "scfoundation/augmented_CRC/age_scfoundation_workflow",
    "SCF_ETH":  BASE / "scfoundation/augmented_CRC/ethnicity_scfoundation_workflow",
}

# Expected output files per step per workflow
def expected_outputs(wf_dir, model, axis):
    emb    = "X_geneformer" if model == "GF" else ("X_scGPT" if model == "SCGPT" else "X_scfoundation")
    suffix = "geneformer"   if model == "GF" else ("scgpt"   if model == "SCGPT" else "scfoundation")
    axis_l = axis.lower()

    if axis == "AGE":
        datasets   = ["BalancedAugmented_520Each", "Proportional_1999",
                      "BalancedUpsampled_520Each", "Downsampled_99Each"]
        prefix     = "CRC_Age_Pilot"
        bench_dir  = f"benchmark_outputs_{suffix}_age"
        val_file   = f"CRC_Age_External_Validation_9402_{suffix}.h5ad"
        ref        = "Proportional_1999"
        step4_csv  = f"step4_external_validation_results_age_{suffix}.csv" if model != "GF" else "step4_external_validation_results_age.csv"
        step4a_csv = f"step4a_downstream_results_age_AR_EOS_{suffix}.csv"
        step4b_csv = f"step4b_results_age_labeled_{suffix}.csv"
        step5_csv  = f"step5_summary_age_{suffix}.csv"
        step6_csv  = f"step6_per_age_diagnostics_{suffix}.csv"
        step7_csv  = f"step7_per_age_diagnostics_{suffix}.csv"
        step8_csv  = f"step8_per_age_disease_prediction_{suffix}.csv"
        step8_worst= f"step8_worst_age_bin_summary_{suffix}.csv"
        step4_dir  = f"step4_external_validation_{suffix}_age"
        step4a_dir = f"step4a_downstream_{suffix}_age"
        step4b_dir = f"step4b_model_robustness_tests_age_{suffix}"
        step5_dir  = f"step5_outputs_age_{suffix}"
        step6_dir  = f"step6_outputs_age_{suffix}"
        step7_dir  = f"step7_representation_diagnostics_age_{suffix}"
        step8_dir  = f"step8_age_conditioned_disease_{suffix}"
        step9_dir  = f"step9_visualizations_age_{suffix}"
    else:  # ETH
        datasets   = ["BalancedAugmented_1504Each", "Proportional_1998",
                      "BalancedUpsampled_1504Each", "Downsampled_90Each"]
        prefix     = "CRC_Eth_Pilot"
        bench_dir  = f"benchmark_outputs_{suffix}_ethnicity"
        val_file   = f"CRC_Eth_External_Validation_8572_{suffix}.h5ad"
        ref        = "Proportional_1998"
        step4_csv  = f"step4_external_validation_results_{suffix}.csv" if model != "GF" else "step4_external_validation_results_ethnicity.csv"
        step4a_csv = f"step4a_downstream_results_ethnicity_AR_EOS_{suffix}.csv"
        step4b_csv = f"step4b_results_ethnicity_labeled_{suffix}.csv"
        step5_csv  = f"step5_summary_{suffix}_ethnicity.csv"
        step6_csv  = f"step6_per_ethnicity_diagnostics_{suffix}.csv"
        step7_csv  = f"step7_per_ethnicity_diagnostics_{suffix}.csv"
        step8_csv  = f"step8_per_ethnicity_disease_prediction_{suffix}.csv"
        step8_worst= f"step8_worst_ethnicity_summary_{suffix}.csv"
        step4_dir  = f"step4_external_validation_{suffix}"
        step4a_dir = f"step4a_downstream_{suffix}"
        step4b_dir = f"step4b_model_robustness_tests_{suffix}"
        step5_dir  = f"step5_outputs_{suffix}_ethnicity"
        step6_dir  = f"step6_outputs_{suffix}_ethnicity"
        step7_dir  = f"step7_representation_diagnostics_{suffix}_ethnicity"
        step8_dir  = f"step8_eth_conditioned_disease_{suffix}"
        step9_dir  = f"step9_visualizations_{suffix}_ethnicity"

    checks = {}

    # step2a: embedded pilot h5ads
    emb_subdir = wf_dir / "step2a_embeddings" if (wf_dir / "step2a_embeddings").exists() else wf_dir
    for ds in datasets:
        tag = "AGE" if axis == "AGE" else "ETH"
        fname = f"{prefix}_{ds}_{tag}_{suffix}.h5ad"
        p = emb_subdir / fname
        if not p.exists(): p = wf_dir / fname
        checks[f"step2a_{ds}"] = p

    # step3a: benchmark summary
    checks["step3a_summary"] = wf_dir / bench_dir / "benchmark_summary_all_modes.csv"

    # step3b: labeled files
    for ds in datasets:
        checks[f"step3b_{ds}"] = wf_dir / "step3b_labeled" / f"{prefix}_{ds}_labeled_{suffix}.h5ad"

    # step4
    checks["step4_csv"] = wf_dir / step4_dir / step4_csv

    # step4a
    checks["step4a_csv"] = wf_dir / step4a_dir / step4a_csv

    # step4b
    checks["step4b_csv"] = wf_dir / step4b_dir / step4b_csv

    # step5
    checks["step5_csv"] = wf_dir / step5_dir / step5_csv

    # step6
    checks["step6_csv"] = wf_dir / step6_dir / step6_csv

    # step7
    checks["step7_csv"] = wf_dir / step7_dir / step7_csv

    # step8
    checks["step8_csv"]      = wf_dir / step8_dir / step8_csv
    checks["step8_worst_csv"] = wf_dir / step8_dir / step8_worst

    # step9: at least one PNG
    step9_path = wf_dir / step9_dir
    checks["step9_dir_exists"] = step9_path
    checks["step9_has_pngs"]   = step9_path  # checked separately

    return checks


rows = []
for wf_name, wf_dir in WORKFLOWS.items():
    model, axis = wf_name.split("_", 1)
    checks = expected_outputs(wf_dir, model, axis)

    for step_name, path in checks.items():
        if step_name == "step9_has_pngs":
            exists = path.is_dir() and len(list(path.glob("*.png"))) > 0
            n_pngs = len(list(path.glob("*.png"))) if path.is_dir() else 0
            note   = f"{n_pngs} PNGs" if path.is_dir() else "dir missing"
        else:
            exists = path.exists()
            note   = ""
            if exists and path.suffix == ".csv":
                try:
                    df = pd.read_csv(path)
                    note = f"{len(df)} rows"
                except Exception:
                    note = "unreadable"

        rows.append({
            "workflow":  wf_name,
            "step":      step_name,
            "status":    "OK" if exists else "MISSING",
            "note":      note,
            "path":      str(path.relative_to(BASE)),
        })

df = pd.DataFrame(rows)

# Summary by workflow
print("\n" + "="*70)
print("CRC PIPELINE STATUS SUMMARY")
print("="*70)

for wf in WORKFLOWS:
    wf_df   = df[df["workflow"] == wf]
    ok      = (wf_df["status"] == "OK").sum()
    total   = len(wf_df)
    missing = wf_df[wf_df["status"] == "MISSING"]["step"].tolist()
    status  = "COMPLETE" if ok == total else f"{ok}/{total}"
    print(f"\n{wf:12s}  {status}")
    for m in missing:
        print(f"             MISSING: {m}")

# Full table for missing items
missing_df = df[df["status"] == "MISSING"][["workflow", "step", "path"]]
if not missing_df.empty:
    print("\n" + "="*70)
    print("ALL MISSING FILES")
    print("="*70)
    print(missing_df.to_string(index=False))
else:
    print("\nAll checks passed.")

# Save full report
out = pathlib.Path("/oscar/home/fperalta/data/fperalta/CRC_pipeline_status.csv")
df.to_csv(out, index=False)
print(f"\nFull report saved -> {out}")
