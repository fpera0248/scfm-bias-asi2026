import scanpy as sc
import scipy.sparse as sp
import numpy as np

for fname in ['AIDA_RawCounts_ETHNICITY.h5ad', 'AIDA_Ethnicity_External_Validation_12500.h5ad']:
    print(f'\nProcessing {fname}', flush=True)
    a = sc.read_h5ad(fname)
    print(f'  Original X type: {type(a.X).__name__}', flush=True)

    a.X = sp.csr_matrix(a.X)
    print(f'  Converted to: {type(a.X).__name__}, dtype={a.X.dtype}, format={a.X.format}', flush=True)

    out = fname.replace('.h5ad', '_v2.h5ad')
    a.write_h5ad(out, compression='gzip')
    print(f'  Wrote {out}', flush=True)
