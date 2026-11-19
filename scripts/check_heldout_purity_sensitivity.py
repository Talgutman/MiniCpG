from pathlib import Path
import json
import numpy as np,pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
import argparse
parser=argparse.ArgumentParser(description="Leave-one-class-out purity sensitivity; outputs remain local")
parser.add_argument('--analysis-dir',required=True)
args=parser.parse_args()
O=Path(args.analysis_dir)
d=pd.read_csv(O/'sample_matching_and_predictions.tsv',sep='\t').dropna(subset=['purity_exact'])
assert (d.error==(d.tcga_label!=d.y_pred).astype(int)).all()
vary=d.groupby('tcga_label').error.nunique();d=d[d.tcga_label.isin(vary[vary==2].index)].copy()
rows=[]
for cl in sorted(d.tcga_label.unique()):
 z=d[d.tcga_label!=cl];m=ConditionalLogit(z.error.to_numpy(float),(-z.purity_exact.to_numpy()/0.1).reshape(-1,1),groups=z.tcga_label.to_numpy());f=m.fit(disp=False,maxiter=300,gtol=1e-8)
 rows.append({'omitted_class':cl,'OR_per_10pp_lower':np.exp(f.params[0]),'p':f.pvalues[0]})
pd.DataFrame(rows).to_csv(O/'leave_one_class_out.tsv',sep='\t',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print('\nExcluded specimen/vial matches:')
allrows=pd.read_csv(O/'sample_matching_and_predictions.tsv',sep='\t');print(allrows.loc[allrows.match_status=='different_specimen_or_vial',['sample_id','sample_submitter_id','source_barcodes','error']].to_string(index=False))
