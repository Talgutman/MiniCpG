"""Post hoc association of specimen-matched ABSOLUTE purity with locked TCGA errors."""
from pathlib import Path
import json, subprocess, sys, hashlib, warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, chi2
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.discrete.discrete_model import Logit
import statsmodels
import argparse
from minicpg.analysis_io import AnalysisInputs
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',required=True)
parser.add_argument('--out-dir',required=True)
args=parser.parse_args();bound=AnalysisInputs(args.config)
O=bound.output_dir(args.out_dir)
paths={k:bound.files[k] for k in ['manifest','predictions','metadata','purity']}
sha={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
m,p,meta,a=[pd.read_csv(paths[k],sep='\t',dtype=str).fillna('') for k in ['manifest','predictions','metadata','purity']]
assert p.sample_id.is_unique and m.sample_id.is_unique and meta.file_uuid.is_unique
assert set(p.sample_id)==set(m.sample_id)
d=p.merge(m[['sample_id','file_uuid']],on='sample_id',validate='one_to_one').merge(meta[['file_uuid','sample_submitter_id','sample_type']],on='file_uuid',how='left',validate='one_to_one')
assert d.sample_submitter_id.notna().all()
assert (d.sample_id==d.sample_submitter_id.str[:12]).all()
a['purity']=pd.to_numeric(a.purity,errors='coerce');a=a[a.purity.between(0,1)].copy();a['patient']=a['sample'].str[:12]
rows=[]
for r in d.to_dict('records'):
 q=a[a.patient==r['sample_id']]; exact=q[q['sample'].str[:16]==r['sample_submitter_id'][:16]] # TCGA specimen + vial identifier has 16 characters
 sample=q[q['sample'].str[:15]==r['sample_submitter_id'][:15]]
 def choose(z):
  if len(z)==1:return float(z.iloc[0].purity)
  return np.nan
 r.update(error=int(r['tcga_label']!=r['y_pred']),n_patient_records=len(q),n_exact_records=len(exact),n_sample_records=len(sample),purity_exact=choose(exact),purity_sample=choose(sample),purity_patient_unique=choose(q),source_barcodes=';'.join(q['sample']),source_values=';'.join(q.purity.astype(str)))
 r['match_status']='exact_specimen_vial' if len(exact)==1 else ('no_patient_purity' if not len(q) else ('ambiguous_exact' if len(exact)>1 else 'different_specimen_or_vial'))
 rows.append(r)
d=pd.DataFrame(rows);d.to_csv(O/'sample_matching_and_predictions.tsv',sep='\t',index=False)
# Primary is exact specimen/vial only; sample-type matches ignoring vial are a sensitivity, not silently pooled.
res={'matching':d.match_status.value_counts().to_dict(),'heldout_n':len(d),'input_sha256':sha,'statsmodels_version':statsmodels.__version__,'methods_url':'https://www.statsmodels.org/stable/generated/statsmodels.discrete.conditional_models.ConditionalLogit.html'}
def fit(frame,col):
 z=frame.dropna(subset=[col]).copy(); z['lower_purity_10pp']=-z[col]/0.1
 varying=z.groupby('tcga_label').error.nunique();eligible=varying[varying==2].index;v=z[z.tcga_label.isin(eligible)].copy()
 mdl=ConditionalLogit(v.error.to_numpy(float),v[['lower_purity_10pp']].to_numpy(float),groups=v.tcga_label.to_numpy())
 f=mdl.fit(method='bfgs',disp=False,maxiter=300,gtol=1e-8)
 b=float(f.params[0]);se=float(f.bse[0]);score=float(mdl.score(f.params)[0]);assert abs(score)<1e-3
 un=Logit(z.error.to_numpy(float),np.column_stack([np.ones(len(z)),z.lower_purity_10pp])).fit(disp=False)
 summary={'n':len(z),'errors':int(z.error.sum()),'classes':z.tcga_label.nunique(),'informative_n':len(v),'informative_classes':len(eligible),'classes_without_outcome_variation':list(varying[varying<2].index),'OR_error_per_10pp_lower_purity':float(np.exp(b)),'CI95':[float(np.exp(b-1.96*se)),float(np.exp(b+1.96*se))],'wald_p':float(f.pvalues[0]),'LR_p':float(chi2.sf(2*(f.llf-mdl.loglike(np.zeros(1))),1)),'score_at_optimum':score,'pooled_OR':float(np.exp(un.params[1])),'pooled_p':float(un.pvalues[1]),'correct_purity_median':float(z.loc[z.error==0,col].median()),'error_purity_median':float(z.loc[z.error==1,col].median()),'purity_range':[float(z[col].min()),float(z[col].max())]}
 for val,name in [(0,'correct'),(1,'error')]:summary[name+'_purity_IQR']=z.loc[z.error==val,col].quantile([.25,.75]).tolist()
 return summary,z,v
for col in ['purity_exact','purity_sample','purity_patient_unique']:
 res[col],z,v=fit(d,col)
 if col=='purity_exact':primary=z;informative=v
# Within-class permutation score test: no linear-logit effect-size assumption, fixed observed number of errors per class.
rng=np.random.default_rng(20261102);B=10000;observed=0.;null=np.zeros(B)
for cl,g in informative.groupby('tcga_label'):
 x=g.purity_exact.to_numpy();x=x-x.mean();y=g.error.to_numpy();observed+=float(x@y)
 for b in range(B):null[b]+=float(x@rng.permutation(y))
res['stratified_permutation']={'statistic':observed,'two_sided_p':float((1+np.sum(abs(null)>=abs(observed)))/(B+1)),'permutations':B,'seed':20261102}
rows=[]
for cl,g in d.groupby('tcga_label'):
 z=g.dropna(subset=['purity_exact']);c=z[z.error==0].purity_exact;e=z[z.error==1].purity_exact
 rows.append({'class':cl,'heldout_n':len(g),'matched_n':len(z),'correct_n':len(c),'error_n':len(e),'correct_median':c.median(),'error_median':e.median(),'median_difference_error_minus_correct':e.median()-c.median()})
classes=pd.DataFrame(rows);classes.to_csv(O/'class_summary.tsv',sep='\t',index=False)
# Fixed descriptive bins; no threshold selection, pooled class mixture acknowledged.
primary['purity_bin']=pd.cut(primary.purity_exact,[-.001,.4,.6,.8,1.0],right=True)
bins=[]
for label,g in primary.groupby('purity_bin',observed=False):
 n=len(g);k=int(g.error.sum());ph=k/n if n else np.nan;zz=norm.ppf(.975);den=1+zz**2/n if n else np.nan
 center=(ph+zz**2/(2*n))/den if n else np.nan;half=zz*np.sqrt(ph*(1-ph)/n+zz**2/(4*n*n))/den if n else np.nan
 bins.append({'purity_bin':str(label),'n':n,'errors':k,'error_rate':ph,'wilson_low':center-half,'wilson_high':center+half})
pd.DataFrame(bins).to_csv(O/'descriptive_purity_bins.tsv',sep='\t',index=False)
(O/'results.json').write_text(json.dumps(res,indent=2,allow_nan=False))
assert sha=={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
print(json.dumps(res,indent=2))

bound.finish(O,{"workflow":"heldout_purity","matching":"exact specimen and vial primary; relaxed matches separate sensitivities"})
