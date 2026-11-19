"""Fit sample-level PCA on frozen development-set locked-panel betas only."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args()
O=Path(args.run_dir);I=O/'inputs';A=O/'analysis';A.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mh(ids):return hashlib.sha256('\n'.join(sorted(set(map(str,ids)))).encode()).hexdigest()
hashes=json.loads((I/'input_hashes.json').read_text())
for name,digest in hashes.items():assert sha(I/name)==digest,(name,'hash mismatch')
panel=pd.read_csv(I/'panel.tsv',sep='\t').sort_values('panel_rank_rule');probes=panel.probe_id.tolist();lock=json.loads((I/'split_lock.json').read_text())
matrices={};missing={};class_counts=[]
for split,manifest_name,mat_name,hashkey,nkey in [('dev','dev_manifest.tsv','dev_matrix.tsv','dev_sample_sha256','dev_samples'),('heldout','heldout_manifest.tsv','heldout_matrix.tsv','heldout_sample_sha256','heldout_samples')]:
 mf=pd.read_csv(I/manifest_name,sep='\t');m=pd.read_csv(I/mat_name,sep='\t')
 assert mf.sample_id.is_unique and m.sample_id.is_unique
 assert set(m.sample_id)==set(mf.sample_id) and len(m)==lock[nkey] and mh(m.sample_id)==lock[hashkey]
 assert m.columns[2:].tolist()==probes
 joined=m[['sample_id','project_id']].merge(mf[['sample_id','project_id']],on='sample_id',validate='one_to_one',suffixes=('_matrix','_manifest'));assert joined.project_id_matrix.eq(joined.project_id_manifest).all()
 beta=m[probes].apply(pd.to_numeric,errors='coerce');invalid=(beta.lt(0)|beta.gt(1))&beta.notna();beta=beta.mask(invalid)
 matrices[split]=(m[['sample_id','project_id']],beta)
 missing[split]={'n_samples':len(m),'n_classes':m.project_id.nunique(),'missing_before_imputation':int(beta.isna().sum().sum()),'invalid_values_masked':int(invalid.sum().sum())}
 class_counts.extend({'split':split,'tumor_class':c,'n_samples':int(n)} for c,n in m.project_id.value_counts().items())
assert not set(matrices['dev'][0].sample_id)&set(matrices['heldout'][0].sample_id)
medians=matrices['dev'][1].median();assert medians.notna().all()
X={s:b.fillna(medians).to_numpy() for s,(meta,b) in matrices.items()}
for x in X.values():assert np.isfinite(x).all()
model=PCA(svd_solver='full');scores={'dev':model.fit_transform(X['dev'])};scores['heldout']=model.transform(X['heldout'])
for split,x in scores.items():
 d=matrices[split][0].copy();d['split']=split
 for i in range(x.shape[1]):d[f'PC{i+1}']=x[:,i]
 d.to_csv(A/f'{split}_coordinates.tsv',sep='\t',index=False)
 np.testing.assert_allclose(model.inverse_transform(x),X[split],atol=1e-12)
pd.DataFrame(class_counts).to_csv(A/'class_sample_counts.tsv',sep='\t',index=False)
pd.DataFrame({'probe_id':probes,'DEV_median_beta':medians.loc[probes].to_numpy(),'PCA_DEV_mean_beta':model.mean_}).to_csv(A/'preprocessing.tsv',sep='\t',index=False)
pd.DataFrame(model.components_.T,index=probes,columns=[f'PC{i+1}' for i in range(len(probes))]).to_csv(A/'probe_loadings.tsv',sep='\t',index_label='probe_id')
pd.DataFrame({'component':[f'PC{i+1}' for i in range(len(probes))],'explained_variance_ratio':model.explained_variance_ratio_,'explained_variance':model.explained_variance_}).to_csv(A/'explained_variance.tsv',sep='\t',index=False)
cent=[]
for split,scores_ in scores.items():
 coords=pd.DataFrame(scores_[:,:2],columns=['PC1','PC2']);coords['tumor_class']=matrices[split][0].project_id.to_numpy()
 for cls,g in coords.groupby('tumor_class'):cent.append({'split':split,'tumor_class':cls,'PC1_centroid':float(g.PC1.mean()),'PC2_centroid':float(g.PC2.mean()),'n_samples':len(g)})
pd.DataFrame(cent).to_csv(A/'projected_class_centroids.tsv',sep='\t',index=False)
s={'n_probes':len(probes),'splits':missing,'PC1_variance_ratio':float(model.explained_variance_ratio_[0]),'PC2_variance_ratio':float(model.explained_variance_ratio_[1]),'first_two_variance_ratio':float(model.explained_variance_ratio_[:2].sum()),'first_three_variance_ratio':float(model.explained_variance_ratio_[:3].sum()),'fit_population':'TCGA development set only','preprocessing':'Mask nonnumeric/out-of-range beta; impute each probe using DEV median; PCA centers probes using DEV means; no variance scaling. Held-out samples receive the same fitted imputation and PCA transform.','split_membership_and_labels_verified':True,'locked_panel_order_verified':True,'dev_heldout_overlap':0,'input_hashes_verified':True,'reconstruction_verified':True,'limits':['PCA fit does not use labels, but selected CpGs were chosen through supervised DEV feature selection.','Two-dimensional projection represents only part of the original 30-dimensional variation.','Projection is descriptive, not a classification accuracy or independent feature-validation test.','No UMAP or embedding parameter search was performed.']}
(A/'summary.json').write_text(json.dumps(s,indent=2)+'\n');(A/'SUCCESS.json').write_text(json.dumps({'input_gates_passed':True,'reconstruction_passed':True},indent=2)+'\n');print(json.dumps(s,indent=2))
