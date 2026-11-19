"""Descriptive probe-level PCA of immutable locked-panel development inputs."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist,squareform
from scipy.stats import pearsonr

p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args()
O=Path(args.run_dir);I=O/'inputs';R=O/'analysis';R.mkdir(exist_ok=True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
frozen=json.loads((I/'input_hashes.json').read_text())
for name,digest in frozen.items():assert sha(I/name)==digest,(name,'hash mismatch')
panel=pd.read_csv(I/'panel.tsv',sep='\t').sort_values('panel_rank_rule');probes=panel.probe_id.tolist()
dev=pd.read_csv(I/'dev_manifest.tsv',sep='\t');held=pd.read_csv(I/'heldout_manifest.tsv',sep='\t');matrix=pd.read_csv(I/'dev_matrix.tsv',sep='\t')
lock=json.loads((I/'split_lock.json').read_text())
def member_hash(x):return hashlib.sha256('\n'.join(sorted(set(map(str,x)))).encode()).hexdigest()
assert matrix.sample_id.is_unique and dev.sample_id.is_unique and panel.probe_id.is_unique
assert set(matrix.sample_id)==set(dev.sample_id)
assert not set(matrix.sample_id)&set(held.sample_id)
assert member_hash(dev.sample_id)==lock['dev_sample_sha256']
assert len(dev)==lock['dev_samples']
assert matrix.columns[2:].tolist()==probes
joined=matrix.merge(dev[['sample_id','project_id']],on='sample_id',validate='one_to_one',suffixes=('_matrix','_manifest'))
assert joined.project_id_matrix.eq(joined.project_id_manifest).all()
beta=matrix.set_index('sample_id')[probes].apply(pd.to_numeric,errors='coerce')
invalid=(beta.lt(0)|beta.gt(1))&beta.notna();masked=beta.mask(invalid)
missing=int(masked.isna().sum().sum());medians=masked.median();assert medians.notna().all()
imputed=masked.fillna(medians)
assert np.isfinite(imputed.to_numpy()).all()
classes=matrix.set_index('sample_id').project_id
means=imputed.groupby(classes).mean().sort_index();assert len(means)==matrix.project_id.nunique()
means.index.name='tumor_class';means.to_csv(R/'class_mean_beta.tsv',sep='\t')
matrix.groupby('project_id').size().rename('n_samples').to_csv(R/'class_sample_counts.tsv',sep='\t')
presence=pd.read_csv(I/'fold_presence.tsv',sep='\t');assert presence.fold_id.is_unique
counts=presence[probes].sum();assert set(np.unique(presence[probes].to_numpy()))<={0,1}
np.testing.assert_array_equal(counts.to_numpy(),panel.set_index('probe_id').loc[probes,'n_folds_top_panel'].to_numpy())
np.testing.assert_allclose(counts.to_numpy()/len(presence),panel.selection_rate_top_panel.to_numpy(),rtol=0,atol=1e-12)
ann=pd.read_csv(I/'annotations.tsv',sep='\t').set_index('probe_id').loc[probes].reset_index()
assert ann.panel_rank_rule.tolist()==panel.panel_rank_rule.tolist()
np.testing.assert_allclose(ann.selection_rate_top_panel,panel.selection_rate_top_panel,rtol=0,atol=1e-12)
assert ann.context_group.notna().all()
X=means[probes].T
ann['mean_beta_across_classes']=X.mean(axis=1).to_numpy()
ann['sd_beta_across_classes']=X.std(axis=1,ddof=0).to_numpy()
ann.to_csv(R/'probe_annotations_plotted.tsv',sep='\t',index=False)
summary={'n_samples':len(matrix),'n_classes':len(means),'n_probes':len(probes),'n_selection_folds':len(presence),'context_counts':ann.context_group.value_counts().to_dict(),'missing_before_imputation':missing,'invalid_beta_values_masked':int(invalid.sum().sum()),'imputation':'Per-probe median across the TCGA development set only; applied before class means. No held-out or external samples.','class_weighting':'Each class contributes one mean-beta coordinate, independent of its sample count.','panel_order_verified':True,'dev_membership_and_labels_verified':True,'selection_frequency_recomputed_from_presence':True,'heldout_overlap':0,'preprocessing_primary':'PCA on CpG rows and class-mean-beta columns; columns centered across CpGs by sklearn PCA, no variance scaling.','preprocessing_sensitivity':'Each CpG class-mean profile has its own across-class mean subtracted before the same column-centered PCA; no variance scaling.','views':{}}
for name,values in [('class_mean_beta',X),('within_probe_centered',X.sub(X.mean(axis=1),axis=0))]:
 values.to_csv(R/f'pca_input_{name}.tsv',sep='\t',index_label='probe_id')
 model=PCA(svd_solver='full');scores=model.fit_transform(values.to_numpy())
 coords=ann[['probe_id','panel_rank_rule','context_group','selection_rate_top_panel','mean_beta_across_classes','sd_beta_across_classes']].copy()
 for i in range(scores.shape[1]):coords[f'PC{i+1}']=scores[:,i]
 coords.to_csv(R/f'coordinates_{name}.tsv',sep='\t',index=False)
 pd.DataFrame(model.components_.T,index=means.index,columns=[f'PC{i+1}' for i in range(scores.shape[1])]).to_csv(R/f'loadings_{name}.tsv',sep='\t',index_label='tumor_class')
 pd.DataFrame({'component':[f'PC{i+1}' for i in range(scores.shape[1])],'explained_variance':model.explained_variance_,'explained_variance_ratio':model.explained_variance_ratio_}).to_csv(R/f'variance_{name}.tsv',sep='\t',index=False)
 dist=squareform(pdist(values.to_numpy()));pairs=[]
 for i in range(len(probes)):
  for j in range(i+1,len(probes)):
   pairs.append({'probe_a':probes[i],'probe_b':probes[j],'rank_a':int(panel.iloc[i].panel_rank_rule),'rank_b':int(panel.iloc[j].panel_rank_rule),'euclidean_distance':float(dist[i,j]),'class_profile_pearson_r':float(pearsonr(X.iloc[i],X.iloc[j]).statistic)})
 pairs=pd.DataFrame(pairs).sort_values('euclidean_distance');pairs.to_csv(R/f'pairwise_distances_{name}.tsv',sep='\t',index=False)
 summary['views'][name]={'PC1_explained_variance_ratio':float(model.explained_variance_ratio_[0]),'PC2_explained_variance_ratio':float(model.explained_variance_ratio_[1]),'first_two_variance_ratio':float(model.explained_variance_ratio_[:2].sum()),'PC1_mean_beta_pearson_r':float(pearsonr(scores[:,0],X.mean(axis=1)).statistic),'PC2_mean_beta_pearson_r':float(pearsonr(scores[:,1],X.mean(axis=1)).statistic),'nearest_full_space_pairs':pairs.head(5).to_dict('records')}
 np.testing.assert_allclose(model.inverse_transform(scores),values.to_numpy(),atol=1e-12)
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(R/'SUCCESS.json').write_text(json.dumps({'input_hashes_verified':True,'dev_split_verified':True,'panel_annotation_frequency_verified':True,'pca_full_reconstruction_verified':True},indent=2)+'\n')
print(json.dumps(summary,indent=2))
