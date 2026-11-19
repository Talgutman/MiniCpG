from pathlib import Path
import sys,json,pickle
import numpy as np,pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.metrics import f1_score,accuracy_score
import argparse
from minicpg.analysis_io import BoundStage,sha256 as digest
parser=argparse.ArgumentParser(description="Consensus versus single-split panels on a common scoreable subset")
parser.add_argument('--config',required=True)
parser.add_argument('--out-dir',required=True)
parser.add_argument('--bucket',default='main',choices=['main'])
args=parser.parse_args();stage=BoundStage(args.config,args.out_dir);bucket=args.bucket
p=json.loads(stage.input('plan').read_text())
panels=p['primary_panels'];seeds=p['seeds']
with stage.input('model').open('rb') as f:classes=pickle.load(f)['class_order']
C=len(classes);enc={c:i for i,c in enumerate(classes)};K=['gse_id','sample_id'];base=None;preds=[];metricrows=[];qcs=[]
for row in panels:
 for seed in seeds:
  key=row['panel_id']+'_'+str(seed)+'_'+bucket;d=pd.read_csv(stage.input(key),sep='\t',float_precision='round_trip')
  if base is None:base=d[K+['tcga_label']].copy();axis=classes
  assert base.equals(d[K+['tcga_label']]);pr=d[['proba__'+c for c in classes]].to_numpy();assert np.isfinite(pr).all() and np.allclose(pr.sum(axis=1),1,atol=1e-12);assert np.array_equal(np.array(classes)[pr.argmax(axis=1)],d.y_pred);preds.append(d.y_pred.map(enc).to_numpy());metricrows.append(dict(panel_id=row['panel_id'],seed=seed,n=len(d),macro_f1=f1_score(d.tcga_label,d.y_pred,labels=axis,average='macro',zero_division=0),macro_f1_fixed27=f1_score(d.tcga_label,d.y_pred,labels=classes,average='macro',zero_division=0),macro_f1_present=f1_score(d.tcga_label,d.y_pred,labels=sorted(base.tcga_label.unique()),average='macro',zero_division=0),accuracy=accuracy_score(d.tcga_label,d.y_pred)));qcs.append(dict(panel_id=row['panel_id'],seed=seed,argmax_verified=True,hash=digest(stage.input(key))))
metrics=pd.DataFrame(metricrows);metrics.to_csv(stage.output('metrics_by_panel_seed'),sep='\t',index=False);pd.DataFrame(qcs).to_csv(stage.output('prediction_checks'),sep='\t',index=False)
u=base.merge(pd.read_csv(stage.input('units'),sep='\t'),on=K,validate='one_to_one');assert u.resampling_unit.notna().all();y=base.tcga_label.map(enc).to_numpy();pred=np.stack(preds);M,N=pred.shape;cl,cn=pd.factorize(u.resampling_unit,sort=True);cnt=np.bincount(cl);bal=1/cnt[cl];u['patient_balanced_weight']=bal;u.to_csv(stage.output('sample_units'),sep='\t',index=False);groups={}
for c in range(len(cn)):groups.setdefault(tuple(np.unique(y[cl==c])),[]).append(c)
ax=np.array([enc[c] for c in axis]);present=np.unique(y);rows=(np.arange(M)[:,None]*C*C+y[None,:]*C+pred).ravel();cols=np.tile(np.arange(N),M);mat=sparse.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(M*C*C,N)).tocsr()
def fvalues(w):
 cm=np.asarray(mat@w).reshape(M,C,C);tp=cm[:,np.arange(C),np.arange(C)];den=cm.sum(1)+cm.sum(2);f=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0);return f[:,ax].mean(axis=1),f[:,present].mean(axis=1),f
point,pp,fc=fvalues(np.ones(N));bp,_,_=fvalues(bal);assert np.max(abs(point-metrics.macro_f1.to_numpy()))<1e-12;ns=len(seeds);npan=len(panels);pm=metrics.groupby('panel_id')[['macro_f1','macro_f1_fixed27','macro_f1_present','accuracy']].mean().reset_index();meta=pd.DataFrame([{k:v for k,v in row.items() if k!='probes'} for row in panels]);pm=pm.merge(meta,on='panel_id',how='left');pm['rank']=pm.macro_f1.rank(method='min',ascending=False);pm['patient_balanced_macro_f1']=[bp.reshape(npan,ns).mean(axis=1)[[r['panel_id'] for r in panels].index(pid)] for pid in pm.panel_id];pm.to_csv(stage.output('panel_seed_means'),sep='\t',index=False)
meta=meta.set_index('panel_id').loc[[r['panel_id'] for r in panels[1:]]];predictors=['mean_selection_frequency','overlap_consensus'];ranks={k:rankdata(meta[k].to_numpy()) for k in predictors}
def rho(k,scores):
 a=ranks[k]-ranks[k].mean();b=rankdata(scores);b=b-b.mean();den=np.sqrt((a*a).sum()*(b*b).sum());return float((a*b).sum()/den) if den>0 else np.nan
Brep=2000;rng=np.random.default_rng(20261105);boot=np.empty(Brep);bootpresent=np.empty(Brep);bboot=np.empty(Brep);cb=np.empty((Brep,C));corr=np.empty((Brep,len(predictors)))
for i in range(Brep):
 counts=np.zeros(len(cn))
 for ids in groups.values():counts[ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
 w=counts[cl];f,fp,fclass=fvalues(w);bf,_,_=fvalues(w*bal);ss=f.reshape(npan,ns).mean(axis=1);bs=bf.reshape(npan,ns).mean(axis=1);boot[i]=ss[0]-ss[1:].mean();bboot[i]=bs[0]-bs[1:].mean();bootpresent[i]=fp[:ns].mean()-fp[ns:].mean();cb[i]=(fclass[:ns].mean(axis=0)-fclass[ns:].mean(axis=0))/len(ax)
 for j,k in enumerate(predictors):corr[i,j]=rho(k,ss[1:])
 if i%500==0:print(bucket,'bootstrap',i,flush=True)
assert np.max(abs(cb[:,ax].sum(axis=1)-boot))<1e-12;means=point.reshape(npan,ns).mean(axis=1);bmeans=bp.reshape(npan,ns).mean(axis=1);difference=means[0]-means[1:].mean();corrrows=[]
for j,k in enumerate(predictors):
 valid=corr[np.isfinite(corr[:,j]),j];lo,hi=np.quantile(valid,[.025,.975]);corrrows.append(dict(predictor=k,spearman=rho(k,means[1:]),ci_low=lo,ci_high=hi,valid_bootstrap=len(valid),panels=len(means)-1))
pd.DataFrame(corrrows).to_csv(stage.output('correlations'),sep='\t',index=False);pd.DataFrame(dict(iteration=np.arange(Brep),difference=boot,present_class_difference=bootpresent,patient_balanced_difference=bboot)).to_csv(stage.output('bootstrap_replicates'),sep='\t',index=False)
contrast=fc[:ns].mean(axis=0)-fc[ns:].mean(axis=0);cov=np.cov(cb,rowvar=False);allocation=cov.sum(axis=1);totalvar=np.var(boot,ddof=1);diag=pd.DataFrame(dict(tcga_label=classes,support=[int((y==i).sum()) for i in range(C)],class_F1_difference=contrast,contribution=contrast/len(ax),ci_low=np.quantile(cb,.025,axis=0),ci_high=np.quantile(cb,.975,axis=0),covariance_with_total=allocation,signed_share_total_variance=allocation/totalvar));diag.to_csv(stage.output('class_uncertainty'),sep='\t',index=False);cr=[]
for i,row in enumerate(panels):
 for j,c in enumerate(classes):cr.append(dict(panel_id=row['panel_id'],tcga_label=c,mean_f1=fc[i*ns:(i+1)*ns,j].mean(),support=int((y==j).sum())))
pd.DataFrame(cr).to_csv(stage.output('class_metrics'),sep='\t',index=False);np.savez_compressed(stage.output('bootstrap_arrays'),class_contributions=cb,correlations=corr)
lo,hi=np.quantile(boot,[.025,.975]);blo,bhi=np.quantile(bboot,[.025,.975]);q=dict(status='passed',bucket=bucket,n=N,panels=npan,seeds=seeds,models=M,consensus_macro_f1=means[0],alternative_mean_macro_f1=means[1:].mean(),difference=difference,ci_low=lo,ci_high=hi,consensus_rank=int((means[1:]>means[0]+1e-12).sum()+1),patient_balanced_difference=bmeans[0]-bmeans[1:].mean(),patient_balanced_ci_low=blo,patient_balanced_ci_high=bhi,bootstrap_replicates=Brep,unknown_patient_rows=int(u.resampling_unit.str.startswith('unresolved_sample::').sum()),correlations=corrrows,limits=['Fixed original panels/seeds and fullDEV fits; conditional uncertainty, no new panel selection','Common coverage subset, not full external cohort; class attrition explicitly reported','Shared probes/panels are not independent replicates','Patient identity unknown for explicitly counted specimens; study dependence remains','CIs including zero do not establish equivalence']);stage.output('summary').write_text(json.dumps(q,indent=2)+'\n');stage.finish(q);print(json.dumps(q,indent=2))
