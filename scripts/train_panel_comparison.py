from pathlib import Path
import sys,json,pickle,os
import numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_fscore_support
import argparse
from minicpg.analysis_io import BoundStage
parser=argparse.ArgumentParser(description="Train fixed consensus/single-split panels on development data, one of ten task shards")
parser.add_argument('--config',required=True);parser.add_argument('--out-dir',required=True)
parser.add_argument('--task',type=int,required=True,choices=range(10))
args=parser.parse_args();T=args.task;stage=BoundStage(args.config,args.out_dir)
p=json.loads(stage.input('panel_plan').read_text())
with stage.input('k30_model').open('rb') as f:b=pickle.load(f)
orig=json.loads(stage.input('estimator_plan').read_text());classes=list(b['class_order']);assert set(b['probe_ids'])==set(p['primary_panels'][0]['probes']) and b['decision_policy'] is None
for k,v in orig['estimator'].items():assert b['model'].get_params()['clf__'+k]==v
assert b['model'].named_steps['imputer'].strategy=='median';dev=pd.read_pickle(stage.input('dev_union'));held=pd.read_pickle(stage.input('heldout_union'));assert not set(dev.sample_id)&set(held.sample_id)
matrices={bkt:pd.read_csv(stage.input('matrix_'+bkt),sep='\t') for bkt in ['main']};K=['gse_id','sample_id'];checks=[]
for row in p['primary_panels'][T::10]:
 ps=row['probes'];pid=row['panel_id']
 for seed in p['seeds']:
  model=clone(b['model']);model.set_params(clf__random_state=seed,clf__n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK','4')));model.fit(dev[ps],dev.project_id.to_numpy());assert list(model.classes_)==classes;assert np.array_equal(model.named_steps['imputer'].statistics_,np.nanmedian(dev[ps].to_numpy(),axis=0),equal_nan=True)
  hprob=model.predict_proba(held[ps]);hd=pd.DataFrame({'sample_id':held.sample_id,'tcga_label':held.project_id,'y_pred':np.array(classes)[hprob.argmax(axis=1)]});hd.to_csv(stage.output(pid+'_'+str(seed)+'_heldout'),sep='\t',index=False)
  for bucket,x in matrices.items():
   assert x[ps].isna().sum(axis=1).le(p['max_missing']).all();prob=model.predict_proba(x[ps]);assert np.isfinite(prob).all() and np.allclose(prob.sum(axis=1),1,atol=1e-12);d=x[K+['tcga_label']].copy();d['y_pred']=np.array(classes)[prob.argmax(axis=1)]
   for j,c in enumerate(classes):d['proba__'+c]=prob[:,j]
   if pid=='consensus' and seed==orig['canonical_seed']:
    # Full DEV consensus reproduction must match the fresh locked-model predictions.
    rp=pd.read_csv(stage.input('base_predictions_'+bucket),sep='\t');rr=d[K].merge(rp,on=K,validate='one_to_one');delta=float(np.max(abs(rr[['proba__'+c for c in classes]].to_numpy()-prob)));assert delta<1e-12;checks.append(dict(bucket=bucket,consensus_seed=seed,maximum_probability_delta=delta))
   destkey=pid+'_'+str(seed)+'_'+bucket;d.to_csv(stage.output(destkey),sep='\t',index=False)
  print(pid,seed,'complete',flush=True)
stage.output('validation').write_text(json.dumps(dict(status='passed',task=T,consensus_checks=checks,dev_medians_verified=True,estimator_settings_verified=True,seeds=p['seeds']),indent=2)+'\n');stage.finish({'task':T,'consensus_checks':checks,'training_scope':'fixed full DEV only'})
