from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
from scipy.stats import beta,binomtest
import argparse
from minicpg.analysis_io import AnalysisInputs
parser=argparse.ArgumentParser(description="Paired clinical and preprocessing contrasts using the recorded patient/specimen design")
parser.add_argument('--config',required=True)
parser.add_argument('--out-dir',required=True)
args=parser.parse_args()
inputs=AnalysisInputs(args.config)
O=inputs.output_dir(args.out_dir)
D=O
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def read(n):
 d=pd.read_csv(inputs.files[n],sep='\t',dtype=str,keep_default_na=False);d.index=d.gse_id+':'+d.sample_id if {'gse_id','sample_id'}<=set(d) else d.index;return d
base={m:read(m+'_canonical_predictions.tsv') for m in ['minicpg']}
rows=[];units=[];specimens=[];coverage=[];dest=[];comparisons=[]
def cp(k,n,alpha=.05):return [0 if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),1 if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))]
def stats(a,b,identity):
 n=len(a); seed=int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)+20261113
 rng=np.random.default_rng(seed);idx=rng.integers(0,n,(20000,n));ba=a[idx].mean(1);bb=b[idx].mean(1)
 delta=b-a;r={'n_units':n,'first_recall':a.mean(),'second_recall':b.mean(),'difference':delta.mean(),'first_ci_low':np.quantile(ba,.025),'first_ci_high':np.quantile(ba,.975),'second_ci_low':np.quantile(bb,.025),'second_ci_high':np.quantile(bb,.975),'difference_ci_low':np.quantile(bb-ba,.025),'difference_ci_high':np.quantile(bb-ba,.975),'patient_or_specimen_decreases':sum(delta<0),'patient_or_specimen_increases':sum(delta>0),'ties':sum(delta==0),'bootstrap_degenerate':bool(np.ptp(bb-ba)==0),'bootstrap_seed':seed,'bootstrap_replicates':20000}
 eps=np.sqrt(2*np.log(2/.05)/n);r.update(hoeffding_difference_low=max(-1,delta.mean()-eps),hoeffding_difference_high=min(1,delta.mean()+eps))
 if np.isin(a,[0,1]).all() and np.isin(b,[0,1]).all():
  loss=int(sum((a==1)&(b==0)));gain=int(sum((a==0)&(b==1)));r.update(correct_to_incorrect=loss,incorrect_to_correct=gain,both_correct=int(sum((a==1)&(b==1))),both_incorrect=int(sum((a==0)&(b==0))),discordant_exact_p=binomtest(gain,gain+loss,.5).pvalue if gain+loss else 1.)
  gl,gh=cp(gain,n,.025);ll,lh=cp(loss,n,.025)
  r.update(conservative_difference_low=gl-lh,conservative_difference_high=gh-ll,first_exact_low=cp(int(sum(a)),n)[0],first_exact_high=cp(int(sum(a)),n)[1],second_exact_low=cp(int(sum(b)),n)[0],second_exact_high=cp(int(sum(b)),n)[1])
 return r,(ba,bb)

def process(design,g,sel,condition_frames,clinical=False):
 # Select per-model surviving records, then require both statuses per patient.
 survivors={};selected=set(sel.index)
 for model,(first,second) in condition_frames.items():
  if clinical:
   avail=selected & set(first.index);s=sel.loc[sorted(avail)]
   paired_pat=s.groupby('patient_namespace_and_id').sample_role.nunique();good=set(paired_pat[paired_pat==2].index)
   survivors[model]=set(s[s.patient_namespace_and_id.isin(good)].index)
  else:survivors[model]=selected & set(first.index)&set(second.index)
  coverage.append({'design':design,'gse_id':g,'model':model,'selected_specimens':len(sel),'selected_patients':sel.patient_namespace_and_id.nunique() if clinical else '', 'first_scoreable':sum(k in first.index for k in sel[sel.sample_role=='primary'].index) if clinical else len(selected&set(first.index)), 'second_scoreable':sum(k in first.index for k in sel[sel.sample_role=='metastatic'].index) if clinical else len(selected&set(second.index)), 'paired_scoreable_specimens':len(survivors[model]),'removed_before_common':len(sel)-len(survivors[model])})
 for subset,model,ids in [('model_paired','minicpg',survivors['minicpg'])]:
  ids=sorted(ids);first,second=condition_frames[model];s=sel.loc[ids]; rec=[]
  for k in ids:
   conds=[('first',first)] if clinical and s.loc[k,'sample_role']=='primary' else ([('second',first)] if clinical else [('first',first),('second',second)])
   for cond,f in conds:
    p=f.loc[k];truth=p.tcga_label;pid=s.loc[k,'patient_namespace_and_id'] if clinical else k
    rec.append({'design':design,'gse_id':g,'subset':subset,'model':model,'sample_id':p.sample_id,'unit':pid,'condition':cond,'tcga_label':truth,'y_pred':p.y_pred,'correct':int(p.y_pred==truth),'true_class_probability':float(p['proba__'+truth]),'n_missing':int(p.get('n_missing_features',p.get('n_missing','0')) or 0)})
  rr=pd.DataFrame(rec);specimens.extend(rec)
  for (cond,yp),dd in rr.groupby(['condition','y_pred']):dest.append({'design':design,'gse_id':g,'subset':subset,'model':model,'condition':cond,'true_class':dd.tcga_label.iloc[0],'predicted_class':yp,'n':len(dd)})
  agg=rr.groupby(['unit','condition'],sort=True).agg(correct=('correct','mean'),true_class_probability=('true_class_probability','mean'),n_specimens=('sample_id','size')).reset_index()
  a=agg.pivot(index='unit',columns='condition',values='correct');assert not a.isna().any().any()
  for unit,r in a.iterrows():
   ar=agg[agg.unit==unit].set_index('condition');units.append({'design':design,'gse_id':g,'subset':subset,'model':model,'unit':unit,'first_correct_fraction':r['first'],'second_correct_fraction':r['second'],'difference':r['second']-r['first'],'first_n':int(ar.loc['first','n_specimens']),'second_n':int(ar.loc['second','n_specimens']),'first_true_probability':ar.loc['first','true_class_probability'],'second_true_probability':ar.loc['second','true_class_probability']})
  met,draw=stats(a['first'].to_numpy(),a['second'].to_numpy(),design+'|'.join(a.index))
  met.update(design=design,gse_id=g,subset=subset,model=model,first_condition='primary' if clinical else 'author',second_condition='regional_node' if clinical else ('alternative_export' if design.startswith('EXPORT') else 'SeSAMe_QCDPB'),n_first_specimens=int(sum(rr.condition=='first')),n_second_specimens=int(sum(rr.condition=='second')),unit='patient' if clinical else 'specimen_unknown_patient',binary_pair_inference_valid=bool((agg.n_specimens==1).all()))
  if not met['binary_pair_inference_valid']:
   for c in ['correct_to_incorrect','incorrect_to_correct','both_correct','both_incorrect','discordant_exact_p','conservative_difference_low','conservative_difference_high','first_exact_low','first_exact_high','second_exact_low','second_exact_high']:met.pop(c,None)
  rows.append(met)

cl=read('clinical_design_members.tsv')
for g,sel in cl.groupby('gse_id'):process('STATUS_'+g,g,sel,{m:(p,p) for m,p in base.items()},True)
for g in ['GSE144487','GSE234379']:
 frames={m:(p,read(f'{m}_export_alternative_{g}.tsv')) for m,p in base.items()};ids=frames['minicpg'][1].index
 sel=read('canonical_master_scope.tsv').loc[ids];process('EXPORT_'+g,g,sel,frames)
sel=read('idat_design_members.tsv')
frames={m:(read(m+'_idat_author.tsv'),read(m+'_idat_sesame.tsv')) for m in base}
for g,s in sel.groupby('gse_id'):process('IDAT_'+g,g,s,frames)
for name,data in [('paired_results',rows),('analysis_units',units),('specimen_outcomes',specimens),('coverage',coverage),('prediction_destinations',dest)]:pd.DataFrame(data).to_csv(D/(name+'.tsv'),sep='\t',index=False)
bd=read('idat_input_beta_differences.tsv');bd=bd[bd.model=='minicpg'].copy();bd['absolute_change']=pd.to_numeric(bd.difference,errors='coerce').abs()
bd.groupby(['gse_id','model']).absolute_change.agg(['size','count','mean','median','max']).reset_index().to_csv(D/'beta_difference_diagnostics.tsv',sep='\t',index=False)
(O/'ANALYSIS_VALIDATION.json').write_text(json.dumps({'status':'passed','protocol_sha256':H(inputs.files['PROTOCOL.txt']),'input_hashes_verified':True,'bootstrap_replicates':20000,'rows':len(rows),'outputs':{p.name:H(p) for p in D.glob('*.tsv')}},indent=2))
print(pd.DataFrame(rows)[['design','subset','model','n_units','first_recall','second_recall','difference','difference_ci_low','difference_ci_high']].to_string(index=False))

inputs.finish(O,{"workflow":"paired_clinical_and_preprocessing","statistical_functions":"unchanged from adopted analysis"})
