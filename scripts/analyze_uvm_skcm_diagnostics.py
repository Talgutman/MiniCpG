"""Recorded UVM/SKCM study geometry, missing-probe and comparator diagnostics."""
import argparse,json
import numpy as np
import pandas as pd
from minicpg.analysis_io import AnalysisInputs
from minicpg.inference import load_model_bundle
CLASSES=['TCGA-UVM','TCGA-SKCM']
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--out-dir',required=True)
    args=parser.parse_args();bound=AnalysisInputs(args.config);OUT=bound.output_dir(args.out_dir)
    def locked(k):return bound.files[k]
    def save(df,name):df.to_csv(OUT/name,sep='\t',index=False)
    canonical={}
    for k in ['canonical_master','canonical_inventory']:
        x=pd.read_csv(locked(k),sep='\t',low_memory=False);canonical[k]=x.query('include == 1 and target_bucket == "main"')
    raw=pd.read_csv(locked('minicpg_matrix_main'),sep='\t');pred=pd.read_csv(locked('predictions'),sep='\t')
    keys=['gse_id','sample_id']
    def idx(t):
        assert not t.duplicated(keys).any();return t.set_index(keys).sort_index()
    a,b=idx(raw),idx(pred);assert a.index.equals(b.index) and a.tcga_label.equals(b.tcga_label)
    for k,c in canonical.items():
        c=idx(c);assert a.index.isin(c.index).all() and a.tcga_label.equals(c.loc[a.index,'tcga_label'])
        if k=='canonical_inventory':assert a.index.equals(c.index)
    probes=[c for c in raw if c.startswith('cg')]
    bundle=load_model_bundle(locked('minicpg_model'))
    assert bundle['probe_ids']==probes
    imputation=pd.Series(bundle['model'].named_steps['imputer'].statistics_,index=probes)
    datasets={};missing=[];full_counts={}
    for cohort,stem in [('development','dev'),('heldout','lockbox')]:
        p=locked(cohort+'_matrix');s=locked(cohort+'_manifest')
        d=pd.read_csv(p,sep='\t');sp=pd.read_csv(s,sep='\t');assert d.sample_id.is_unique and sp.sample_id.is_unique
        d=d.set_index('sample_id').sort_index();sp=sp.set_index('sample_id').sort_index();assert d.index.equals(sp.index) and d.project_id.equals(sp.project_id)
        d=d.reset_index().rename(columns={'project_id':'tcga_label'});d['gse_id']=cohort;datasets[cohort]=d;full_counts[cohort]=len(d)
    heldpath=locked('heldout_predictions')
    held=pd.read_csv(heldpath,sep='\t');datasets['heldout']=datasets['heldout'].merge(held[['sample_id','y_pred']],on='sample_id',validate='one_to_one')
    datasets['external_primary']=raw.merge(pred[keys+['y_pred']],on=keys,validate='one_to_one')
    rows=[];profiles={}
    for cohort,d in datasets.items():
        d=d[d.tcga_label.isin(CLASSES)].copy();assert [c for c in d if c.startswith('cg')]==probes
        d['cohort']=cohort;d['missing_count']=d[probes].isna().sum(axis=1)
        for (g,c),sub in d.groupby(['gse_id','tcga_label']):
            for p in probes:missing.append(dict(cohort=cohort,gse_id=g,tcga_label=c,probe=p,n=len(sub),missing=int(sub[p].isna().sum())))
        d[probes]=d[probes].fillna(imputation);assert np.isfinite(d[probes]).all().all()
        d['outcome']=np.where(d.get('y_pred',pd.Series('',index=d.index))==d.tcga_label,'correct',np.where(d.get('y_pred',pd.Series('',index=d.index))=='TCGA-SKCM','UVM_to_SKCM','other_error'))
        if cohort=='development':d['outcome']='development_reference'
        datasets[cohort]=d;profiles[cohort]=d.groupby('tcga_label')[probes].mean()
    dev=profiles['development'];summ=[];decomp=[]
    for cohort,d in datasets.items():
        for c in CLASSES:
            sub=d[d.tcga_label==c];v=sub[probes].to_numpy();pooled=v.mean(axis=0)
            total=((v-pooled)**2).sum();within=between=0.
            for g,z in sub.groupby('gse_id'):
                vals=z[probes].to_numpy();center=vals.mean(axis=0);within+=((vals-center)**2).sum();between+=len(z)*((center-pooled)**2).sum()
            np.testing.assert_allclose(total,within+between,atol=1e-10)
            decomp.append(dict(cohort=cohort,tcga_label=c,n=len(sub),total_squared_scatter=total,within_study_squared_scatter=within,between_study_squared_scatter=between,between_study_fraction=between/total))
        for (g,c),z in d.groupby(['gse_id','tcga_label']):
            v=z[probes].to_numpy();center=v.mean(axis=0);other=next(c2 for c2 in CLASSES if c2!=c)
            own=np.linalg.norm(v-dev.loc[c].to_numpy(),axis=1);oth=np.linalg.norm(v-dev.loc[other].to_numpy(),axis=1)
            z=z.copy();z['distance_dev_own']=own;z['distance_dev_other']=oth;z['other_dev_closer']=oth<own;z['distance_study_mean']=np.linalg.norm(v-center,axis=1)
            z['distance_pooled_class_mean']=np.linalg.norm(v-profiles[cohort].loc[c].to_numpy(),axis=1)
            rows.append(z)
            summ.append(dict(cohort=cohort,gse_id=g,tcga_label=c,n=len(z),correct=int((z.outcome=='correct').sum()) if cohort!='development' else np.nan,dispersion_median_to_study_mean=float(np.median(z.distance_study_mean)),mean_shift_euclidean_to_dev_own=float(np.linalg.norm(center-dev.loc[c])),mean_shift_mae_to_dev_own=float(np.mean(np.abs(center-dev.loc[c]))),study_mean_distance_to_dev_other=float(np.linalg.norm(center-dev.loc[other])),n_other_dev_closer=int((oth<own).sum())))
    allsamples=pd.concat(rows,ignore_index=True);save(allsamples,'sample_geometry_and_probes.tsv');save(pd.DataFrame(summ),'study_geometry.tsv');save(pd.DataFrame(decomp),'scatter_decomposition.tsv');save(pd.DataFrame(missing),'probe_missingness.tsv')
    save(allsamples.groupby(['cohort','gse_id','tcga_label','outcome'],dropna=False).agg(n=('sample_id','size'),n_other_dev_closer=('other_dev_closer','sum'),median_distance_dev_own=('distance_dev_own','median'),median_distance_dev_other=('distance_dev_other','median'),median_distance_study_mean=('distance_study_mean','median')).reset_index(),'outcome_geometry.tsv')
    pair=[]
    for co,pr in profiles.items():pair.append(dict(cohort=co,distance=float(np.linalg.norm(pr.loc[CLASSES[0]]-pr.loc[CLASSES[1]]))))
    ext=datasets['external_primary']
    for gu,u in ext[ext.tcga_label=='TCGA-UVM'].groupby('gse_id'):
        for gs,s in ext[ext.tcga_label=='TCGA-SKCM'].groupby('gse_id'):pair.append(dict(cohort=f'{gu}_UVM__{gs}_SKCM',distance=float(np.linalg.norm(u[probes].mean()-s[probes].mean()))))
    save(pd.DataFrame(pair),'pair_distances.tsv')
    delta=dev.loc['TCGA-UVM']-dev.loc['TCGA-SKCM'];rank=delta.abs().rank(method='min',ascending=False).astype(int)
    discr=pd.DataFrame(dict(probe=probes,dev_UVM_mean=dev.loc['TCGA-UVM'].values,dev_SKCM_mean=dev.loc['TCGA-SKCM'].values,dev_signed_difference=delta.values,absolute_dev_difference=delta.abs().values,dev_rank=rank.values)).sort_values('dev_rank');save(discr,'development_probe_separation.tsv')
    pr=[]
    for (co,g,c,outcome),z in allsamples.groupby(['cohort','gse_id','tcga_label','outcome']):
        for p in probes:pr.append(dict(cohort=co,gse_id=g,tcga_label=c,outcome=outcome,n=len(z),probe=p,dev_rank=int(rank[p]),mean=float(z[p].mean()),sd=float(z[p].std()),q25=float(z[p].quantile(.25)),q50=float(z[p].quantile(.5)),q75=float(z[p].quantile(.75)),shift_from_dev_own=float(z[p].mean()-dev.loc[c,p])))
    save(pd.DataFrame(pr),'probe_distribution_by_outcome.tsv')
    shifts=[]
    for (co,g,c),z in allsamples.groupby(['cohort','gse_id','tcga_label']):
        sq=(z[probes].mean()-dev.loc[c])**2;order=sq.sort_values(ascending=False);den=sq.sum()
        for p in probes:shifts.append(dict(cohort=co,gse_id=g,tcga_label=c,n=len(z),probe=p,dev_separation_rank=int(rank[p]),mean=float(z[p].mean()),shift=float(z[p].mean()-dev.loc[c,p]),squared_shift_fraction=float(sq[p]/den) if den>0 else 0.,shift_rank=int(sq.rank(ascending=False,method='min')[p])))
    save(pd.DataFrame(shifts),'probe_shift_by_study.tsv')
    mdpath=locked('methydeep_predictions');md=pd.read_csv(mdpath,sep='\t');assert not md.duplicated(keys).any()
    mdmat=pd.read_csv(locked('methydeep_matrix_main'),sep='\t');assert idx(md).index.equals(idx(mdmat).index) and idx(md).tcga_label.equals(idx(mdmat).tcga_label)
    md=md.rename(columns={'tcga_label':'methydeep_truth','y_pred':'methydeep_prediction','y_pred_raw_argmax':'methydeep_raw_argmax'})
    comp=allsamples.query('cohort == "external_primary"').merge(md[keys+['methydeep_truth','methydeep_prediction','methydeep_raw_argmax']],on=keys,how='left',validate='one_to_one')
    covered=comp.methydeep_prediction.notna();assert (comp.loc[covered,'tcga_label']==comp.loc[covered,'methydeep_truth']).all();assert (comp.loc[covered,'methydeep_prediction']==comp.loc[covered,'methydeep_raw_argmax']).all()
    comp['methydeep_covered']=covered;comp['methydeep_correct']=covered & (comp.methydeep_prediction==comp.tcga_label)
    save(comp.drop(columns=probes),'matched_comparator.tsv');save(comp.groupby(['gse_id','tcga_label','outcome'],dropna=False).agg(n=('sample_id','size'),methydeep_covered=('methydeep_covered','sum'),methydeep_correct=('methydeep_correct','sum')).reset_index(),'matched_comparator_summary.tsv')
    save(comp.groupby(['gse_id','tcga_label','outcome','methydeep_prediction'],dropna=False).size().reset_index(name='n'),'matched_comparator_destinations.tsv')
    # Diagnostic dimensions observed in every specimen of each study. Not a new
    # panel, classifier, prediction, or replacement for approved 30-probe geometry.
    observed=[];observed_samples=[]
    for (g,c),z in ext.groupby(['gse_id','tcga_label']):
        original=raw[(raw.gse_id==g)&(raw.tcga_label==c)];keep=[p for p in probes if original[p].notna().all()]
        other=next(x for x in CLASSES if x!=c);v=z[keep].to_numpy();center=v.mean(axis=0);own=dev.loc[c,keep].to_numpy();oth=dev.loc[other,keep].to_numpy()
        observed.append(dict(gse_id=g,tcga_label=c,n=len(z),n_observed_probes=len(keep),omitted_probes=';'.join(p for p in probes if p not in keep),observed_mean_shift_euclidean=float(np.linalg.norm(center-own)),observed_mean_distance_other=float(np.linalg.norm(center-oth)),observed_dev_pair_distance=float(np.linalg.norm(own-oth))))
        for i,r in enumerate(z.itertuples()):observed_samples.append(dict(gse_id=g,sample_id=r.sample_id,tcga_label=c,outcome=r.outcome,n_probes=len(keep),observed_distance_own=float(np.linalg.norm(v[i]-own)),observed_distance_other=float(np.linalg.norm(v[i]-oth)),observed_other_closer=bool(np.linalg.norm(v[i]-oth)<np.linalg.norm(v[i]-own))))
    save(pd.DataFrame(observed),'observed_dimensions_diagnostic.tsv');save(pd.DataFrame(observed_samples),'observed_dimensions_samples.tsv')
    coverage_path=locked('coverage_status');coverage=pd.read_csv(coverage_path,sep='\t');save(coverage[(coverage.target_bucket=='main')&coverage.tcga_label.isin(CLASSES)],'exact_sample_coverage.tsv')
    bound.finish(OUT,{'workflow':'UVM_SKCM_diagnostics','policy':'Stored training medians and arithmetic mean centroids; original frozen predictions'})

if __name__=='__main__':main()
