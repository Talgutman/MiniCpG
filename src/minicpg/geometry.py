"""Arithmetic class profiles, methylation shift and class-label permutation."""
from __future__ import annotations
import argparse,itertools,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata,beta
from minicpg.analysis_io import AnalysisInputs
from minicpg.inference import load_model_bundle


def imputed_beta(frame, bundle):
    probes=bundle['probe_ids']
    if not set(probes)<=set(frame):raise ValueError('Geometry input must contain every locked probe column')
    x=frame[probes].apply(pd.to_numeric,errors='coerce').astype(float)
    x=x.where(np.isfinite(x)&x.ge(0)&x.le(1))
    med=pd.Series(bundle['model'].named_steps['imputer'].statistics_,index=probes)
    x=x.fillna(med)
    if not np.isfinite(x).all().all():raise ValueError('Nonfinite class geometry after locked imputation')
    return x


def class_profiles(frame, bundle, label='tcga_label'):
    probes=bundle['probe_ids'];x=imputed_beta(frame,bundle)
    x[label]=frame[label].to_numpy()
    return x.groupby(label)[probes].mean()


def dispersion_tables(datasets, bundle):
    """Descriptive within-class dispersion and UCS/UCEC development-reference distances."""
    profiles={k:class_profiles(d,bundle) for k,d in datasets.items()}
    dispersion=[];separation=[];study=[]
    pair=['TCGA-UCS','TCGA-UCEC'];dev=profiles.get('development')
    for cohort,d in datasets.items():
        values=imputed_beta(d,bundle);profile=profiles[cohort]
        for c in sorted(profile.index):
            selected=d.tcga_label.eq(c).to_numpy();v=values.loc[selected].to_numpy()
            distance=np.linalg.norm(v-profile.loc[c].to_numpy(),axis=1)
            dispersion.append(dict(cohort=cohort,tcga_label=c,n=len(v),median_to_own_centroid=float(np.median(distance)),mean_to_own_centroid=float(np.mean(distance)),p25_to_own_centroid=float(np.quantile(distance,.25)),p75_to_own_centroid=float(np.quantile(distance,.75))))
        if set(pair)<=set(profile.index):
            separation.append(dict(cohort=cohort,distance=float(np.linalg.norm(profile.loc[pair[0]]-profile.loc[pair[1]]))))
        if dev is not None and set(pair)<=set(dev.index):
            groups=d.assign(gse_id=d['gse_id'] if 'gse_id' in d else 'TCGA')
            for (g,c),z in groups[groups.tcga_label.isin(pair)].groupby(['gse_id','tcga_label']):
                v=values.loc[z.index].to_numpy();center=v.mean(axis=0);other=pair[1] if c==pair[0] else pair[0]
                study.append(dict(cohort=cohort,gse_id=g,tcga_label=c,n=len(v),median_to_study_centroid=float(np.median(np.linalg.norm(v-center,axis=1))),distance_to_dev_own=float(np.linalg.norm(center-dev.loc[c])),distance_to_dev_other=float(np.linalg.norm(center-dev.loc[other]))))
    return pd.DataFrame(dispersion),pd.DataFrame(separation),pd.DataFrame(study)


def permutation_test(similarity, confusion_rate, n_classes, permutations=999999, seed=20261113):
    if permutations<1:raise ValueError('permutations must be positive')
    i,j=np.triu_indices(n_classes,k=1)
    if len(similarity)!=len(i) or len(confusion_rate)!=len(i):raise ValueError('Expected every unordered class pair, including zero-confusion pairs')
    xr,yr=rankdata(similarity,method='average'),rankdata(confusion_rate,method='average')
    xr-=xr.mean();yr-=yr.mean();denom=np.linalg.norm(xr)*np.linalg.norm(yr)
    if denom==0:raise ValueError('Spearman association is undefined for constant ranks')
    observed=float(np.dot(xr,yr)/denom)
    xm=np.full((n_classes,n_classes),np.nan);xm[i,j]=xm[j,i]=xr
    rng=np.random.default_rng(seed);null=np.empty(permutations)
    for offset in range(0,permutations,2048):
        count=min(2048,permutations-offset)
        p=np.stack([rng.permutation(n_classes) for _ in range(count)])
        null[offset:offset+count]=np.einsum('ij,j->i',xm[p[:,i],p[:,j]],yr)/denom
    tolerance=100*np.finfo(float).eps*max(1,abs(observed))
    exceed=int((np.abs(null)>=abs(observed)-tolerance).sum())
    low=0. if exceed==0 else float(beta.ppf(.025,exceed,permutations-exceed+1))
    high=1. if exceed==permutations else float(beta.ppf(.975,exceed+1,permutations-exceed))
    return dict(observed_spearman=observed,p_two_sided_plus_one=(exceed+1)/(permutations+1),
                exceedances_absolute=exceed,n_permutations=permutations,seed=seed,
                monte_carlo_tail_probability_ci95=[low,high]),null


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--out-dir',required=True)
    args=parser.parse_args();bound=AnalysisInputs(args.config);cfg=bound.config
    bundle=load_model_bundle(bound.files['model'])
    read=lambda name:pd.read_csv(bound.files[name],sep='\t',float_precision='round_trip')
    external=read('external');pred=read('predictions');keys=['gse_id','sample_id']
    if external.duplicated(keys).any() or pred.duplicated(keys).any():raise ValueError('Duplicate specimen keys')
    e=external.set_index(keys).sort_index();p=pred.set_index(keys).sort_index()
    if not e.index.equals(p.index) or not e.tcga_label.equals(p.tcga_label):raise ValueError('Matrix/prediction membership or truth mismatch')
    for name in ['canonical_master','canonical_inventory']:
        canonical=read(name)
        if canonical.duplicated(keys).any():raise ValueError('Duplicate canonical keys')
        canonical=canonical.set_index(keys)
        if not e.index.isin(canonical.index).all():raise ValueError('Matrix contains noncanonical specimens')
        selected=canonical.loc[e.index]
        if not selected.tcga_label.equals(e.tcga_label):raise ValueError('Canonical truth mismatch')
        if not selected['include'].astype(str).eq('1').all() or not selected.target_bucket.eq('main').all():raise ValueError('Geometry requires the adopted primary cohort')
    if not pred.y_pred.isin(bundle['class_order']).all():raise ValueError('Unknown predicted class')
    prof=class_profiles(external,bundle);support=external.tcga_label.value_counts();classes=sorted(prof.index)
    counts=pred.groupby(['tcga_label','y_pred']).size().to_dict();rows=[]
    for a,b in itertools.combinations(classes,2):
        dist=float(np.linalg.norm(prof.loc[a]-prof.loc[b]));ab=counts.get((a,b),0);ba=counts.get((b,a),0)
        rows.append(dict(class_a=a,class_b=b,n_a=int(support[a]),n_b=int(support[b]),n_a_pred_b=int(ab),n_b_pred_a=int(ba),pair_confusion_rate=(ab+ba)/(support[a]+support[b]),euclidean_distance=dist))
    pairs=pd.DataFrame(rows);dist=pairs.euclidean_distance.to_numpy()
    if np.ptp(dist)==0:raise ValueError('All pair distances equal')
    pairs['similarity']=1-(dist-dist.min())/np.ptp(dist)
    result,null=permutation_test(pairs.similarity,pairs.pair_confusion_rate,len(classes),cfg.get('permutations',999999),cfg.get('seed',20261113))
    out=bound.output_dir(args.out_dir);pairs.to_csv(out/'pairs.tsv',sep='\t',index=False);prof.to_csv(out/'external_mean_profiles.tsv',sep='\t')
    hist,edges=np.histogram(null,bins=np.linspace(-1,1,101));pd.DataFrame({'lower':edges[:-1],'upper':edges[1:],'count':hist}).to_csv(out/'null_histogram.tsv',sep='\t',index=False)
    datasets={'external_primary':external}
    if 'heldout' in bound.files:
        held=read('heldout');hp=class_profiles(held,bundle,cfg.get('heldout_label','project_id'))
        datasets['heldout']=held.rename(columns={cfg.get('heldout_label','project_id'):'tcga_label'})
        recalls=read('recalls').set_index('tcga_label');shifts=[]
        for c in sorted(set(prof.index)&set(hp.index)&set(recalls.index)):
            shifts.append(dict(tcga_label=c,mean_absolute_beta_shift=float(np.abs(prof.loc[c]-hp.loc[c]).mean()),recall_decline=float(recalls.loc[c,'heldout_recall']-recalls.loc[c,'external_recall'])))
        shifts=pd.DataFrame(shifts);rho,pvalue=spearmanr(shifts.mean_absolute_beta_shift,shifts.recall_decline)
        shifts.to_csv(out/'methylation_shift.tsv',sep='\t',index=False)
        result['methylation_shift']={'spearman':float(rho),'nominal_two_sided_p':float(pvalue),'n_classes':len(shifts)}
    if 'development' in bound.files:
        datasets['development']=read('development').rename(columns={cfg.get('development_label','project_id'):'tcga_label'})
        dev=class_profiles(read('development'),bundle,cfg.get('development_label','project_id'));dc=sorted(dev.index)
        a=dev.loc[dc].to_numpy();distance=np.linalg.norm(a[:,None,:]-a[None,:,:],axis=2);np.fill_diagonal(distance,np.inf)
        nearest=np.argmin(distance,axis=1)
        pd.DataFrame([{'class':c,'nearest_class':dc[nearest[i]],'distance':distance[i,nearest[i]],'mutual_nearest':bool(nearest[nearest[i]]==i)} for i,c in enumerate(dc)]).to_csv(out/'development_nearest_neighbors.tsv',sep='\t',index=False)
    for filename,table in zip(['class_dispersion','ucs_ucec_separation','ucs_ucec_study_geometry'],dispersion_tables(datasets,bundle)):
        table.to_csv(out/(filename+'.tsv'),sep='\t',index=False)
    (out/'association.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    bound.finish(out,{'workflow':'mean_class_geometry','model_refitted':False,'class_count':len(classes)})


if __name__=='__main__':main()
