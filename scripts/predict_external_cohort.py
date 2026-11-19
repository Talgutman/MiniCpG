#!/usr/bin/env python3
"""Infer canonical cohort rows and emit the evaluation runner's prediction schema."""
import argparse
import pandas as pd
from minicpg.analysis_io import AnalysisInputs
from minicpg.inference import load_model_bundle,predict_beta_table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--out-dir',required=True)
    args=parser.parse_args();bound=AnalysisInputs(args.config);keys=['gse_id','sample_id']
    read=lambda k:pd.read_csv(bound.files[k],sep='\t',dtype=str,keep_default_na=False)
    matrix=read('matrix');master=read('canonical_master');inventory=read('canonical_inventory')
    for d in [matrix,master,inventory]:
        if d.duplicated(keys).any():raise ValueError('Duplicate canonical/matrix keys')
    for label,d in [('master',master),('inventory',inventory)]:
        joined=matrix[keys+['tcga_label']].merge(d,on=keys,how='left',suffixes=('','_source'),validate='one_to_one')
        if not joined.tcga_label.eq(joined.tcga_label_source).all():raise ValueError(f'Canonical identity/truth mismatch: {label}')
        for field,value in bound.config['canonical_filters'].items():
            if not joined[field].eq(value).all():raise ValueError(f'Cohort mismatch: {field}')
    bundle=load_model_bundle(bound.files['model'])
    x=matrix.copy();x.index=pd.Index([str(tuple(row)) for row in matrix[keys].to_numpy()])
    result=predict_beta_table(x,bundle,include_probabilities=True)
    out=bound.output_dir(args.out_dir)
    coverage=matrix[keys+['tcga_label']].copy()
    for col in ['n_measured_probes','n_missing_probes','status']:coverage[col]=result[col].to_numpy()
    coverage.to_csv(out/'coverage.tsv',sep='\t',index=False)
    selected=result.status.eq('predicted').to_numpy()
    pred=matrix.loc[selected,keys+['tcga_label']].reset_index(drop=True)
    pred['y_pred']=result.loc[selected,'predicted_label'].to_numpy()
    for c in bundle['model'].classes_:
        pred['proba__'+c]=result.loc[selected,'prob_'+c.removeprefix('TCGA-')].to_numpy()
    pred.to_csv(out/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
    bound.finish(out,{'workflow':'canonical_external_inference','n_input':len(matrix),'n_predicted':len(pred),'missingness_policy':'at least25 finite beta values in [0,1]; saved training medians','decision_policy':'raw_argmax'})


if __name__=='__main__':main()
