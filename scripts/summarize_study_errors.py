#!/usr/bin/env python3
"""Aggregate study/class errors and supplied source annotations without changing eligibility."""
import argparse
import pandas as pd
from minicpg.analysis_io import AnalysisInputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--out-dir',required=True)
    args=parser.parse_args();bound=AnalysisInputs(args.config);keys=['gse_id','sample_id']
    read=lambda k:pd.read_csv(bound.files[k],sep='\t',dtype=str,keep_default_na=False)
    pred=read('predictions');master=read('canonical_master');inventory=read('canonical_inventory')
    for d in [pred,master,inventory]:
        if d.duplicated(keys).any():raise ValueError('Duplicate specimen keys')
    for name,d in [('canonical_master',master),('canonical_inventory',inventory)]:
        joined=pred[keys+['tcga_label']].merge(d[keys+['tcga_label']],on=keys,how='left',suffixes=('','_source'),validate='one_to_one')
        if not joined.tcga_label.eq(joined.tcga_label_source).all():raise ValueError(f'Canonical truth mismatch: {name}')
    pred['correct']=pred.tcga_label.eq(pred.y_pred).astype(int)
    out=bound.output_dir(args.out_dir)
    by=pred.groupby(['gse_id','tcga_label']).agg(n=('correct','size'),correct=('correct','sum')).reset_index()
    by['errors']=by.n-by.correct;by['recall']=by.correct/by.n
    by.to_csv(out/'study_class_recall.tsv',sep='\t',index=False)
    pred.groupby(['gse_id','tcga_label','y_pred']).size().rename('n').reset_index().to_csv(out/'study_confusion.tsv',sep='\t',index=False)
    if 'annotations' in bound.files:
        ann=read('annotations')
        if ann.duplicated(keys).any():raise ValueError('Annotation keys must be unique')
        fields=bound.config['annotation_fields']
        if any(c in keys for c in fields):raise ValueError('Do not aggregate on specimen identifiers')
        joined=pred.merge(ann[keys+fields],on=keys,how='left',validate='one_to_one')
        joined[fields]=joined[fields].fillna('unreported')
        joined.groupby(['gse_id','tcga_label']+fields,dropna=False).agg(n=('correct','size'),correct=('correct','sum')).reset_index().to_csv(out/'annotation_error_counts.tsv',sep='\t',index=False)
    bound.finish(out,{'workflow':'study_error_aggregation','interpretation':'Descriptive co-occurrence; source annotations do not alter eligibility or establish causality'})


if __name__=='__main__':main()
