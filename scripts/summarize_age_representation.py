#!/usr/bin/env python3
"""Aggregate curated age categories on explicitly supplied cohort memberships."""
import argparse
import pandas as pd
from minicpg.analysis_io import AnalysisInputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--out-dir',required=True)
    args=parser.parse_args();bound=AnalysisInputs(args.config)
    membership=pd.read_csv(bound.files['membership'],sep='\t',dtype=str,keep_default_na=False)
    age=pd.read_csv(bound.files['age_annotations'],sep='\t',dtype=str,keep_default_na=False)
    keys=['gse_id','sample_id']
    if membership.duplicated(['cohort']+keys).any() or age.duplicated(keys).any():raise ValueError('Duplicate cohort/specimen or age keys')
    if not age.age_category.isin(['adult','paediatric','unknown']).all():raise ValueError('Use adult, paediatric or unknown age categories')
    # Ages/bands are source-curated. No imputation from the class, predictions or cohort name.
    joined=membership.merge(age[keys+['age_category']],on=keys,how='left',validate='many_to_one')
    joined.age_category=joined.age_category.fillna('unknown')
    out=bound.output_dir(args.out_dir)
    for label,group in [('cohort',['cohort']),('study_class',['cohort','gse_id','tcga_label'])]:
        joined.groupby(group+['age_category']).size().rename('n').reset_index().to_csv(out/(label+'_age_counts.tsv'),sep='\t',index=False)
    bound.finish(out,{'workflow':'age_representation','definition':'Paediatric <18 years; adult >=18; unresolved, conflicting and crossing age bands remain unknown','population':'Explicit cohort membership supplied with the input data'})


if __name__=='__main__':main()
