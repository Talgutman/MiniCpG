"""Run the hash-bound external evaluation CLI on explicitly synthetic data."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from minicpg.analysis_io import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new/empty example directory')
    inputs = out/'inputs'; inputs.mkdir(parents=True)
    keys = {'gse_id': ['synthetic']*6, 'sample_id': ['synthetic_'+str(i) for i in range(6)]}
    truth = ['TCGA-A','TCGA-A','TCGA-A','TCGA-B','TCGA-B','TCGA-B']
    frames = {
        'class_axis': pd.DataFrame({'tcga_label':['TCGA-A','TCGA-B','TCGA-C']}),
        'canonical_master': pd.DataFrame({**keys,'tcga_label':truth,'include':'1','target_bucket':'main'}),
        'coverage': pd.DataFrame({**keys,'target_bucket':'main','minicpg_eligible':'1'}),
        'patient_units': pd.DataFrame({**keys,'resampling_unit':['u1','u1','u2','u3','u4','u5']}),
        'minicpg': pd.DataFrame({**keys,'tcga_label':truth,'y_pred':['TCGA-A','TCGA-A','TCGA-B','TCGA-B','TCGA-C','TCGA-A'],
            'proba__TCGA-A':[.8,.7,.2,.1,.1,.6], 'proba__TCGA-B':[.1,.2,.7,.8,.3,.3], 'proba__TCGA-C':[.1,.1,.1,.1,.6,.1]})}
    frames['canonical_inventory'] = frames['canonical_master'].copy()
    config = json.loads((root/'config/workflows/primary.json').read_text())
    config['evaluations'] = config['evaluations'][:1]
    config['prediction_inputs'] = ['minicpg']
    config['bootstrap_replicates'] = 32
    config['files'] = {}
    for name, data in frames.items():
        path = inputs/(name+'.tsv'); data.to_csv(path, sep='\t', index=False)
        config['files'][name] = {'path':'inputs/'+path.name, 'sha256':sha256(path)}
    cfg = out/'evaluation.json'; cfg.write_text(json.dumps(config, indent=2)+'\n')
    subprocess.run([sys.executable, str(root/'scripts/evaluate_external.py'), '--config', str(cfg), '--out-dir', str(out/'results')], check=True)
    print('Synthetic demonstration only; results written to', out/'results')


if __name__ == '__main__':
    main()
