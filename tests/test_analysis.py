"""Synthetic checks against independent metric/permutation calculations."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from minicpg.analysis_io import AnalysisInputs, BoundStage, sha256
from minicpg.evaluation import evaluate_subset
from minicpg.geometry import class_profiles, permutation_test

ROOT = Path(__file__).resolve().parents[1]


def fixture_predictions():
    return pd.DataFrame({
        'gse_id': ['synthetic'] * 6, 'sample_id': list('abcdef'),
        'tcga_label': ['A', 'A', 'A', 'B', 'B', 'B'],
        'y_pred': ['A', 'A', 'B', 'B', 'C', 'A'],
        'proba__A': [.8, .7, .2, .1, .1, .6],
        'proba__B': [.1, .2, .7, .8, .3, .3],
        'proba__C': [.1, .1, .1, .1, .6, .1]})


@pytest.mark.parametrize('balanced', [False, True])
def test_metrics_against_sklearn(balanced):
    d = fixture_predictions()
    units = np.array(['u1', 'u1', 'u2', 'u3', 'u4', 'u5'])
    r = evaluate_subset({'one': d}, units, list('ABC'), axis=list('ABC'), seed=7, replicates=12)
    w = np.array([.5, .5, 1, 1, 1, 1]) if balanced else np.ones(6)
    got = r['point'][int(balanced), 0]
    assert got[0] == pytest.approx(f1_score(d.tcga_label, d.y_pred, labels=list('ABC'), average='macro', sample_weight=w))
    assert got[1] == pytest.approx(f1_score(d.tcga_label, d.y_pred, average='weighted', sample_weight=w))
    assert got[2] == pytest.approx(accuracy_score(d.tcga_label, d.y_pred, sample_weight=w))
    auc = np.mean([roc_auc_score(d.tcga_label.eq(c), d['proba__'+c], sample_weight=w) for c in 'AB'])
    assert got[4] == pytest.approx(auc)


def test_joint_draws_preserve_clusters_and_truth_strata():
    d = fixture_predictions()
    r = evaluate_subset({'one': d, 'identical': d.copy()}, ['u1','u1','u2','u3','u4','u5'], list('ABC'), axis='present', seed=11, replicates=20)
    np.testing.assert_equal(r['samples'][:, :, 0], r['samples'][:, :, 1])
    assert (r['cluster_multiplicities'][:, :2].sum(1) == 2).all()
    assert (r['cluster_multiplicities'][:, 2:].sum(1) == 3).all()
    fixed = evaluate_subset({'one': d}, list('abcdef'), list('ABC'), axis=list('ABC'), seed=11, replicates=0, require_full_auc=True)
    assert np.isnan(fixed['point'][0, 0, 4])
    assert np.isfinite(fixed['point'][0, 0, 5])
    assert fixed['missing_probability_classes']['one'] == ['C']


def test_mismatched_truth_fails_before_metrics():
    d = fixture_predictions(); other = d.copy(); other.loc[0, 'tcga_label'] = 'B'
    with pytest.raises(ValueError, match='membership/truth'):
        evaluate_subset({'one': d, 'two': other}, list('abcdef'), list('ABC'), axis='present', seed=1)


def test_class_permutation_matches_direct_scipy():
    similarity = np.array([.8, .1, .4, .3, .8, .2])
    confusion = np.array([.2, 0, .1, .2, .4, 0])
    got, null = permutation_test(similarity, confusion, 4, permutations=31, seed=8)
    matrix = np.zeros((4, 4)); i, j = np.triu_indices(4, 1)
    matrix[i, j] = matrix[j, i] = similarity
    rng = np.random.default_rng(8)
    expected = []
    for _ in range(31):
        p = rng.permutation(4)
        expected.append(spearmanr(matrix[p[:, None], p][i, j], confusion).statistic)
    np.testing.assert_allclose(null, expected, atol=1e-14)
    assert got['observed_spearman'] == pytest.approx(spearmanr(similarity, confusion).statistic)


def test_profiles_use_stored_training_medians():
    model = SimpleNamespace(named_steps={'imputer': SimpleNamespace(statistics_=np.array([.2, .8]))})
    d = pd.DataFrame({'tcga_label': ['A','A','B'], 'cg1': [0, np.nan, 1], 'cg2': [2, .4, np.nan]})
    got = class_profiles(d, {'probe_ids': ['cg1','cg2'], 'model': model})
    np.testing.assert_allclose(got.to_numpy(), [[.1, .6], [1, .8]])


def bind(tmp_path, files, **extra):
    config = tmp_path/'bound.json'
    config.write_text(json.dumps({'files': {k: {'path': str(p), 'sha256': sha256(p)} for k,p in files.items()}, **extra}))
    return config


def test_hash_contract_detects_tampering_and_output_escape(tmp_path):
    p = tmp_path/'synthetic.tsv'; p.write_text('x\n1\n')
    config = bind(tmp_path, {'data': p}, outputs={'bad': '../escape.tsv'})
    bound = AnalysisInputs(config)
    with pytest.raises(ValueError, match='empty'):
        bound.output_dir(tmp_path)
    stage = BoundStage(config, tmp_path/'out')
    with pytest.raises(ValueError, match='inside'):
        stage.output('bad')
    p.write_text('x\n2\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        AnalysisInputs(config)
    with pytest.raises(RuntimeError, match='changed'):
        bound.finish(tmp_path/'out', {})


def run(script, config, out):
    subprocess.run([sys.executable, str(ROOT/'scripts'/script), '--config', str(config), '--out-dir', str(out)], check=True, capture_output=True, text=True)


def test_age_aggregation_keeps_unknown_and_overlapping_membership(tmp_path):
    members = tmp_path/'members.tsv'
    pd.DataFrame({'cohort':['FFPE','EPIC','EPIC'], 'gse_id':['synthetic']*3, 'sample_id':['a','a','b'], 'tcga_label':['A']*3}).to_csv(members, sep='\t', index=False)
    ages = tmp_path/'ages.tsv'
    pd.DataFrame({'gse_id':['synthetic'], 'sample_id':['a'], 'age_category':['adult']}).to_csv(ages, sep='\t', index=False)
    cfg = bind(tmp_path, {'membership':members, 'age_annotations':ages})
    run('summarize_age_representation.py', cfg, tmp_path/'out')
    counts = pd.read_csv(tmp_path/'out/cohort_age_counts.tsv', sep='\t')
    assert counts.n.sum() == 3
    assert counts.loc[counts.age_category.eq('unknown'), 'n'].sum() == 1


def test_study_recall_aggregation(tmp_path):
    pred = fixture_predictions(); path = tmp_path/'pred.tsv'; pred.to_csv(path, sep='\t', index=False)
    cfg = bind(tmp_path, {'predictions':path, 'canonical_master':path, 'canonical_inventory':path})
    run('summarize_study_errors.py', cfg, tmp_path/'out')
    result = pd.read_csv(tmp_path/'out/study_class_recall.tsv', sep='\t').set_index('tcga_label')
    assert result.loc['A','recall'] == pytest.approx(2/3)
    assert result.loc['B','errors'] == 2


def test_patient_split_guards():
    sys.path.insert(0, str(ROOT/'scripts'))
    from split_integrity import validate_partition, validate_folds
    dev = [{'sample_id':'synthetic1','patient_id':'person1'}, {'sample_id':'synthetic2','patient_id':'person1'}]
    with pytest.raises(ValueError, match='patient overlap'):
        validate_partition(dev, [{'sample_id':'synthetic3','patient_id':'person1'}])
    with pytest.raises(ValueError, match='patient occurs'):
        validate_folds(dev, [{'sample_id':'synthetic1','outer_fold':0}, {'sample_id':'synthetic2','outer_fold':1}])


def test_documented_synthetic_cli_example(tmp_path):
    subprocess.run([sys.executable, str(ROOT/'examples/run_synthetic_evaluation.py'), '--out-dir', str(tmp_path/'example')], check=True, capture_output=True)
    got = pd.read_csv(tmp_path/'example/results/metrics.tsv', sep='\t')
    assert got.n.eq(6).all() and got.n_units.eq(5).all()
    assert (tmp_path/'example/results/provenance.json').is_file()


def test_canonical_inference_preserves_coverage_and_schema(tmp_path):
    from minicpg.inference import load_model_bundle, predict_beta_table
    model = ROOT/'model/minicpg_k30_model_bundle.pkl.gz'
    bundle = load_model_bundle(model)
    matrix = pd.DataFrame(.5, index=range(2), columns=bundle['probe_ids'])
    matrix.loc[1, bundle['probe_ids'][:6]] = np.nan
    matrix['gse_id'] = 'synthetic'; matrix['sample_id'] = ['a','b']
    matrix['tcga_label'] = bundle['class_order'][0]
    canonical = matrix[['gse_id','sample_id','tcga_label']].copy()
    canonical['include'] = 1; canonical['target_bucket'] = 'main'
    mpath = tmp_path/'matrix.tsv'; matrix.to_csv(mpath, sep='\t', index=False)
    cpath = tmp_path/'canonical.tsv'; canonical.to_csv(cpath, sep='\t', index=False)
    cfg = bind(tmp_path, {'model':model,'matrix':mpath,'canonical_master':cpath,'canonical_inventory':cpath}, canonical_filters={'include':'1','target_bucket':'main'})
    run('predict_external_cohort.py', cfg, tmp_path/'out')
    pred = pd.read_csv(tmp_path/'out/predictions.tsv', sep='\t')
    cov = pd.read_csv(tmp_path/'out/coverage.tsv', sep='\t')
    expected = predict_beta_table(matrix, bundle, include_probabilities=True)
    assert pred.sample_id.tolist() == ['a']
    assert pred.y_pred.iloc[0] == expected.predicted_label.iloc[0]
    assert cov.n_measured_probes.tolist() == [30,24]
    assert len([c for c in pred if c.startswith('proba__')]) == len(bundle['class_order'])


def test_fixed_panel_training_and_generic_bundle_prediction(tmp_path):
    """A tiny synthetic fit checks the training CLI and development-only medians."""
    import pickle
    dev_ids = ['synthetic_dev_'+str(i) for i in range(8)]
    held_ids = ['synthetic_held_'+str(i) for i in range(2)]
    labels = ['TCGA-A']*4+['TCGA-B']*4
    for name, ids, truth in [('dev',dev_ids,labels),('held',held_ids,['TCGA-A','TCGA-B'])]:
        pd.DataFrame({'sample_id':ids,'patient_id':ids,'project_id':truth}).to_csv(tmp_path/(name+'.tsv'),sep='\t',index=False)
    beta = pd.DataFrame({'probe_id':['cg1','cg2'], **{sid:[.1,.2] if i<4 else [.8,.9] for i,sid in enumerate(dev_ids)}, held_ids[0]:[.1,.2], held_ids[1]:[.8,.9]})
    beta.to_parquet(tmp_path/'beta.parquet',index=False)
    pd.DataFrame({'probe_id':['cg1','cg2']}).to_csv(tmp_path/'panel.tsv',sep='\t',index=False)
    args = ['--probe-matrix',str(tmp_path/'beta.parquet'),'--dev-manifest',str(tmp_path/'dev.tsv'),'--lockbox-manifest',str(tmp_path/'held.tsv'),'--panel-tsv',str(tmp_path/'panel.tsv'),'--k','2','--out-dir',str(tmp_path/'fit'),'--model','extra_trees','--class-weight-mode','balanced','--tune-metric','macro_f1','--inner-folds','2','--et-n-estimators-grid','5','--et-max-features-grid','1.0','--et-max-depth-grid','0','--et-min-samples-leaf-grid','1','--n-jobs','1']
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_lockbox_from_panel.py'),*args],check=True,capture_output=True,text=True)
    with (tmp_path/'fit/model_bundle.pkl').open('rb') as f:
        bundle=pickle.load(f)
    assert bundle['decision_policy'] is None
    np.testing.assert_allclose(bundle['model'].named_steps['imputer'].statistics_,[.45,.55])
    external=pd.DataFrame({'gse_id':['synthetic']*2,'sample_id':['a','b'],'tcga_label':['TCGA-A','TCGA-B'],'target_bucket':['main']*2,'cg1':[.1,.8],'cg2':[.2,.9]})
    external.to_csv(tmp_path/'external.tsv',sep='\t',index=False)
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_external_from_bundle.py'),'--panel-matrix',str(tmp_path/'external.tsv'),'--model-bundle',str(tmp_path/'fit/model_bundle.pkl'),'--out-dir',str(tmp_path/'external')],check=True,capture_output=True,text=True)
    got=pd.read_csv(tmp_path/'external/external_predictions_allow_missing1.tsv',sep='\t')
    assert got.y_pred.tolist()==external.tcga_label.tolist()
