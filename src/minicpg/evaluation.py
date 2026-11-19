"""Fixed-class metrics and paired, truth-stratified patient-cluster resampling.

Ported from the adopted primary and overlapping-domain evaluations. Class axes,
probability availability, specimen ordering and stratum ordering remain explicit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from minicpg.analysis_io import AnalysisInputs

KEYS = ['gse_id', 'sample_id']
METRICS = ['macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy',
           'auroc_ovr_macro', 'auroc_available_class_macro']


def unique_keys(frame, name):
    if not set(KEYS) <= set(frame) or frame.duplicated(KEYS).any():
        raise ValueError(f'{name} needs unique gse_id/sample_id keys')


def auc_setup(y, frame, classes, axis):
    prepared, missing = [], []
    for j in axis:
        column = 'proba__'+classes[j]
        if column not in frame or (y == j).sum() in (0, len(y)):
            missing.append(classes[j])
            continue
        scores = frame[column].to_numpy(float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f'Invalid probabilities in {column}')
        order = np.argsort(scores, kind='stable')
        start = np.r_[0, np.flatnonzero(np.diff(scores[order]) != 0)+1]
        prepared.append((order, start, (y[order] == j).astype(float)))
    return prepared, missing


def metrics(y, pred, axis, weights, prepared_auc, nc, full_auc=True):
    cm = np.bincount(y*nc+pred, weights=weights, minlength=nc*nc).reshape(nc, nc)
    tp, support, predicted = cm.diagonal(), cm.sum(1), cm.sum(0)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(nc), where=support+predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(nc), where=support > 0)
    auc = []
    for order, start, positive in prepared_auc:
        pos = np.add.reduceat(weights[order]*positive, start)
        neg = np.add.reduceat(weights[order]*(1-positive), start)
        denominator = pos.sum()*neg.sum()
        if denominator > 0:
            auc.append(np.sum(pos*(np.cumsum(neg)-.5*neg))/denominator)
    available = float(np.mean(auc)) if auc else np.nan
    return np.array([f1[axis].mean(), np.sum(f1*support)/support.sum(),
                     tp.sum()/support.sum(), recall[support > 0].mean(),
                     available if full_auc else np.nan, available])


def cluster_structure(y, units, order):
    cluster, names = pd.factorize(np.asarray(units), sort=True)
    counts = np.bincount(cluster)
    groups = {}
    for i in range(len(names)):
        signature = tuple(np.unique(y[cluster == i]))
        groups.setdefault(signature, []).append(i)
    if order not in {'first_seen', 'sorted'}:
        raise ValueError('stratum_order must be first_seen or sorted')
    signatures = sorted(groups) if order == 'sorted' else list(groups)
    return cluster, 1/counts[cluster], [np.asarray(groups[s]) for s in signatures], len(names)


def evaluate_subset(data, units, classes, *, axis, seed, replicates=2000,
                    stratum_order='sorted', require_full_auc=False):
    if replicates < 0:
        raise ValueError('replicates cannot be negative')
    tags = list(data)
    first = data[tags[0]]
    if not len(first):
        raise ValueError('Cannot evaluate an empty subset')
    for tag, d in data.items():
        if not d[KEYS+['tcga_label']].equals(first[KEYS+['tcga_label']]):
            raise ValueError(f'Prediction membership/truth mismatch for {tag}')
    if len(units) != len(first) or pd.isna(units).any() or any(not str(u) for u in units):
        raise ValueError('Every selected specimen needs an explicit resampling unit')
    enc = {c: i for i, c in enumerate(classes)}
    try:
        y = np.asarray([enc[c] for c in first.tcga_label])
        pred = {t: np.asarray([enc[c] for c in d.y_pred]) for t, d in data.items()}
    except KeyError as exc:
        raise ValueError(f'Label outside declared class space: {exc}') from exc
    axis = np.unique(y) if axis == 'present' else np.array([enc[c] for c in axis])
    if not len(axis) or len(np.unique(axis)) != len(axis):
        raise ValueError('Metric axis must be nonempty and unique')
    cluster, balanced, strata, n_units = cluster_structure(y, units, stratum_order)
    ap, missing = {}, {}
    for tag, d in data.items():
        # Primary AUROC averages available truth classes, including absent-axis omissions.
        auc_axis = axis if require_full_auc else np.unique(y)
        ap[tag], missing[tag] = auc_setup(y, d, classes, auc_axis)
    point = np.empty((2, len(tags), len(METRICS)))
    for i, weights in enumerate([np.ones(len(y)), balanced]):
        for j, tag in enumerate(tags):
            point[i, j] = metrics(y, pred[tag], axis, weights, ap[tag], len(classes),
                                  not missing[tag] or not require_full_auc)
    rng = np.random.default_rng(seed)
    draws = np.zeros((replicates, n_units), dtype=np.int64)
    samples = np.empty((replicates, 2, len(tags), len(METRICS)))
    for b in range(replicates):
        for group in strata:
            draws[b, group] = rng.multinomial(len(group), np.full(len(group), 1/len(group)))
        for i, weights in enumerate([draws[b, cluster], draws[b, cluster]*balanced]):
            for j, tag in enumerate(tags):
                samples[b, i, j] = metrics(y, pred[tag], axis, weights, ap[tag], len(classes),
                                           not missing[tag] or not require_full_auc)
    return dict(point=point, samples=samples, cluster_multiplicities=draws,
                classes=[classes[i] for i in axis], models=tags, n=len(y), n_units=n_units,
                missing_probability_classes=missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    inputs = AnalysisInputs(args.config)
    cfg = inputs.config
    def read(key, **kwargs):
        return pd.read_csv(inputs.files[key], sep='\t', keep_default_na=False, **kwargs)
    classes = read('class_axis').tcga_label.tolist()
    if len(classes) != len(set(classes)):
        raise ValueError('Class axis has duplicate labels')
    master, inventory = read('canonical_master', dtype=str), read('canonical_inventory', dtype=str)
    coverage, units = read('coverage', dtype=str), read('patient_units', dtype=str)
    for name, table in [('master', master), ('inventory', inventory), ('units', units)]:
        unique_keys(table, name)
    predictions = {tag: read(tag, float_precision=cfg.get('float_precision', 'round_trip'))
                   for tag in cfg['prediction_inputs']}
    for tag, table in predictions.items():
        unique_keys(table, tag)
    all_points, intervals, per_class, confusion = [], [], [], []
    results = []
    for job in cfg['evaluations']:
        if job['axis'] not in {'all','shared_fixed','present'}:
            raise ValueError('axis must be all, shared_fixed or present')
        if not job['name'].replace('_','').isalnum():
            raise ValueError('Evaluation names must be alphanumeric/underscore')
        selected = coverage.loc[coverage[job['cohort_field']].eq(job['cohort_value']) &
                                coverage[job['eligibility_field']].eq('1'), KEYS].copy()
        if job.get('sort_keys', True):
            selected = selected.sort_values(KEYS)
        selected = selected.reset_index(drop=True)
        unique_keys(selected, job['name'])
        truth = selected.merge(master, on=KEYS, how='left', validate='one_to_one')
        if len(truth) != len(selected) or truth.tcga_label.isna().any():
            raise ValueError('Selected specimens are absent from canonical master')
        checked = selected.merge(inventory[KEYS], on=KEYS, how='inner', validate='one_to_one')
        if len(checked) != len(selected):
            raise ValueError('Selected specimens are absent from source inventory')
        for field, expected in job['canonical_filters'].items():
            if not truth[field].eq(expected).all():
                raise ValueError(f'Canonical cohort eligibility mismatch: {field}')
        data = {}
        for tag in job['models']:
            d = selected.merge(predictions[tag], on=KEYS, how='left', validate='one_to_one')
            if not d.tcga_label.equals(truth.tcga_label) or d.y_pred.isna().any():
                raise ValueError(f'Missing predictions or canonical truth mismatch: {tag}')
            data[tag] = d
        if job.get('shared', False) and truth.tcga_label.eq('TCGA-ESCA').any():
            raise ValueError('ESCA/STAD truth cannot enter the shared comparison')
        if job.get('runnerup_sensitivity', False):
            d = data['minicpg'].copy()
            cols = [c for c in d if c.startswith('proba__') and c != 'proba__TCGA-ESCA']
            take = d.y_pred.eq('TCGA-ESCA')
            d.loc[take, 'y_pred'] = d.loc[take, cols].idxmax(axis=1).str.removeprefix('proba__')
            data['minicpg_runnerup'] = d
        u = selected.merge(units, on=KEYS, how='left', validate='one_to_one')
        axis = classes if job['axis'] == 'all' else [c for c in classes if c != 'TCGA-ESCA'] if job['axis'] == 'shared_fixed' else 'present'
        result = evaluate_subset(data, u.resampling_unit.to_numpy(), classes, axis=axis,
            seed=job['seed'], replicates=cfg.get('bootstrap_replicates', 2000),
            stratum_order=job['stratum_order'], require_full_auc=job['require_full_auc'])
        tags = result['models']
        for w, weighting in enumerate(['specimen', 'patient_balanced']):
            contrasts = [(tag, result['point'][w,j], result['samples'][:,w,j]) for j,tag in enumerate(tags)]
            if 'methydeep' in tags and 'minicpg' in tags:
                for tag in [t for t in tags if t.startswith('minicpg')]:
                    i,j=tags.index(tag),tags.index('methydeep')
                    contrasts.append((tag+'_minus_methydeep',result['point'][w,i]-result['point'][w,j],result['samples'][:,w,i]-result['samples'][:,w,j]))
            for tag, values, draws in contrasts:
                row=dict(evaluation=job['name'],model=tag,weighting=weighting,n=result['n'],n_units=result['n_units'],**dict(zip(METRICS,values)))
                all_points.append(row)
                for k, metric in enumerate(METRICS):
                    finite=draws[:,k][np.isfinite(draws[:,k])]
                    low,high=np.percentile(finite,[2.5,97.5]) if len(finite) else (np.nan,np.nan)
                    intervals.append(dict(evaluation=job['name'],comparison=tag,weighting=weighting,metric=metric,point=values[k],ci_low=low,ci_high=high,finite_replicates=len(finite),seed=job['seed']))
        for tag,d in data.items():
            for c in classes:
                support=int(d.tcga_label.eq(c).sum());pn=int(d.y_pred.eq(c).sum());tp=int((d.tcga_label.eq(c)&d.y_pred.eq(c)).sum())
                per_class.append(dict(evaluation=job['name'],model=tag,tcga_label=c,support=support,predicted_n=pn,true_positive=tp,recall=tp/support if support else 0,precision=tp/pn if pn else 0,f1=2*tp/(support+pn) if support+pn else 0))
            for (actual,predicted),n in d.groupby(['tcga_label','y_pred']).size().items():
                confusion.append(dict(evaluation=job['name'],model=tag,truth=actual,prediction=predicted,n=int(n)))
        results.append((job['name'], result))
    out=inputs.output_dir(args.out_dir)
    for name, rows in [('metrics',all_points),('uncertainty',intervals),('per_class',per_class),('confusion',confusion)]:
        pd.DataFrame(rows).to_csv(out/(name+'.tsv'),sep='\t',index=False,float_format='%.17g')
    for name,r in results:
        if not name.replace('_','').isalnum():raise ValueError('Evaluation names must be alphanumeric/underscore')
        np.savez_compressed(out/(name+'_bootstrap.npz'),samples=r['samples'],cluster_multiplicities=r['cluster_multiplicities'],models=r['models'],metrics=METRICS)
    inputs.finish(out, {'workflow':'external_evaluation','evaluations':[dict(name=n,n=r['n'],units=r['n_units'],class_axis=r['classes'],missing_probability_classes=r['missing_probability_classes']) for n,r in results]})


if __name__ == '__main__':
    main()
