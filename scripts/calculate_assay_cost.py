"""Recalculate the adopted Lam-based research operating budget in SM8/Table S27."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def calculate(config, patients, **overrides):
    """Cost a conditional batch; controls and repeats do not enter its denominator."""
    q = config['parameters'] | overrides
    prices = config['unit_prices']
    if patients <= 0 or int(patients) != patients:
        raise ValueError('patients must be a positive integer')
    controls = len(q['methylation_levels']) + q['blank_controls']
    if q['barcodes'] <= controls:
        raise ValueError('barcode capacity must exceed controls per run')
    for key in ('depth', 'seq_usable', 'weakest', 'raw_pairs_per_run'):
        if q[key] <= 0:
            raise ValueError(f'{key} must be positive')
    runs = q['runs'] or max(1, math.ceil((patients + q['repeats']) / (q['barcodes'] - controls)))
    raw_pairs = q['targets'] * q['depth'] / (q['seq_usable'] * q['weakest'])
    if raw_pairs * controls >= q['raw_pairs_per_run']:
        raise ValueError('sequencing capacity must exceed the control allocation')
    while True:
        libraries = patients + q['repeats'] + controls * runs
        needed = max(math.ceil(libraries / q['barcodes']),
                     math.ceil(libraries * raw_pairs / q['raw_pairs_per_run']))
        if needed <= runs:
            break
        runs = needed
    rows = []
    def add(name, quantity, unit, basis):
        rows.append({'component': name, 'quantity': quantity, 'unit_usd': unit,
                     'cost_usd': quantity * unit, 'basis': basis,
                     'sources': config['component_sources'][name]})
    over = 1 + q['reagent_overage']
    pools = q['pools']
    rx = libraries * pools * q['replicates']
    add('Lightning conversion', math.ceil(libraries / q['conversion_plate_capacity']) * q['conversion_plate_capacity'] if q['whole_conversion'] else libraries,
        prices['conversion_position'], 'Whole plate capacity charged; no reuse of empty wells assumed')
    add('GoTaq target and library PCR', over * (rx * q['target_enzyme_units'] + libraries * q['library_enzyme_units']),
        prices['enzyme_unit'], 'Lam target and library PCR volumes; buffer/Mg supplied')
    add('dNTPs', over * (rx * q['target_dntp_ul'] + libraries * q['library_dntp_ul']),
        prices['dntp_ul'], '10 mM each stock; Lam volumes')
    add('Target and universal PCR oligos', over * (rx * q['target_primer_pmol'] * q['target_primer_bases'] + libraries * q['library_primer_pmol'] * q['library_primer_bases']),
        prices['oligo_base_pmol'], 'Pooled stock concentration, lengths and delivered yield are assumptions')
    add('Primer phosphorylation PNK', math.ceil(over * rx * q['primer_stock_ul_per_rxn'] / q['pnk_batch_ul']) * q['pnk_units_per_batch'],
        prices['pnk_unit'], 'Phosphorylation stock batches rounded up')
    add('UltraII ligation mix and enhancer', over * libraries * q['ligation_mix_ul'] * q['ligation_added_ul'] / q['ligation_master_ul'],
        prices['ligation_mix_ul'], 'Small aliquot from Lam supplementary recipe; enhancer included')
    add('Indexed ligation adapters', libraries, q['adapter_price'],
        'Monetary proxy scaled to dilute Lam dose; compatibility and aliquot reuse unvalidated')
    beads = over * (libraries * (pools * q['pcr_ul'] * q['replicates'] * q['target_bead_ratio'] + q['library_cleanup_beads_ul']) + runs * q['pool_cleanup_beads_ul'])
    add('Cleanup beads', beads, prices['beads_ul'], 'Triplicates pooled before target cleanup')
    qc = (q['representative_libraries_per_run'] + 1) * runs
    add('Representative TapeStation QC', max(q['qc_minimum'], qc), prices['qc_sample'],
        'Representative libraries and run pools; not every library sized')
    qpcr = q['qpcr_per_run'] * runs
    tips = libraries * (8 + 3 * pools * q['replicates'] + 14 + 7 + 5) + qpcr * 2 + 80 * runs
    hours = {'controls_records': 1, 'conversion': libraries / 64,
             'target_setup': rx / 240, 'pool_clean': libraries / 80,
             'ligation_library_setup_cleanup': libraries / 40,
             'quantification_pooling': libraries / 80, 'QC_submission': .5}
    plates = math.ceil(rx / 96) + 5 * math.ceil(libraries / 96) + math.ceil(qpcr / 96)
    qubit = patients + q['repeats'] + libraries + 3 * runs
    add('Qubit assays and tubes', qubit, prices['qubit_with_tube'], 'Input DNA, final libraries and run pools/standards')
    add('qPCR quantification', qpcr * over, prices['qpcr'], 'Pool-only quantification; contemporary NEB price proxy for KAPA')
    add('Filter tips', math.ceil(tips * q['tip_factor']), prices['tip'], 'Task-based handling allowance')
    add('PCR plates and seals', plates, prices['plate_with_seal'], 'Whole plates rounded; conversion plastics in conversion price')
    add('Storage tubes', 2 * libraries + 20 * runs, prices['tube'], 'Qubit tubes excluded here')
    add('Minor buffers and chemicals', libraries, prices['minor_buffers'], 'Explicit unsourced operating allowance')
    add('Methylation control DNA', len(q['methylation_levels']) * runs * q['control_input_ng'] * over,
        prices['control_ng'], 'Symmetric standards; process blank contains no DNA')
    add('PhiX', q['phix_ul_per_run'] * runs, prices['phix_ul'], 'Exclude if sequencing invoice already includes this')
    add('Sequencing service', runs, q['run_fee'], 'Full MiSeq v2 run; no additional operator labour')
    add('Demultiplexing', runs, prices['demultiplexing'], 'Exclude if included in sequencing fee')
    add('Hands-on bench labour', sum(hours.values()) * q['labour_factor'], prices['bench_hour'],
        'Assumed task durations; excludes purchased QC/sequencing services')
    add('Basic analysis labour', q['analysis_hours'] * q['labour_factor'], prices['analysis_hour'],
        'Established pipeline execution/QC/export; development excluded')
    add('Compute and storage', 1, prices['compute_batch'], 'Batch allowance, not a cloud quote')
    total = sum(row['cost_usd'] for row in rows)
    return {'workflow': 'Lam', 'patients': patients, 'repeats': q['repeats'],
            'controls': controls * runs, 'libraries': libraries, 'runs': runs,
            'target_pcrs': rx, 'raw_pairs_per_library': raw_pairs,
            'total_raw_pairs': libraries * raw_pairs, 'hours': hours,
            'components': rows, 'total_usd': total, 'per_patient_usd': total / patients,
            'epic_ratio': q['epic'] / (total / patients)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/assay_cost.json')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        parser.error(f'Cost configuration not found: {config_path}')
    config = json.loads(config_path.read_text())
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        parser.error(f'Output directory must be empty: {out}')
    scenarios = {row['id']: calculate(config, row['patients'], **row.get('changes', {}))
                 for row in config['scenarios']}
    result = {'model': config['model'], 'parameters': config['parameters'],
              'unit_prices': config['unit_prices'], 'references': config['references'],
              'interpretation': config['interpretation'], 'source_date': config['source_date'],
              'assumption_notes': config['assumption_notes'],
              'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'scenarios': scenarios}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cost_model.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'components.tsv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['scenario', 'patients'] + list(next(iter(scenarios.values()))['components'][0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for name, batch in scenarios.items():
            for row in batch['components']:
                writer.writerow({'scenario': name, 'patients': batch['patients'], **row})
    with (out / 'scenario_summary.tsv').open('w') as handle:
        fields = ['scenario', 'patients', 'controls', 'libraries', 'runs', 'total_usd', 'per_patient_usd']
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for name, batch in scenarios.items():
            writer.writerow({'scenario': name, **{key: batch[key] for key in fields[1:]}})


if __name__ == '__main__':
    main()
