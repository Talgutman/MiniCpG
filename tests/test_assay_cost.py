"""Regression against manuscript tables, independent of calculator formulas."""
import csv
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('assay_cost', ROOT / 'scripts/calculate_assay_cost.py')
assay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assay)
CONFIG = json.loads((ROOT / 'config/assay_cost.json').read_text())


class AssayCostTests(unittest.TestCase):
    def test_s24_scenarios(self):
        ids = ['Lam_standalone_96', 'Lam_full_run_88patients', 'Lam_8patients',
               'Lam_24patients', 'Lam_96_three_pools', 'Lam_96_six_pools',
               'Lam_96_10patients_repeated_within_existing_runs',
               'Lam_96_third_full_run', 'Lam_96_adapter_full_well',
               'Lam_96_joint_more_work']
        definitions = {s['id']: s for s in CONFIG['scenarios']}
        with (ROOT / 'results/revision_20261119/Table_S27_03.tsv').open() as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        self.assertEqual(len(rows), len(ids))
        for name, row in zip(ids, rows):
            with self.subTest(scenario=name):
                s = definitions[name]
                result = assay.calculate(CONFIG, s['patients'], **s['changes'])
                self.assertEqual(f"{result['per_patient_usd']:.2f}", row['USD per patient'])
                for key, field in [('patients', 'Patients'), ('controls', 'Controls'), ('runs', 'Runs')]:
                    self.assertEqual(result[key], int(row[field]))

    def test_s24_component_costs(self):
        components = assay.calculate(CONFIG, 96)['components']
        rows = []
        for name in ('Table_S27_01.tsv', 'Table_S27_02.tsv'):
            with (ROOT / 'results/revision_20261119' / name).open() as f:
                rows.extend(csv.DictReader(f, delimiter='\t'))
        self.assertEqual(len(rows), len(components))
        for result, row in zip(components, rows):
            with self.subTest(component=result['component']):
                self.assertTrue(row['Component and source'].startswith(result['component']))
                self.assertEqual(f"${result['cost_usd']:,.2f}", row['Batch cost'])

    def test_impossible_capacity_fails(self):
        for changes in ({'barcodes': 8}, {'seq_usable': 0}, {'raw_pairs_per_run': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                assay.calculate(CONFIG, 96, **changes)

    def test_invalid_denominator_fails(self):
        for patients in (0, -1, 1.5):
            with self.subTest(patients=patients), self.assertRaises(ValueError):
                assay.calculate(CONFIG, patients)


if __name__ == '__main__':
    unittest.main()
