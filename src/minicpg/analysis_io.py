"""Explicit, hash-bound analysis inputs and output provenance."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class AnalysisInputs:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text())
        self.files = {}
        for name, record in self.config['files'].items():
            path = (self.config_path.parent / record['path']).resolve()
            expected = record.get('sha256', '')
            if len(expected) != 64:
                raise ValueError(f'Input {name} requires a SHA256 digest')
            if not path.is_file():
                raise FileNotFoundError(f'Missing input {name}: {path}')
            if sha256(path) != expected:
                raise ValueError(f'Input hash mismatch: {name}')
            self.files[name] = path

    def output_dir(self, path):
        out = Path(path).resolve()
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'Output directory must be empty: {out}')
        if any(p == out or out in p.parents for p in self.files.values()):
            raise ValueError('Output directory must not contain input files')
        out.mkdir(parents=True, exist_ok=True)
        return out

    def finish(self, out, details):
        for name, path in self.files.items():
            if sha256(path) != self.config['files'][name]['sha256']:
                raise RuntimeError(f'Input changed during analysis: {name}')
        record = {'config_sha256': sha256(self.config_path),
                  'input_sha256': {k: v['sha256'] for k, v in self.config['files'].items()},
                  'output_sha256': {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
                  **details}
        (out / 'provenance.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


class BoundStage:
    """Small adapter for analysis scripts with named input/output contracts."""
    def __init__(self, config_path, output_dir):
        self.bound = AnalysisInputs(config_path)
        self.out = self.bound.output_dir(output_dir)
        self.config = self.bound.config

    def input(self, name):
        return self.bound.files[name]

    def output(self, name):
        filename = self.config.get('outputs', {}).get(name, name+'.tsv')
        path = (self.out / filename).resolve()
        if not path.is_relative_to(self.out) or path == self.out:
            raise ValueError('Outputs must remain inside the analysis directory')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def finish(self, details):
        self.bound.finish(self.out, {'workflow': self.config.get('workflow'), 'details': details})
