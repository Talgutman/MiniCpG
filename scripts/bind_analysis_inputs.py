#!/usr/bin/env python3
"""Bind local permitted files to a JSON analysis template using SHA256 digests."""
import argparse,json,os
from pathlib import Path
from minicpg.analysis_io import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    p.add_argument('--file',action='append',default=[],metavar='KEY=PATH')
    args=p.parse_args();config=json.loads(Path(args.template).read_text());out=Path(args.output).resolve()
    if out.exists():raise ValueError('Refusing to overwrite a bound config')
    provided={}
    for item in args.file:
        key,path=item.split('=',1)
        if key in provided:raise ValueError(f'Duplicate input key: {key}')
        provided[key]=Path(path).resolve()
    expected=set(config['files'])
    if set(provided)!=expected:raise ValueError(f'Expected file keys {sorted(expected)}; received {sorted(provided)}')
    config['files']={key:{'path':os.path.relpath(path,out.parent),'sha256':sha256(path)} for key,path in provided.items()}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(config,indent=2)+'\n')


if __name__=='__main__':main()
