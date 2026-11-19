#!/usr/bin/env python3
"""Validate a bound input manifest before an analysis or cluster submission."""
import argparse,json
from minicpg.analysis_io import AnalysisInputs
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    a=p.parse_args();bound=AnalysisInputs(a.config)
    print(json.dumps({'status':'ok','verified_inputs':len(bound.files)}))
