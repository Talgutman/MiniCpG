from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_external_panel_matrix import _normalize_panel_value
from annotate_panel_chromatin_context import load_probes, annotate_ccre


@pytest.mark.parametrize("value", ["NA", "nan", "N/A", "null", ".", "inf", "-inf", ""])
def test_missing_measurements_remain_missing(value):
    assert _normalize_panel_value(value) == ""


def test_extractor_preserves_precision_and_rejects_unknown_strings():
    assert _normalize_panel_value("0.123456789123") == "0.123456789123"
    with pytest.raises(ValueError, match="nonnumeric"):
        _normalize_panel_value("unexpected text")


def test_ccre_half_open_boundary(tmp_path):
    path = tmp_path / "probes.tsv"
    pd.DataFrame([dict(probe_id="synthetic", panel_rank=1, CpG_chrm="chr1",
        CpG_beg=100, CpG_end=102, gene_list="example", final_biological_label="example")]).to_csv(path, sep="\t", index=False)
    probes = load_probes(path)
    bed = tmp_path / "GRCh38-cCREs.PLS.bed"
    bed.write_text("chr1\t99\t100\tbefore\te\tPLS\nchr1\t100\t102\texact\te\tPLS\nchr1\t102\t103\tafter\te\tPLS\n")
    hits = annotate_ccre(probes, [bed])
    assert hits.ccre_accession_d.tolist() == ["exact"]
    assert probes.bed_start.iloc[0] == 100
