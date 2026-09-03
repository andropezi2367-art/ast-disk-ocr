from __future__ import annotations

import json

from ast_disk_ocr.core import load_catalog, select_catalog_candidate


def test_catalog_ignores_metadata_and_invalid_values(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps({"_meta": {"source": "test"}, "cip": ["5"], "bad": "x"}),
        encoding="utf-8",
    )
    assert load_catalog(path) == {"CIP": ["5"]}


def test_exact_high_confidence_candidate_is_accepted():
    result = select_catalog_candidate("cip", "5", 0.96, 0.90, {"CIP": ["5"]})
    assert result.accepted
    assert result.label == "CIP 5"
    assert result.status == "accepted"


def test_low_confidence_candidate_stays_in_review():
    result = select_catalog_candidate("CIP", "5", 0.30, 0.30, {"CIP": ["5"]})
    assert not result.accepted
    assert result.label == "CIP 5"
    assert result.status == "review"


def test_ambiguous_nearest_candidate_does_not_choose_a_code():
    result = select_catalog_candidate(
        "CA", "5", 0.99, 0.99, {"CIP": ["5"], "CTX": ["5"]}
    )
    assert not result.accepted
    assert result.code is None
    assert result.label is None


def test_one_edit_can_be_accepted_only_when_unique():
    result = select_catalog_candidate("C1P", "5", 0.99, 0.99, {"CIP": ["5"]})
    assert result.accepted
    assert result.label == "CIP 5"
    assert result.edit_distance == 1

