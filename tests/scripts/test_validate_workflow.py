"""Regression tests for scripts/validate_workflow.py.

Anchored by PR #170 review: the validator must report problems rather than
raise on malformed-but-valid JSON. These tests pin that behavior plus a
sampling of the existing happy-path and consistency checks so future edits
cannot silently regress the safety guarantees.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import pytest

# Load the script directly — it lives under scripts/, not under a package.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_workflow.py"
_spec = importlib.util.spec_from_file_location("validate_workflow", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate = _module.validate
main = _module.main


def _minimal_clean() -> dict:
    """Smallest workflow that should validate cleanly: two nodes, one link."""
    return {
        "last_node_id": 2,
        "last_link_id": 1,
        "nodes": [
            {
                "id": 1,
                "type": "Src",
                "inputs": [],
                "outputs": [
                    {"name": "out", "type": "FOO", "links": [1]},
                ],
            },
            {
                "id": 2,
                "type": "Dst",
                "inputs": [
                    {"name": "in", "type": "FOO", "link": 1},
                ],
                "outputs": [],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "FOO"]],
    }


# --- Happy path -------------------------------------------------------------

def test_minimal_clean_workflow_has_no_problems() -> None:
    assert validate(_minimal_clean()) == []


# --- PR #170 review repros: malformed-but-valid JSON must not crash --------

@pytest.mark.parametrize("data", [None, 42, 3.14, "string", [1, 2], (1, 2)])
def test_top_level_not_object_reports_does_not_crash(data) -> None:
    problems = validate(data)
    assert problems, "expected at least one problem"
    assert "top-level JSON must be an object" in problems[0]


def test_node_entry_not_dict_reports_does_not_crash() -> None:
    # Reviewer's repro 1: nodes contains an int.
    data = {"nodes": [1], "links": [], "last_node_id": 0, "last_link_id": 0}
    problems = validate(data)
    assert any("nodes[0]" in p and "object" in p for p in problems), problems


def test_nodes_field_not_list_reports() -> None:
    data = {"nodes": {"oops": "dict not list"}, "links": []}
    problems = validate(data)
    assert any("'nodes' must be a list" in p for p in problems), problems


def test_links_field_not_list_reports() -> None:
    data = {"nodes": [], "links": "not-a-list"}
    problems = validate(data)
    assert any("'links' must be a list" in p for p in problems), problems


def test_inputs_field_not_list_reports() -> None:
    data = _minimal_clean()
    data["nodes"][1]["inputs"] = {"oops": "dict not list"}
    problems = validate(data)
    assert any("'inputs' must be a list" in p for p in problems), problems


def test_input_entry_not_dict_reports() -> None:
    data = _minimal_clean()
    data["nodes"][1]["inputs"] = ["string-not-dict"]
    problems = validate(data)
    assert any("inputs[0]" in p and "object" in p for p in problems), problems


def test_output_entry_not_dict_reports() -> None:
    data = _minimal_clean()
    data["nodes"][0]["outputs"] = [99]
    problems = validate(data)
    assert any("outputs[0]" in p and "object" in p for p in problems), problems


def test_malformed_link_not_list_reports() -> None:
    data = _minimal_clean()
    data["links"].append("not-a-list")
    problems = validate(data)
    assert any("links[1]" in p for p in problems), problems


def test_malformed_link_wrong_length_reports() -> None:
    data = _minimal_clean()
    data["links"].append([2, 1, 0])  # too short
    problems = validate(data)
    assert any("links[1]" in p and "6 elements" in p for p in problems), problems


# --- Existing consistency checks -------------------------------------------

def test_link_to_missing_src_node_reports() -> None:
    data = _minimal_clean()
    data["links"] = [[1, 99, 0, 2, 0, "FOO"]]
    data["nodes"][0]["outputs"][0]["links"] = []  # not referenced from anywhere now
    problems = validate(data)
    assert any("src node 99 missing" in p for p in problems), problems


def test_type_mismatch_on_slot_reports() -> None:
    data = _minimal_clean()
    data["links"] = [[1, 1, 0, 2, 0, "BAR"]]  # link declares BAR, slots are FOO
    problems = validate(data)
    assert any("type mismatch" in p for p in problems), problems


def test_duplicate_node_ids_report() -> None:
    data = _minimal_clean()
    data["nodes"][1]["id"] = 1  # collision with first node
    problems = validate(data)
    assert any("Duplicate node IDs" in p for p in problems), problems


def test_endpoint_ref_desync_reports() -> None:
    data = _minimal_clean()
    data["nodes"][0]["outputs"][0]["links"] = []  # source side forgot link 1
    problems = validate(data)
    assert any("missing from node 1 output[0]" in p for p in problems), problems


def test_last_node_id_too_low_reports() -> None:
    data = _minimal_clean()
    data["last_node_id"] = 0  # but max is 2
    problems = validate(data)
    assert any("last_node_id" in p for p in problems), problems


# --- CLI smoke (covers main() exit codes) ----------------------------------

def test_cli_returns_zero_on_clean_workflow(tmp_path: Path) -> None:
    p = tmp_path / "clean.json"
    p.write_text(json.dumps(_minimal_clean()), encoding="utf-8")
    assert main([str(p), "--quiet"]) == 0


def test_cli_returns_one_on_problems(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text(json.dumps({"nodes": [1], "links": []}), encoding="utf-8")
    assert main([str(p), "--quiet"]) == 1


def test_cli_returns_two_on_missing_file(tmp_path: Path) -> None:
    assert main([str(tmp_path / "does-not-exist.json"), "--quiet"]) == 2


def test_cli_returns_two_on_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid", encoding="utf-8")
    assert main([str(p), "--quiet"]) == 2
