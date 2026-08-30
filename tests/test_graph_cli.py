from __future__ import annotations

import json

from measurepilot.graph_cli import main
from measurepilot.parameter_graph import load_graph
from measurepilot.provenance import canonical_json_bytes


def _write_analysis(path) -> None:
    report = {
        "configuration": {
            "features": {},
            "segmentation": {"px_per_mm": 4.0},
        },
        "contours": {"outer": {"simplification_rms_mm": 0.1}},
        "coordinate_system": {
            "origin": "top_left_of_rectified_a4_page",
            "unit": "mm",
            "x_axis": "right",
            "y_axis": "down",
        },
        "features": {
            "circular_holes": [],
            "line_segments": [
                {
                    "end_mm": [54.4, 0.0],
                    "id": "outer-line-000",
                    "length_mm": 54.4,
                    "start_mm": [0.0, 0.0],
                }
            ],
        },
        "format": "measurepilot-planar-analysis",
        "version": 1,
    }
    path.write_bytes(canonical_json_bytes(report))


def test_cli_import_append_correct_inspect_and_validate(tmp_path, capsys) -> None:
    analysis = tmp_path / "analysis.json"
    graph_path = tmp_path / "graph.json"
    _write_analysis(analysis)

    assert main(["import-analysis", str(analysis), str(graph_path)]) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["counts"]["parameters"] == 5

    assert (
        main(
            [
                "append-measurement",
                str(graph_path),
                "plate.hole_pitch",
                "52.8",
                "0.1",
                "--quantity",
                "length",
                "--note",
                "initial",
            ]
        )
        == 0
    )
    appended = json.loads(capsys.readouterr().out)
    old_id = appended["observation"]["observation_id"]
    assert appended["resolution"]["value"] == 52.8

    assert (
        main(
            [
                "correct-measurement",
                str(graph_path),
                old_id,
                "54.4",
                "0.1",
                "--note",
                "corrected",
            ]
        )
        == 0
    )
    corrected = json.loads(capsys.readouterr().out)
    assert corrected["observation"]["supersedes"] == old_id
    assert corrected["resolution"]["value"] == 54.4

    assert (
        main(
            [
                "inspect",
                str(graph_path),
                "--parameter",
                "plate.hole_pitch",
            ]
        )
        == 0
    )
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["resolution"]["value"] == 54.4
    assert len(inspected["observations"]) == 2

    assert main(["validate", str(graph_path)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["format"] == "measurepilot-parameter-graph"
    assert validated["status"] == "VALID"
    assert validated["version"] == 1
    assert len(load_graph(graph_path).observations) == 7


def test_cli_failure_does_not_replace_graph(tmp_path, capsys) -> None:
    analysis = tmp_path / "analysis.json"
    graph_path = tmp_path / "graph.json"
    _write_analysis(analysis)
    assert main(["import-analysis", str(analysis), str(graph_path)]) == 0
    capsys.readouterr()
    before = graph_path.read_bytes()
    assert (
        main(
            [
                "append-measurement",
                str(graph_path),
                "bad.parameter",
                "5",
                "0",
                "--quantity",
                "length",
            ]
        )
        == 2
    )
    assert "ERROR:" in capsys.readouterr().err
    assert graph_path.read_bytes() == before


def test_cli_rejects_analysis_output_path_collision(tmp_path, capsys) -> None:
    analysis = tmp_path / "analysis.json"
    _write_analysis(analysis)
    before = analysis.read_bytes()
    assert main(["import-analysis", str(analysis), str(analysis)]) == 2
    assert "must be distinct" in capsys.readouterr().err
    assert analysis.read_bytes() == before
