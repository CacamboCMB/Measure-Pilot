from __future__ import annotations

import json

import cv2
import pytest

from measurepilot.analysis import analyze_rectified_image, canonical_analysis_json
from measurepilot.calibration import render_calibration_image
from measurepilot.graph_cli import main
from measurepilot.parameter_graph import load_graph
from measurepilot.provenance import ParameterStatus


def _write_analysis(path) -> None:
    image = render_calibration_image(4.0)
    cv2.rectangle(image, (240, 740), (600, 940), 35, -1)
    cv2.circle(image, (320, 840), 20, 255, -1)
    cv2.circle(image, (520, 840), 20, 255, -1)
    report = analyze_rectified_image(image, px_per_mm=4.0).report
    path.write_bytes(canonical_analysis_json(report))


def test_cli_import_measure_correct_inspect_and_validate(tmp_path, capsys) -> None:
    analysis = tmp_path / "analysis.json"
    graph_path = tmp_path / "part.graph.json"
    _write_analysis(analysis)

    assert main(["import-analysis", str(analysis), str(graph_path)]) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["status"] == "CREATED"
    assert imported["parameters"] == 32

    parameter_id = "feature.circular-hole-000.diameter"
    assert (
        main(
            [
                "add-measurement",
                str(graph_path),
                parameter_id,
                "10.02",
                "--uncertainty",
                "0.05",
                "--source",
                "digital caliper",
            ]
        )
        == 0
    )
    measured = json.loads(capsys.readouterr().out)
    first_measurement_id = measured["observation_id"]
    assert measured["parameter"]["status"] == "MEASURED"
    assert measured["parameter"]["value"] == pytest.approx(10.02)

    assert (
        main(
            [
                "add-measurement",
                str(graph_path),
                parameter_id,
                "10.10",
                "--uncertainty",
                "0.05",
                "--source",
                "digital caliper recheck",
                "--supersedes",
                first_measurement_id,
            ]
        )
        == 0
    )
    corrected = json.loads(capsys.readouterr().out)
    assert corrected["parameter"]["value"] == pytest.approx(10.10)

    assert main(["inspect", str(graph_path), "--parameter", parameter_id]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["status"] == "MEASURED"
    assert inspected["value"] == pytest.approx(10.10)

    assert main(["validate", str(graph_path)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "VALID"

    graph = load_graph(graph_path)
    assert graph.revision == 34
    assert len(graph.observations) == 34
    assert graph.observations[-1].supersedes == first_measurement_id
    assert graph.resolve_parameter(parameter_id).status is ParameterStatus.MEASURED


def test_cli_rejects_unknown_superseded_observation(tmp_path, capsys) -> None:
    analysis = tmp_path / "analysis.json"
    graph_path = tmp_path / "part.graph.json"
    _write_analysis(analysis)
    assert main(["import-analysis", str(analysis), str(graph_path)]) == 0
    capsys.readouterr()

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "add-measurement",
                str(graph_path),
                "feature.circular-hole-000.diameter",
                "10.0",
                "--uncertainty",
                "0.05",
                "--source",
                "caliper",
                "--supersedes",
                "observation-does-not-exist",
            ]
        )
    assert exit_info.value.code == 2
    assert "ERROR:" in capsys.readouterr().err


def test_cli_rejects_invalid_graph_json(tmp_path, capsys) -> None:
    graph_path = tmp_path / "invalid.graph.json"
    graph_path.write_text("not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["validate", str(graph_path)])
    assert exit_info.value.code == 2
    assert "ERROR:" in capsys.readouterr().err
