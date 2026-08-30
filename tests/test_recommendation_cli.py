from __future__ import annotations

import json

from measurepilot.measurement_candidates import combine_candidates
from measurepilot.parameter_graph import ParameterGraph, load_graph, save_graph
from measurepilot.provenance import QuantityKind
from measurepilot.recommendation_cli import main


def _estimated_graph() -> ParameterGraph:
    graph = ParameterGraph()
    graph.ensure_parameter("x", QuantityKind.LENGTH)
    return graph


def test_cli_recommend_inspect_and_record(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    report_path = tmp_path / "recommendations.json"
    graph = _estimated_graph()
    save_graph(graph, graph_path)
    candidate = combine_candidates(graph)[0]

    assert (
        main(
            [
                "recommend",
                str(graph_path),
                str(report_path),
                "--limit",
                "3",
            ]
        )
        == 0
    )
    report_stdout = json.loads(capsys.readouterr().out)
    assert json.loads(report_path.read_text(encoding="utf-8")) == report_stdout
    assert report_stdout["recommendations"][0]["candidate"]["candidate_id"] == candidate.candidate_id

    assert (
        main(
            [
                "inspect-candidate",
                str(graph_path),
                candidate.candidate_id,
            ]
        )
        == 0
    )
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["candidate"]["candidate_id"] == candidate.candidate_id

    assert (
        main(
            [
                "record",
                str(graph_path),
                candidate.candidate_id,
                "54.4",
                "0.1",
                "--note",
                "physical reading",
            ]
        )
        == 0
    )
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["resolution"]["value"] == 54.4
    loaded = load_graph(graph_path)
    assert loaded.resolve_observation_values()["x"].value == 54.4


def test_cli_conflict_failure_is_atomic(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    graph = ParameterGraph()
    graph.append_measurement(
        parameter_id="pitch",
        quantity=QuantityKind.LENGTH,
        value=52.8,
        uncertainty=0.1,
    )
    graph.append_measurement(
        parameter_id="pitch",
        quantity=QuantityKind.LENGTH,
        value=54.4,
        uncertainty=0.1,
        note="conflict",
    )
    save_graph(graph, graph_path)
    candidate = next(
        item for item in combine_candidates(graph) if item.direct_parameter_id == "pitch"
    )
    before = graph_path.read_bytes()

    assert (
        main(
            [
                "record",
                str(graph_path),
                candidate.candidate_id,
                "54.4",
                "0.1",
            ]
        )
        == 2
    )
    assert "requires --supersedes" in capsys.readouterr().err
    assert graph_path.read_bytes() == before


def test_cli_rejects_graph_output_collision_without_modification(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    graph = _estimated_graph()
    save_graph(graph, graph_path)
    before = graph_path.read_bytes()
    assert main(["recommend", str(graph_path), str(graph_path)]) == 2
    assert "must be distinct" in capsys.readouterr().err
    assert graph_path.read_bytes() == before
