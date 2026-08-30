"""Standalone CLI for deterministic M4 measurement recommendation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

from .constraints import ConstraintError
from .measurement_candidates import (
    CandidateError,
    MeasurementCatalog,
    combine_candidates,
    load_catalog,
)
from .parameter_graph import GraphError, load_graph, save_graph
from .provenance import ProvenanceError, canonical_json_bytes
from .recommendation import (
    RecommendationError,
    find_candidate,
    recommend_measurements,
    record_measurement,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m measurepilot.recommendation_cli"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    recommend_parser = commands.add_parser(
        "recommend",
        help="rank next physical measurements and write canonical JSON",
    )
    recommend_parser.add_argument("graph", type=Path)
    recommend_parser.add_argument("output", type=Path)
    recommend_parser.add_argument("--catalog", type=Path)
    recommend_parser.add_argument("--limit", type=int, default=5)

    inspect_parser = commands.add_parser(
        "inspect-candidate",
        help="print one generated or catalog candidate with its score",
    )
    inspect_parser.add_argument("graph", type=Path)
    inspect_parser.add_argument("candidate_id")
    inspect_parser.add_argument("--catalog", type=Path)

    record_parser = commands.add_parser(
        "record",
        help="record a selected direct or composite measurement",
    )
    record_parser.add_argument("graph", type=Path)
    record_parser.add_argument("candidate_id")
    record_parser.add_argument("value", type=float)
    record_parser.add_argument("uncertainty", type=float)
    record_parser.add_argument("--catalog", type=Path)
    record_parser.add_argument("--supersedes")
    record_parser.add_argument("--note", default="")

    return parser


def _normalised(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve(strict=False)))


def _catalog(path: Path | None) -> MeasurementCatalog | None:
    return load_catalog(path) if path is not None else None


def _emit(value: object) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode("utf-8"))


def _atomic_write(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_catalog_path(graph_path: Path, catalog_path: Path | None) -> None:
    if catalog_path is not None and _normalised(graph_path) == _normalised(catalog_path):
        raise RecommendationError("graph and catalog paths must be distinct")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        _validate_catalog_path(arguments.graph, getattr(arguments, "catalog", None))
        graph = load_graph(arguments.graph)
        catalog = _catalog(getattr(arguments, "catalog", None))
        candidates = combine_candidates(graph, catalog=catalog)

        if arguments.command == "recommend":
            if _normalised(arguments.graph) == _normalised(arguments.output):
                raise RecommendationError("graph input and recommendation output must be distinct")
            if arguments.catalog is not None and _normalised(arguments.catalog) == _normalised(arguments.output):
                raise RecommendationError("catalog input and recommendation output must be distinct")
            report = recommend_measurements(
                graph,
                candidates,
                limit=arguments.limit,
            )
            _atomic_write(arguments.output, report.canonical_bytes())
            _emit(report.to_dict())
            return 0

        if arguments.command == "inspect-candidate":
            candidate = find_candidate(candidates, arguments.candidate_id)
            report = recommend_measurements(graph, (candidate,), limit=1)
            _emit(report.recommendations[0].to_dict())
            return 0

        if arguments.command == "record":
            candidate = find_candidate(candidates, arguments.candidate_id)
            result = record_measurement(
                graph,
                candidate,
                value=arguments.value,
                uncertainty=arguments.uncertainty,
                supersedes=arguments.supersedes,
                note=arguments.note,
            )
            save_graph(result.graph, arguments.graph)
            _emit(result.to_dict())
            return 0
    except (
        CandidateError,
        RecommendationError,
        GraphError,
        ConstraintError,
        ProvenanceError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
