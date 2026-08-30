"""Standalone module CLI for the MeasurePilot M3 parameter graph."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

from .constraints import ConstraintError
from .parameter_graph import (
    GraphError,
    import_m2_analysis,
    inspect_graph,
    load_graph,
    save_graph,
)
from .provenance import (
    ParameterStatus,
    ProvenanceError,
    QuantityKind,
    canonical_json_bytes,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m measurepilot.graph_cli")
    commands = parser.add_subparsers(dest="command", required=True)

    import_parser = commands.add_parser(
        "import-analysis",
        help="create a graph from canonical M2 analysis JSON",
    )
    import_parser.add_argument("analysis", type=Path)
    import_parser.add_argument("graph", type=Path)

    append_parser = commands.add_parser(
        "append-measurement",
        help="append one physical scalar observation",
    )
    append_parser.add_argument("graph", type=Path)
    append_parser.add_argument("parameter_id")
    append_parser.add_argument("value", type=float)
    append_parser.add_argument("uncertainty", type=float)
    append_parser.add_argument(
        "--quantity",
        choices=tuple(item.value for item in QuantityKind),
        required=True,
    )
    append_parser.add_argument(
        "--status",
        choices=(
            ParameterStatus.MEASURED.value,
            ParameterStatus.ASSUMED.value,
            ParameterStatus.LOCKED.value,
        ),
        default=ParameterStatus.MEASURED.value,
    )
    append_parser.add_argument("--note", default="")

    correct_parser = commands.add_parser(
        "correct-measurement",
        help="append a correction that supersedes an observation",
    )
    correct_parser.add_argument("graph", type=Path)
    correct_parser.add_argument("observation_id")
    correct_parser.add_argument("value", type=float)
    correct_parser.add_argument("uncertainty", type=float)
    correct_parser.add_argument(
        "--status",
        choices=(
            ParameterStatus.MEASURED.value,
            ParameterStatus.ASSUMED.value,
            ParameterStatus.LOCKED.value,
        ),
        default=ParameterStatus.MEASURED.value,
    )
    correct_parser.add_argument("--note", default="")

    inspect_parser = commands.add_parser(
        "inspect",
        help="print graph state or one parameter",
    )
    inspect_parser.add_argument("graph", type=Path)
    inspect_parser.add_argument("--parameter")

    validate_parser = commands.add_parser(
        "validate",
        help="validate canonical graph JSON",
    )
    validate_parser.add_argument("graph", type=Path)

    return parser


def _normalised(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve(strict=False)))


def _emit(value: object) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "import-analysis":
            if _normalised(arguments.analysis) == _normalised(arguments.graph):
                raise GraphError("analysis input and graph output must be distinct")
            report_bytes = arguments.analysis.read_bytes()
            graph = import_m2_analysis(report_bytes)
            save_graph(graph, arguments.graph)
            _emit(inspect_graph(graph))
            return 0

        if arguments.command == "append-measurement":
            graph = load_graph(arguments.graph)
            observation = graph.append_measurement(
                parameter_id=arguments.parameter_id,
                quantity=QuantityKind(arguments.quantity),
                value=arguments.value,
                uncertainty=arguments.uncertainty,
                status=ParameterStatus(arguments.status),
                note=arguments.note,
            )
            save_graph(graph, arguments.graph)
            _emit(
                {
                    "observation": observation.to_dict(),
                    "resolution": inspect_graph(
                        graph,
                        parameter_id=arguments.parameter_id,
                    )["resolution"],
                }
            )
            return 0

        if arguments.command == "correct-measurement":
            graph = load_graph(arguments.graph)
            observation = graph.correct_measurement(
                arguments.observation_id,
                value=arguments.value,
                uncertainty=arguments.uncertainty,
                status=ParameterStatus(arguments.status),
                note=arguments.note,
            )
            save_graph(graph, arguments.graph)
            _emit(
                {
                    "observation": observation.to_dict(),
                    "resolution": inspect_graph(
                        graph,
                        parameter_id=observation.parameter_id,
                    )["resolution"],
                }
            )
            return 0

        if arguments.command == "inspect":
            graph = load_graph(arguments.graph)
            _emit(inspect_graph(graph, parameter_id=arguments.parameter))
            return 0

        if arguments.command == "validate":
            graph = load_graph(arguments.graph)
            graph.validate()
            _emit(
                {
                    "format": "measurepilot-parameter-graph",
                    "path": str(arguments.graph),
                    "status": "VALID",
                    "version": 1,
                }
            )
            return 0
    except (GraphError, ConstraintError, ProvenanceError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
