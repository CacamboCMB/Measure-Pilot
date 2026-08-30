"""Standalone CLI for M3 parameter graph import and measurement revision."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

from .parameter_graph import ParameterGraph, load_graph, save_graph
from .provenance import (
    EvidenceSource,
    GraphValidationError,
    ParameterStatus,
    QuantityKind,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m measurepilot.graph_cli")
    commands = parser.add_subparsers(dest="command", required=True)

    import_parser = commands.add_parser(
        "import-analysis",
        help="create a parameter graph from a canonical M2 report",
    )
    import_parser.add_argument("analysis", type=Path)
    import_parser.add_argument("graph", type=Path)

    measurement = commands.add_parser(
        "add-measurement",
        help="append a physical observation or explicit correction",
    )
    measurement.add_argument("graph", type=Path)
    measurement.add_argument("parameter_id")
    measurement.add_argument("value", type=float)
    measurement.add_argument("--uncertainty", type=float, required=True)
    measurement.add_argument("--source", required=True)
    measurement.add_argument(
        "--kind",
        choices=("length", "angle", "dimensionless"),
        default="length",
    )
    measurement.add_argument(
        "--status",
        choices=("MEASURED", "LOCKED", "ASSUMED"),
        default="MEASURED",
    )
    measurement.add_argument("--supersedes")
    measurement.add_argument("--label")

    inspect_parser = commands.add_parser("inspect", help="print graph resolution")
    inspect_parser.add_argument("graph", type=Path)
    inspect_parser.add_argument("--parameter")

    validate_parser = commands.add_parser("validate", help="validate canonical graph JSON")
    validate_parser.add_argument("graph", type=Path)
    return parser


def _kind(value: str) -> QuantityKind:
    return {
        "length": QuantityKind.LENGTH,
        "angle": QuantityKind.ANGLE,
        "dimensionless": QuantityKind.DIMENSIONLESS,
    }[value]


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "import-analysis":
            raw = arguments.analysis.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise GraphValidationError("analysis report root must be an object")
            graph = ParameterGraph.from_analysis_report(
                payload,
                content_sha256=sha256(raw).hexdigest(),
                reference=str(arguments.analysis),
            )
            digest = save_graph(graph, arguments.graph)
            _print(
                {
                    "graph": str(arguments.graph),
                    "parameters": len(graph.parameters),
                    "revision": graph.revision,
                    "sha256": digest,
                    "status": "CREATED",
                }
            )
            return 0

        if arguments.command == "add-measurement":
            graph = load_graph(arguments.graph)
            source = EvidenceSource.create(
                kind="PHYSICAL_MEASUREMENT",
                reference=arguments.source,
                metadata={"entrypoint": "graph_cli"},
            )
            graph.add_source(source)
            kind = _kind(arguments.kind)
            graph.ensure_parameter(
                arguments.parameter_id,
                kind=kind,
                label=arguments.label,
            )
            observation = graph.add_observation(
                arguments.parameter_id,
                value=arguments.value,
                uncertainty=arguments.uncertainty,
                status=ParameterStatus(arguments.status),
                source_id=source.source_id,
                supersedes=arguments.supersedes,
            )
            digest = save_graph(graph, arguments.graph)
            _print(
                {
                    "graph": str(arguments.graph),
                    "observation_id": observation.observation_id,
                    "parameter": graph.resolve_parameter(
                        arguments.parameter_id
                    ).as_dict(),
                    "revision": graph.revision,
                    "sha256": digest,
                    "status": "UPDATED",
                }
            )
            return 0

        if arguments.command == "inspect":
            graph = load_graph(arguments.graph)
            if arguments.parameter:
                payload = graph.resolve_parameter(arguments.parameter).as_dict()
            else:
                payload = {
                    "constraints": len(graph.constraints),
                    "graph_sha256": graph.sha256(),
                    "observations": len(graph.observations),
                    "parameters": {
                        key: value.as_dict()
                        for key, value in graph.resolved_parameters().items()
                    },
                    "revision": graph.revision,
                    "sources": len(graph.sources),
                }
            _print(payload)
            return 0

        if arguments.command == "validate":
            graph = load_graph(arguments.graph)
            _print(
                {
                    "path": str(arguments.graph),
                    "sha256": graph.sha256(),
                    "status": "VALID",
                }
            )
            return 0
    except (
        GraphValidationError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
