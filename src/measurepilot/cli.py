"""Command-line entry point for MeasurePilot M0 and M1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .calibration import generate_calibration_pdf
from .errors import MeasurePilotError
from .model import MeasurePilotProject
from .project import load_project, save_project
from .rectification import rectify_image_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="measurepilot")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    project_parser = commands.add_parser("project", help="create and inspect .mpilot projects")
    project_commands = project_parser.add_subparsers(dest="project_command", required=True)

    create_parser = project_commands.add_parser("create", help="create a new project")
    create_parser.add_argument("path", type=Path)
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--project-id")
    create_parser.add_argument("--timestamp")

    inspect_parser = project_commands.add_parser("inspect", help="print project metadata as JSON")
    inspect_parser.add_argument("path", type=Path)

    validate_parser = project_commands.add_parser("validate", help="validate a project archive")
    validate_parser.add_argument("path", type=Path)

    calibration_parser = commands.add_parser(
        "calibration", help="generate and use the versioned calibration sheet"
    )
    calibration_commands = calibration_parser.add_subparsers(
        dest="calibration_command", required=True
    )

    sheet_parser = calibration_commands.add_parser(
        "sheet", help="generate the printable A4 calibration PDF"
    )
    sheet_parser.add_argument("path", type=Path)

    rectify_parser = calibration_commands.add_parser(
        "rectify", help="rectify a photographed calibration sheet to metric A4 coordinates"
    )
    rectify_parser.add_argument("input", type=Path)
    rectify_parser.add_argument("output", type=Path)
    rectify_parser.add_argument("--report", type=Path)
    rectify_parser.add_argument("--px-per-mm", type=float, default=4.0)

    return parser


def _summary(project: MeasurePilotProject) -> dict[str, object]:
    return {
        "created_at": project.created_at,
        "modified_at": project.modified_at,
        "name": project.name,
        "project_id": project.project_id,
        "schema_version": 1,
        "units": project.units,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "project" and arguments.project_command == "create":
            project = MeasurePilotProject.create(
                arguments.name,
                project_id=arguments.project_id,
                timestamp=arguments.timestamp,
            )
            save_project(project, arguments.path)
            print(json.dumps(_summary(project), ensure_ascii=False, sort_keys=True))
            return 0

        if arguments.command == "project" and arguments.project_command == "inspect":
            project = load_project(arguments.path)
            print(json.dumps(_summary(project), ensure_ascii=False, sort_keys=True))
            return 0

        if arguments.command == "project" and arguments.project_command == "validate":
            load_project(arguments.path)
            print(f"VALID {arguments.path}")
            return 0

        if arguments.command == "calibration" and arguments.calibration_command == "sheet":
            output = generate_calibration_pdf(arguments.path)
            print(f"CREATED {output}")
            return 0

        if arguments.command == "calibration" and arguments.calibration_command == "rectify":
            _output, _report_path, report = rectify_image_file(
                arguments.input,
                arguments.output,
                report_destination=arguments.report,
                output_px_per_mm=arguments.px_per_mm,
            )
            print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
            return 0
    except MeasurePilotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    parser.error("unsupported command")
    return 2
