"""Command-line entry point for MeasurePilot M0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .errors import MeasurePilotError
from .model import MeasurePilotProject
from .project import load_project, save_project


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
    except MeasurePilotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    parser.error("unsupported command")
    return 2
