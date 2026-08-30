from __future__ import annotations

import json
from pathlib import Path

from measurepilot.cli import main

PROJECT_ID = "12345678-1234-5678-9234-567812345678"
TIMESTAMP = "2026-08-30T17:00:00Z"


def test_cli_create_inspect_and_validate(tmp_path: Path, capsys: object) -> None:
    path = tmp_path / "cli.mpilot"

    assert (
        main(
            [
                "project",
                "create",
                str(path),
                "--name",
                "CLI plate",
                "--project-id",
                PROJECT_ID,
                "--timestamp",
                TIMESTAMP,
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["project_id"] == PROJECT_ID

    assert main(["project", "inspect", str(path)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected == created

    assert main(["project", "validate", str(path)]) == 0
    assert capsys.readouterr().out == f"VALID {path}\n"


def test_cli_reports_validation_failure(tmp_path: Path, capsys: object) -> None:
    path = tmp_path / "broken.mpilot"
    path.write_text("not a zip", encoding="utf-8")

    assert main(["project", "validate", str(path)]) == 2
    assert "ERROR:" in capsys.readouterr().err
