from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import measurepilot.project as project_module
from measurepilot.errors import ProjectFormatError, UnsafeArchiveError
from measurepilot.model import MeasurePilotProject
from measurepilot.project import REQUIRED_ENTRY_NAMES, load_project, save_project

PROJECT_ID = "12345678-1234-5678-9234-567812345678"
TIMESTAMP = "2026-08-30T17:00:00Z"


def _project() -> MeasurePilotProject:
    project = MeasurePilotProject.create(
        "Reference plate", project_id=PROJECT_ID, timestamp=TIMESTAMP
    )
    project.measurements.append(
        {
            "measurement_id": "width",
            "status": "measured",
            "uncertainty_mm": 0.1,
            "value_mm": 84.4,
        }
    )
    project.geometry["profile"] = {"kind": "unresolved"}
    return project


def _rewrite_entry(path: Path, entry_name: str, replacement: bytes) -> None:
    with zipfile.ZipFile(path, "r") as source:
        entries = {info.filename: source.read(info) for info in source.infolist()}
    entries[entry_name] = replacement
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as target:
        for name, content in entries.items():
            target.writestr(name, content)


def test_round_trip_preserves_project(tmp_path: Path) -> None:
    source = _project()
    path = tmp_path / "part.mpilot"

    save_project(source, path)
    loaded = load_project(path)

    assert loaded == source


def test_same_project_produces_byte_identical_archive(tmp_path: Path) -> None:
    first = tmp_path / "first.mpilot"
    second = tmp_path / "second.mpilot"

    save_project(_project(), first)
    save_project(_project(), second)

    assert first.read_bytes() == second.read_bytes()
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(
        second.read_bytes()
    ).hexdigest()


def test_archive_has_exact_required_members_in_lexical_order(tmp_path: Path) -> None:
    path = save_project(_project(), tmp_path / "part.mpilot")

    with zipfile.ZipFile(path) as archive:
        assert tuple(archive.namelist()) == REQUIRED_ENTRY_NAMES
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = save_project(_project(), tmp_path / "part.mpilot")
    _rewrite_entry(
        path,
        "project.json",
        json.dumps({"schema_version": 1, "project_id": PROJECT_ID}).encode("utf-8"),
    )

    with pytest.raises(ProjectFormatError, match="hash verification"):
        load_project(path)


def test_unexpected_or_nested_member_is_rejected(tmp_path: Path) -> None:
    path = save_project(_project(), tmp_path / "part.mpilot")
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("../escape.json", b"{}")

    with pytest.raises(UnsafeArchiveError, match="unsafe archive member"):
        load_project(path)


def test_atomic_save_keeps_existing_file_when_writer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "part.mpilot"
    destination.write_bytes(b"existing-good-bytes")

    def fail_writer(path: Path, entries: object) -> None:
        path.write_bytes(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(project_module, "_write_archive", fail_writer)

    with pytest.raises(OSError, match="simulated disk failure"):
        save_project(_project(), destination)

    assert destination.read_bytes() == b"existing-good-bytes"
    assert list(tmp_path.glob(".*.tmp")) == []
