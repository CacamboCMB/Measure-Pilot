"""Deterministic and atomic `.mpilot` persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .errors import ProjectFormatError, ProjectValidationError, UnsafeArchiveError
from .model import FORMAT_NAME, INTERNAL_LENGTH_UNIT, SCHEMA_VERSION, MeasurePilotProject

REQUIRED_ENTRY_NAMES = (
    "geometry.json",
    "history.json",
    "hypotheses.json",
    "manifest.json",
    "measurements.json",
    "project.json",
)
DATA_ENTRY_NAMES = tuple(name for name in REQUIRED_ENTRY_NAMES if name != "manifest.json")
MAX_ENTRY_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON in the project-wide canonical representation."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ProjectValidationError(f"value is not canonical JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _project_data(project: MeasurePilotProject) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "project_id": project.project_id,
        "name": project.name,
        "created_at": project.created_at,
        "modified_at": project.modified_at,
        "units": project.units,
    }


def _collection_data(key: str, items: Any) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, key: items}


def project_entries(project: MeasurePilotProject) -> dict[str, bytes]:
    """Build all archive entries for a validated project."""

    project.validate()
    data_entries = {
        "project.json": canonical_json_bytes(_project_data(project)),
        "measurements.json": canonical_json_bytes(_collection_data("items", project.measurements)),
        "geometry.json": canonical_json_bytes(_collection_data("model", project.geometry)),
        "hypotheses.json": canonical_json_bytes(_collection_data("items", project.hypotheses)),
        "history.json": canonical_json_bytes(_collection_data("events", project.history)),
    }
    manifest = {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "project_id": project.project_id,
        "units": INTERNAL_LENGTH_UNIT,
        "entries": {name.removesuffix(".json"): name for name in sorted(DATA_ENTRY_NAMES)},
        "sha256": {name: _sha256(data_entries[name]) for name in sorted(DATA_ENTRY_NAMES)},
    }
    return {**data_entries, "manifest.json": canonical_json_bytes(manifest)}


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits = 0
    return info


def _write_archive(path: Path, entries: Mapping[str, bytes]) -> None:
    with path.open("wb") as raw:
        with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
            for name in sorted(entries):
                archive.writestr(_zip_info(name), entries[name])
        raw.flush()
        os.fsync(raw.fileno())


def save_project(project: MeasurePilotProject, destination: str | os.PathLike[str]) -> Path:
    """Atomically write a deterministic project archive."""

    target = Path(destination)
    if target.exists() and target.is_dir():
        raise ProjectValidationError(f"destination is a directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = project_entries(project)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_archive(temporary_path, entries)
        os.replace(temporary_path, target)
        _fsync_directory(target.parent)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return target


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _validate_member_name(name: str) -> None:
    if not name or "\x00" in name or "\\" in name:
        raise UnsafeArchiveError(f"unsafe archive member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise UnsafeArchiveError(f"unsafe archive member path: {name!r}")
    if len(path.parts) != 1:
        raise UnsafeArchiveError(f"nested archive member is not allowed: {name!r}")


def _read_json_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any]:
    if info.file_size > MAX_ENTRY_BYTES:
        raise ProjectFormatError(f"archive member exceeds {MAX_ENTRY_BYTES} bytes: {info.filename}")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ProjectFormatError(f"unsupported compression for {info.filename}")
    try:
        raw = archive.read(info)
    except (RuntimeError, zipfile.BadZipFile) as exc:
        raise ProjectFormatError(f"cannot read {info.filename}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectFormatError(f"{info.filename} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProjectFormatError(f"{info.filename} must contain a JSON object")
    return value


def _require_schema(value: dict[str, Any], entry_name: str) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ProjectFormatError(
            f"{entry_name} uses unsupported schema_version {value.get('schema_version')!r}"
        )


def load_project(source: str | os.PathLike[str]) -> MeasurePilotProject:
    """Load and strictly validate a version-1 project archive."""

    path = Path(source)
    try:
        archive_size = path.stat().st_size
    except OSError as exc:
        raise ProjectFormatError(f"cannot access project: {path}") from exc
    if archive_size > MAX_ARCHIVE_BYTES:
        raise ProjectFormatError(f"project exceeds {MAX_ARCHIVE_BYTES} bytes")

    try:
        archive = zipfile.ZipFile(path, mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProjectFormatError(f"not a valid .mpilot ZIP archive: {path}") from exc

    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        for name in names:
            _validate_member_name(name)
        if len(names) != len(set(names)):
            raise UnsafeArchiveError("duplicate archive members are not allowed")
        if tuple(sorted(names)) != REQUIRED_ENTRY_NAMES:
            missing = sorted(set(REQUIRED_ENTRY_NAMES) - set(names))
            unexpected = sorted(set(names) - set(REQUIRED_ENTRY_NAMES))
            raise UnsafeArchiveError(
                f"archive membership mismatch; missing={missing}, unexpected={unexpected}"
            )
        if sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
            raise ProjectFormatError("uncompressed project content exceeds safety limit")

        values = {info.filename: _read_json_entry(archive, info) for info in infos}
        raw_entries = {info.filename: archive.read(info) for info in infos if info.filename != "manifest.json"}

    manifest = values["manifest.json"]
    if manifest.get("format") != FORMAT_NAME:
        raise ProjectFormatError("manifest format is not measurepilot-project")
    _require_schema(manifest, "manifest.json")
    if manifest.get("units") != INTERNAL_LENGTH_UNIT:
        raise ProjectFormatError(f"manifest units must be '{INTERNAL_LENGTH_UNIT}'")
    expected_entry_map = {name.removesuffix(".json"): name for name in sorted(DATA_ENTRY_NAMES)}
    if manifest.get("entries") != expected_entry_map:
        raise ProjectFormatError("manifest entry map is invalid")
    expected_hashes = {name: _sha256(raw_entries[name]) for name in sorted(DATA_ENTRY_NAMES)}
    if manifest.get("sha256") != expected_hashes:
        raise ProjectFormatError("archive content hash verification failed")

    project_data = values["project.json"]
    _require_schema(project_data, "project.json")
    for entry_name in DATA_ENTRY_NAMES:
        _require_schema(values[entry_name], entry_name)

    project_id = project_data.get("project_id")
    if manifest.get("project_id") != project_id:
        raise ProjectFormatError("manifest and project project_id values differ")

    try:
        project = MeasurePilotProject(
            project_id=project_id,
            name=project_data["name"],
            created_at=project_data["created_at"],
            modified_at=project_data["modified_at"],
            units=project_data["units"],
            measurements=values["measurements.json"]["items"],
            geometry=values["geometry.json"]["model"],
            hypotheses=values["hypotheses.json"]["items"],
            history=values["history.json"]["events"],
        )
    except KeyError as exc:
        raise ProjectFormatError(f"required project field is missing: {exc.args[0]}") from exc
    project.validate()
    return project
