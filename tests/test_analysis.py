from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from measurepilot.analysis import (
    ANALYSIS_FORMAT,
    ANALYSIS_VERSION,
    analyze_file,
    analyze_rectified_image,
    canonical_analysis_json,
    main,
)
from measurepilot.calibration import render_calibration_image
from measurepilot.segmentation import AnalysisError

PX_PER_MM = 4.0


def _capture() -> np.ndarray:
    image = render_calibration_image(PX_PER_MM)
    cv2.rectangle(image, (240, 740), (600, 940), 35, -1)
    cv2.circle(image, (320, 840), 20, 255, -1)
    cv2.circle(image, (520, 840), 20, 255, -1)
    return image


def test_analysis_report_is_metric_and_byte_deterministic() -> None:
    first = analyze_rectified_image(_capture(), px_per_mm=PX_PER_MM)
    second = analyze_rectified_image(_capture(), px_per_mm=PX_PER_MM)
    assert first.report == second.report
    assert canonical_analysis_json(first.report) == canonical_analysis_json(second.report)
    assert first.report["format"] == ANALYSIS_FORMAT
    assert first.report["version"] == ANALYSIS_VERSION
    assert first.report["coordinate_system"]["unit"] == "mm"
    assert len(first.report["features"]["line_segments"]) == 4
    assert len(first.report["features"]["circular_holes"]) == 2
    assert first.overlay.shape == (1188, 840, 3)
    assert first.mask.shape == (1188, 840)


def test_analyze_file_writes_json_and_overlay(tmp_path) -> None:
    source = tmp_path / "rectified.png"
    report_path = tmp_path / "analysis.json"
    overlay_path = tmp_path / "overlay.png"
    assert cv2.imwrite(str(source), _capture())

    report = analyze_file(
        source,
        report_path,
        overlay_path=overlay_path,
        px_per_mm=PX_PER_MM,
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    assert overlay is not None
    assert overlay.shape == (1188, 840, 3)
    assert not list(tmp_path.glob("*.tmp"))


def test_path_collision_is_rejected_before_source_write(tmp_path) -> None:
    source = tmp_path / "rectified.png"
    assert cv2.imwrite(str(source), _capture())
    before = source.read_bytes()
    with pytest.raises(AnalysisError, match="paths must be distinct"):
        analyze_file(source, source, px_per_mm=PX_PER_MM)
    assert source.read_bytes() == before


def test_module_cli_generates_outputs(tmp_path, capsys) -> None:
    source = tmp_path / "rectified.png"
    report_path = tmp_path / "analysis.json"
    overlay_path = tmp_path / "overlay.png"
    assert cv2.imwrite(str(source), _capture())
    exit_code = main(
        [
            str(source),
            str(report_path),
            "--overlay",
            str(overlay_path),
            "--px-per-mm",
            "4",
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert json.loads(stdout)["format"] == ANALYSIS_FORMAT
    assert report_path.exists()
    assert overlay_path.exists()


def test_module_cli_fails_without_candidate(tmp_path, capsys) -> None:
    source = tmp_path / "blank.png"
    report_path = tmp_path / "analysis.json"
    assert cv2.imwrite(str(source), np.full((1188, 840), 255, dtype=np.uint8))
    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(report_path), "--px-per-mm", "4", "--polarity", "dark"])
    assert exit_info.value.code == 2
    assert "ERROR:" in capsys.readouterr().err
    assert not report_path.exists()
