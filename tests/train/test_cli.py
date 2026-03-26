from __future__ import annotations

import json
from pathlib import Path

from bp_train import cli


def test_prepare_cli_dispatches_to_prepare_artifact(monkeypatch):
    captured: dict[str, object] = {}

    def fake_prepare_artifact(*, input_json, output_json, custom_py, config):
        captured["input_json"] = input_json
        captured["output_json"] = output_json
        captured["custom_py"] = custom_py
        captured["config"] = config

    monkeypatch.setattr(cli, "prepare_artifact", fake_prepare_artifact)

    exit_code = cli.main(
        [
            "prepare",
            "--input",
            "input.json",
            "--output",
            "output.json",
            "--custom",
            "custom.py",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "input_json": "input.json",
        "output_json": "output.json",
        "custom_py": "custom.py",
        "config": None,
    }


def test_prepare_cli_loads_config_json(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    config_path = tmp_path / "prepare-config.json"
    config_path.write_text(
        json.dumps({"required_control_names": ["carbon_feed", "temperature"]}),
        encoding="utf-8",
    )

    def fake_prepare_artifact(*, input_json, output_json, custom_py, config):
        captured["input_json"] = input_json
        captured["output_json"] = output_json
        captured["custom_py"] = custom_py
        captured["config"] = config

    monkeypatch.setattr(cli, "prepare_artifact", fake_prepare_artifact)

    exit_code = cli.main(
        [
            "prepare",
            "--input",
            "input.json",
            "--output",
            "output.json",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    assert captured == {
        "input_json": "input.json",
        "output_json": "output.json",
        "custom_py": None,
        "config": {"required_control_names": ["carbon_feed", "temperature"]},
    }
