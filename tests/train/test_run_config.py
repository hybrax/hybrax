from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from bp_train.run_config import (
    DefaultCustomConfig,
    ForwardRunConfig,
    RunConfig,
    load_forward_config,
    load_loo_config,
    load_prepare_config,
    load_train_config,
)

_ROOT = Path(__file__).parents[1]
_EXAMPLES = _ROOT / "examples"
_EXAMPLE_LOADERS = {
    "forward": load_forward_config,
    "loo": load_loo_config,
    "prepare": load_prepare_config,
    "train": load_train_config,
}
_EXAMPLE_CONFIGS = tuple(
    _ROOT / path
    for path in subprocess.check_output(
        ["git", "ls-files", "examples/**/*.json"],
        cwd=_ROOT,
        text=True,
    ).splitlines()
    if not any(part.startswith("output") for part in Path(path).parts)
    and Path(path).name.split("-", 1)[0] in _EXAMPLE_LOADERS
)
assert _EXAMPLE_CONFIGS, "no example configs discovered; schema-drift guard is inert"


def _write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize(
    "config_path",
    _EXAMPLE_CONFIGS,
    ids=lambda path: str(path.relative_to(_EXAMPLES)),
)
def test_active_example_configs_load(config_path: Path) -> None:
    kind = config_path.name.split("-", 1)[0]
    _EXAMPLE_LOADERS[kind](config_path)


def test_config_accepts_comments_without_preprocessing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        "// training input\n"
        "{\n"
        "  // paths are resolved from this file\n"
        '  "data": {"prepared": "prepared.json"},\n'
        '  "custom": {"url": "https://example.com/data",\n'
        '             "note": "before\u2028//must-stay\u2028after"}\n'
        "}\n"
        "// final comment",
        encoding="utf-8",
    )

    loaded = load_train_config(config_path)

    assert loaded.config.data is not None
    assert loaded.config.custom.url == "https://example.com/data"
    assert loaded.config.custom.note == "before\u2028//must-stay\u2028after"


@pytest.mark.parametrize(
    "document",
    [
        '{"data": {"prepared": "prepared.json"} // inline\n}',
        '{/* block comment */ "data": {"prepared": "prepared.json"}}',
    ],
)
def test_config_accepts_inline_and_block_comments(
    tmp_path: Path, document: str
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(document, encoding="utf-8")

    assert load_train_config(config_path).config.data is not None


def test_unknown_top_level_keys_fail(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path / "config.json",
        {"data": {"prepared": "prepared.json"}, "typo": 1},
    )

    with pytest.raises(ValueError, match="unknown top-level"):
        load_train_config(config_path)


def test_unknown_keys_in_command_relevant_typed_sections_fail(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path / "config.json",
        {"data": {"prepared": "prepared.json", "typo": 1}},
    )

    with pytest.raises(ValidationError) as exc_info:
        load_train_config(config_path)

    assert "data.typo" in str(exc_info.value)


def test_command_specific_validation_ignores_irrelevant_known_sections(
    tmp_path: Path,
) -> None:
    config_path = _write_json(
        tmp_path / "config.json",
        {
            "prepare": {"raw_input": "raw.json"},
            "data": {"prepared": "prepared.json", "typo": 1},
        },
    )

    loaded = load_prepare_config(config_path)

    assert loaded.config.prepare is not None
    assert loaded.config.data is None


def test_train_validation_ignores_irrelevant_prepare_section(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path / "config.json",
        {
            "data": {"prepared": "prepared.json"},
            "prepare": {"raw_input": "raw.json", "typo": 1},
        },
    )

    loaded = load_train_config(config_path)

    assert loaded.config.data is not None
    assert loaded.config.prepare is None


def test_bad_typed_path_values_fail_with_pydantic_location(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path / "config.json",
        {"data": {"prepared": 123}},
    )

    with pytest.raises(ValidationError) as exc_info:
        load_train_config(config_path)

    assert "data.prepared" in str(exc_info.value)


def test_custom_must_be_object_or_null(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path / "config.json",
        {"data": {"prepared": "prepared.json"}, "custom": []},
    )

    with pytest.raises(TypeError, match="custom section"):
        load_train_config(config_path)


@pytest.mark.parametrize("custom_value", [None, pytest.param("absent", id="absent")])
def test_absent_or_null_custom_resolves_to_none(
    tmp_path: Path, custom_value: object
) -> None:
    raw: dict[str, object] = {"data": {"prepared": "prepared.json"}}
    if custom_value != "absent":
        raw["custom"] = custom_value
    config_path = _write_json(tmp_path / "config.json", raw)

    loaded = load_train_config(config_path)

    assert loaded.config.custom is None


def test_object_custom_uses_permissive_default_custom_config(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path / "config.json",
        {"data": {"prepared": "prepared.json"}, "custom": {"alpha": 2}},
    )

    loaded = load_train_config(config_path)

    assert isinstance(loaded.config.custom, DefaultCustomConfig)
    assert loaded.config.custom.alpha == 2


def test_get_custom_config_hook_return_is_stored(tmp_path: Path) -> None:
    custom_py = tmp_path / "custom.py"
    custom_py.write_text(
        "def get_custom_config(raw_custom, config):\n"
        "    assert config.custom is None\n"
        "    return {'raw': raw_custom, 'prepared': config.data.prepared}\n"
    )
    config_path = _write_json(
        tmp_path / "config.json",
        {
            "data": {"prepared": "prepared.json"},
            "custom_py": "custom.py",
            "custom": {"alpha": 2},
        },
    )

    loaded = load_train_config(config_path)

    assert loaded.config.custom == {
        "raw": {"alpha": 2},
        "prepared": tmp_path / "prepared.json",
    }


def test_prepare_get_custom_config_hook_receives_prepare_config(tmp_path: Path) -> None:
    custom_py = tmp_path / "custom.py"
    custom_py.write_text(
        "def get_custom_config(raw_custom, config):\n"
        "    assert config.custom is None\n"
        "    return {'raw': raw_custom, 'raw_input': config.prepare.raw_input}\n"
    )
    config_path = _write_json(
        tmp_path / "config.json",
        {
            "prepare": {"raw_input": "raw.json"},
            "custom_py": "custom.py",
            "custom": {},
        },
    )

    loaded = load_prepare_config(config_path)

    assert loaded.config.custom == {"raw": {}, "raw_input": tmp_path / "raw.json"}


def test_reresolve_custom_rewraps_dict_for_attribute_access():
    """A custom section reconstructed from config.json comes back as a raw dict;
    reresolve_custom re-wraps it so hooks can use attribute access (the
    resume/load_run/forward path that crashed with
    'dict object has no attribute target_loss_weights')."""
    from bp_train.run_config import RunConfig, reresolve_custom

    cfg = RunConfig.model_validate(
        {
            "data": {"prepared": "prepared.json"},
            "custom": {"target_loss_weights": {"biomass": 10.0}},
        }
    )
    assert isinstance(cfg.custom, dict)  # raw after JSON round-trip
    resolved = reresolve_custom(cfg, None)  # no module -> DefaultCustomConfig
    assert resolved.custom.target_loss_weights == {"biomass": 10.0}
    # No-op when there is nothing to resolve.
    none_cfg = RunConfig.model_validate({"data": {"prepared": "p"}})
    assert reresolve_custom(none_cfg, None).custom is None


def test_relative_typed_paths_resolve_relative_to_config_parent(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    custom_py = config_dir / "custom.py"
    custom_py.write_text("VALUE = 1\n")
    config_path = _write_json(
        config_dir / "config.json",
        {
            "data": {"prepared": "../data/prepared.json"},
            "prepare": {"raw_input": "raw/input.json"},
            "custom_py": "custom.py",
            "custom": {"path": "not/resolved"},
        },
    )

    train_loaded = load_train_config(config_path)
    prepare_loaded = load_prepare_config(config_path)

    assert train_loaded.config.data is not None
    assert train_loaded.config.data.prepared == tmp_path / "data" / "prepared.json"
    assert train_loaded.config.custom_py == custom_py
    assert train_loaded.config.custom.path == "not/resolved"
    assert prepare_loaded.config.prepare is not None
    assert prepare_loaded.config.prepare.raw_input == config_dir / "raw" / "input.json"


def test_custom_py_hash_and_null(tmp_path: Path) -> None:
    custom_py = tmp_path / "custom.py"
    custom_source = b"VALUE = 1\n"
    custom_py.write_bytes(custom_source)
    with_custom_path = _write_json(
        tmp_path / "with_custom.json",
        {"data": {"prepared": "prepared.json"}, "custom_py": "custom.py"},
    )
    without_custom_path = _write_json(
        tmp_path / "without_custom.json",
        {"data": {"prepared": "prepared.json"}, "custom_py": None},
    )

    with_custom = load_train_config(with_custom_path)
    without_custom = load_train_config(without_custom_path)

    assert with_custom.custom_module is not None
    assert with_custom.custom_py_sha256 == hashlib.sha256(custom_source).hexdigest()
    assert without_custom.custom_module is None
    assert without_custom.custom_py_sha256 is None


def test_train_typed_fields_resolve_from_config(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path / "config.json",
        {
            "data": {
                "prepared": "prepared.json",
                "processes": ["p1", "p2"],
                "targets": ["X", "P"],
                "target_source": "reactor_components",
            },
            "train": {
                "epochs": 7,
                "seed": 12,
                "optimizer": "sgd",
                "learning_rate": 0.02,
                "grad_clip_norm": 3.0,
                "batch_size": 4,
                "shuffle": False,
                "batch_seed": 99,
                "allow_stateful_models": True,
            },
            "solver": {
                "max_steps": 250000,
                "rtol": 1e-4,
                "atol": 1e-6,
                "jump_ts": False,
            },
        },
    )

    loaded = load_train_config(config_path)
    config = loaded.config

    assert config.data is not None
    assert config.data.prepared == tmp_path / "prepared.json"
    assert config.data.processes == ("p1", "p2")
    assert config.data.targets == ("X", "P")
    assert config.data.target_source == "reactor_components"
    assert config.train.epochs == 7
    assert config.train.seed == 12
    assert config.train.optimizer == "sgd"
    assert config.train.learning_rate == 0.02
    assert config.train.grad_clip_norm == 3.0
    assert config.train.batch_size == 4
    assert config.train.shuffle is False
    assert config.train.batch_seed == 99
    assert config.train.allow_stateful_models is True
    assert config.solver.max_steps == 250000
    assert config.solver.rtol == 1e-4
    assert config.solver.atol == 1e-6
    assert config.solver.jump_ts is False


@pytest.mark.parametrize(
    "section",
    [
        {"train": {"steps": 1}},
        {"logging": {"every": 1}},
        {"logging": {"header_every": 1}},
        {"checkpoint": {"keep": "all"}},
        {"loo": {"monitor_every": 1}},
    ],
)
def test_removed_training_cadence_fields_are_rejected(section):
    with pytest.raises(ValueError):
        RunConfig.model_validate(section)


def test_prediction_exports_default_to_none():
    assert RunConfig().output.predictions == "none"
    assert ForwardRunConfig(models=("model",)).output.predictions == "none"


def test_checkpoint_every_defaults_to_auto_and_accepts_explicit_cadence():
    assert RunConfig().checkpoint.every is None
    assert (
        RunConfig.model_validate({"checkpoint": {"every": None}}).checkpoint.every
        is None
    )
    assert RunConfig.model_validate({"checkpoint": {"every": 5}}).checkpoint.every == 5


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -0.1])
def test_checkpoint_every_must_be_finite_and_nonnegative(value):
    with pytest.raises(ValueError, match="checkpoint"):
        RunConfig.model_validate({"checkpoint": {"every": value}})


@pytest.mark.parametrize("value", [0, -3])
def test_epochs_must_be_positive(value):
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"train": {"epochs": value}})


def test_train_requires_data(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "config.json", {})

    with pytest.raises(ValueError, match="requires a data"):
        load_train_config(config_path)


def test_prepare_requires_prepare(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "config.json", {})

    with pytest.raises(ValueError, match="requires a prepare"):
        load_prepare_config(config_path)
