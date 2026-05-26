from pathlib import Path

from examples import validate_migrated_examples

EXPECTED_MIGRATED_EXAMPLES = (
    "examples/14_simulation_intracellular",
    "examples/01_kittler_2022",
)


def test_migrated_example_list_is_explicit():
    assert (
        validate_migrated_examples.MIGRATED_TARGET_EXAMPLES
        == EXPECTED_MIGRATED_EXAMPLES
    )
    for example in validate_migrated_examples.MIGRATED_TARGET_EXAMPLES:
        assert (validate_migrated_examples.REPO_ROOT / example).is_dir()


def test_list_command_prints_migrated_examples(capsys):
    assert validate_migrated_examples.main(["--list"]) == 0
    assert capsys.readouterr().out.splitlines() == list(EXPECTED_MIGRATED_EXAMPLES)


def test_migrated_target_layout_contracts_pass_for_repo_examples():
    for example in validate_migrated_examples.MIGRATED_TARGET_EXAMPLES:
        result = validate_migrated_examples.check_migrated_target_layout(
            validate_migrated_examples.REPO_ROOT / example
        )
        assert result["ok"] is True
        assert result["errors"] == []


def test_migrated_target_layout_contract_reports_missing_required(tmp_path):
    root = tmp_path / "repo"
    example = root / "examples/01_kittler_2022"
    example.mkdir(parents=True)
    result = validate_migrated_examples.check_migrated_target_layout(example)

    assert result["ok"] is False
    assert any("missing file" in error for error in result["errors"])


def test_migrated_target_layout_contract_reports_unexpected_artifact(tmp_path):
    root = tmp_path / "repo"
    example = root / "examples/01_kittler_2022"
    for relative_path, _ in validate_migrated_examples.MIGRATED_TARGET_LAYOUT_CONTRACTS[
        "examples/01_kittler_2022"
    ]["required"]:
        path = example / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    unexpected = example / "_target_generation.py"
    unexpected.write_text("# stale helper\n")

    result = validate_migrated_examples.check_migrated_target_layout(example)

    assert result["ok"] is False
    assert any(
        "_target_generation.py" in error
        and "unexpected target-layout artifact" in error
        for error in result["errors"]
    )


def test_delegates_to_shared_validator(monkeypatch):
    calls: list[list[str]] = []

    def fake_validate(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(
        validate_migrated_examples, "call_validate_example", fake_validate
    )

    exit_code = validate_migrated_examples.main(
        [
            "--check-structure",
            "--example",
            "examples/01_kittler_2022",
        ]
    )

    assert exit_code == 0
    assert calls == [
        [
            str(
                (
                    validate_migrated_examples.REPO_ROOT / "examples/01_kittler_2022"
                ).resolve()
            ),
            "--check-structure",
        ]
    ]


def test_delegates_all_examples_by_default(monkeypatch):
    calls: list[list[str]] = []

    def fake_validate(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(
        validate_migrated_examples, "call_validate_example", fake_validate
    )

    assert validate_migrated_examples.main([]) == 0
    assert [Path(call[0]).name for call in calls] == [
        "14_simulation_intracellular",
        "01_kittler_2022",
    ]


def test_failure_aggregation(monkeypatch):
    calls: list[list[str]] = []

    def fake_validate(argv: list[str]) -> int:
        calls.append(argv)
        return 1 if Path(argv[0]).name == "01_kittler_2022" else 0

    monkeypatch.setattr(
        validate_migrated_examples, "call_validate_example", fake_validate
    )

    assert validate_migrated_examples.main([]) == 1
    assert len(calls) == len(EXPECTED_MIGRATED_EXAMPLES)


def test_rejects_non_migrated_example(monkeypatch, capsys):
    def fake_validate(argv: list[str]) -> int:
        raise AssertionError(f"validator should not run for {argv}")

    monkeypatch.setattr(
        validate_migrated_examples, "call_validate_example", fake_validate
    )

    assert validate_migrated_examples.main(["--example", "examples/11_tub_2026"]) == 2
    assert "Unsupported example" in capsys.readouterr().err


def test_rejects_arbitrary_path(monkeypatch, capsys):
    def fake_validate(argv: list[str]) -> int:
        raise AssertionError(f"validator should not run for {argv}")

    monkeypatch.setattr(
        validate_migrated_examples, "call_validate_example", fake_validate
    )

    assert validate_migrated_examples.main(["--example", "README.md"]) == 2
    assert "Unsupported example" in capsys.readouterr().err
