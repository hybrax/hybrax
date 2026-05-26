import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from bp_format import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    BiologicalOde,
    FeedMedium,
    FeedVolumeChange,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.serialization import save_process_collection_json
from examples import validate_example


def _make_process(name="process_1", *, valid=True, component_names=("biomass",)):
    components = {}
    if valid:
        for index, component_name in enumerate(component_names):
            components[component_name] = ReactorMediumComponent(
                name=component_name,
                unit="g/L",
                concentration=TimeSeries(
                    times=jnp.array([0.0, 1.0, 2.0]),
                    values=jnp.array([0.1, 0.2, 0.4]) + index,
                ),
            )

    return BioProcess(
        metadata=BioProcessMetadata(name=name, process_type="batch"),
        time_axis=TimeAxis(
            unit="hours",
            start=0.0,
            end=2.0,
            time_reference="inoculation",
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components=components,
        ),
        process_variables={},
    )


def _write_collection(path: Path, processes):
    collection = BioProcessCollection(metadata=None, processes=processes)
    save_process_collection_json(collection, path)


def _make_example(
    tmp_path,
    *,
    single_processes=None,
    all_processes=None,
    name="01_example",
):
    root = tmp_path / name
    (root / "00_simulation").mkdir(parents=True)
    (root / "01_single_process" / "output").mkdir(parents=True)
    (root / "02_all_processes" / "output").mkdir(parents=True)
    (root / "03_validate").mkdir(parents=True)
    (root / "01_single_process" / "load_single_process.py").write_text("\n")
    (root / "02_all_processes" / "load_all_processes.py").write_text("\n")

    if single_processes is None:
        single_processes = {"process_1": _make_process("process_1")}
    if all_processes is None:
        all_processes = {"process_1": _make_process("process_1")}

    _write_collection(
        root / "01_single_process" / "output" / "data.json",
        single_processes,
    )
    _write_collection(
        root / "02_all_processes" / "output" / "data.json",
        all_processes,
    )
    return root


def _summary(root: Path):
    path = root / "03_validate" / "output" / "validation_summary.json"
    return json.loads(path.read_text())


def _write_simulation_dense_output(root: Path, content: str) -> Path:
    path = root / "00_simulation" / "simulation_dense_output.csv"
    path.write_text(content)
    return path


def _simulation_dense_output_for_process(
    process_id: str,
    state_columns: tuple[str, ...],
):
    rows = (
        validate_example.SimulationDenseOutputRow(
            process_id=process_id,
            time=0.0,
            row_type="online",
            states={name: 1.0 for name in state_columns},
            volume=1.0,
            line_number=2,
        ),
        validate_example.SimulationDenseOutputRow(
            process_id=process_id,
            time=1.0,
            row_type="online",
            states={name: 1.1 for name in state_columns},
            volume=1.0,
            line_number=3,
        ),
        validate_example.SimulationDenseOutputRow(
            process_id=process_id,
            time=2.0,
            row_type="online",
            states={name: 1.2 for name in state_columns},
            volume=1.0,
            line_number=4,
        ),
    )
    return validate_example.SimulationDenseOutput(
        path=Path("simulation_dense_output.csv"),
        state_columns=state_columns,
        processes={process_id: rows},
    )


def _flow_process(*, modeled_fvc_values=(0.0, 0.2, 0.4)):
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={},
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="flow", process_type="fed_batch"),
        time_axis=TimeAxis(
            unit="hours",
            start=0.0,
            end=2.0,
            time_reference="inoculation",
        ),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "controlled_feed": FeedVolumeChange(
                    name="controlled_feed",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.array([0.0, 1.0, 2.0]),
                        values=jnp.array([0.0, 0.1, 0.2]),
                    ),
                    feed_medium=feed_medium,
                ),
                "modeled_feed": FeedVolumeChange(
                    name="modeled_feed",
                    unit="L",
                    is_controlled=False,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.array([0.0, 1.0, 2.0]),
                        values=jnp.array(modeled_fvc_values),
                    ),
                    feed_medium=feed_medium,
                ),
                "modeled_sample": SampleVolumeChange(
                    name="modeled_sample",
                    unit="L",
                    is_controlled=False,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.array([0.0, 1.0, 2.0]),
                        values=jnp.array([0.0, -0.05, -0.1]),
                    ),
                ),
            },
        ),
        reactor_medium=ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.array([0.0, 1.0, 2.0]),
                        values=jnp.array([1.0, 1.1, 1.2]),
                    ),
                )
            },
        ),
        process_variables={
            "pH": ProcessVariable(
                name="pH",
                unit="1",
                is_controlled=True,
                values=StaticVariable(7.0),
            )
        },
        biological_ode=BiologicalOde(
            algebraic={},
            rates={"q_biomass": (None, None)},
            derivatives={"biomass": "q_biomass * biomass"},
        ),
    )


VALID_SIMULATION_DENSE_OUTPUT = """\
process_id,time,row_type,biomass,volume
process_1,0.0,online,0.1,1.0
process_1,1.0,online,0.2,1.0
process_1,2.0,online,0.4,1.0
"""


def test_check_structure_passes_and_creates_output_dirs(tmp_path, capsys):
    root = _make_example(tmp_path)

    exit_code = validate_example.main(["--check-structure", str(root)])

    assert exit_code == 0
    assert (root / "03_validate" / "output").is_dir()
    assert (root / "03_validate" / "output" / "plots").is_dir()
    summary = _summary(root)
    assert summary["ok"] is True
    assert summary["structure"]["ok"] is True
    capsys.readouterr()


def test_check_structure_fails_for_missing_required_file(tmp_path, capsys):
    root = _make_example(tmp_path)
    (root / "02_all_processes" / "load_all_processes.py").unlink()

    exit_code = validate_example.main(["--check-structure", str(root)])

    assert exit_code == 1
    summary = _summary(root)
    assert summary["ok"] is False
    assert any("load_all_processes.py" in p for p in summary["structure"]["missing"])
    capsys.readouterr()


def test_normal_run_defaults_to_real_and_writes_report_sections(tmp_path, capsys):
    root = _make_example(tmp_path)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 0
    summary = _summary(root)
    assert summary["config"]["values"]["kind"] == "real"
    assert summary["files"]["simulation_dense_output"] is None
    assert summary["single_process"]["ok"] is True
    assert summary["all_processes"]["ok"] is True
    assert summary["all_processes_case_study"]["ok"] is True
    assert summary["sparse_real_diagnostics"]["ok"] is True
    assert summary["sparse_real_diagnostics"]["warning_count"] > 0
    text = (root / "03_validate" / "output" / "validation.txt").read_text()
    assert "Sparse/real diagnostics:" in text
    assert "Single-process validation:" in text
    assert "All-processes consistency:" in text
    assert "RHS smoke:" in text
    capsys.readouterr()


def test_sparse_real_diagnostics_warnings_do_not_fail_cli(tmp_path, capsys):
    process = _make_process("process_1")
    component = process.reactor_medium.components["biomass"]
    component.concentration = TimeSeries(
        times=jnp.array([0.0]),
        values=jnp.array([0.1]),
    )
    root = _make_example(
        tmp_path,
        single_processes={"process_1": process},
        all_processes={"process_1": process},
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 0
    summary = _summary(root)
    section = summary["sparse_real_diagnostics"]
    assert section["ok"] is True
    assert section["status"] == "ok_with_warnings"
    assert section["warning_count"] > 0
    assert any("low point count" in warning for warning in section["warnings"])
    capsys.readouterr()


@pytest.mark.filterwarnings("ignore:.*Explicitly requested dtype.*:UserWarning")
def test_sparse_real_diagnostics_nonfinite_values_fail(tmp_path, capsys):
    process = _make_process("process_1")
    component = process.reactor_medium.components["biomass"]
    component.concentration = TimeSeries(
        times=jnp.array([0.0, 1.0, 2.0]),
        values=jnp.array([0.1, jnp.nan, 0.4]),
    )
    root = _make_example(
        tmp_path,
        single_processes={"process_1": process},
        all_processes={"process_1": process},
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    section = summary["sparse_real_diagnostics"]
    assert section["ok"] is False
    assert section["status"] == "failed"
    assert any("nonfinite" in error for error in section["errors"])
    capsys.readouterr()


def test_sparse_real_diagnostics_plot_created(tmp_path, capsys):
    root = _make_example(tmp_path)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 0
    section = _summary(root)["sparse_real_diagnostics"]
    plot_paths = section["processes"]["process_1"]["plot_paths"]
    assert len(plot_paths) == 1
    assert Path(plot_paths[0]).is_file()
    assert Path(plot_paths[0]).parent == root / "03_validate" / "output" / "plots"
    capsys.readouterr()


def test_sparse_real_diagnostics_does_not_parse_raw_csv(tmp_path, capsys):
    root = _make_example(tmp_path)
    original_data = root / "00_original_data"
    original_data.mkdir()
    (original_data / "poison.csv").write_text('not,a,validated,input\n"unterminated\n')

    exit_code = validate_example.main([str(root)])

    assert exit_code == 0
    assert _summary(root)["sparse_real_diagnostics"]["ok"] is True
    capsys.readouterr()


def test_sparse_real_diagnostics_scope_does_not_overclaim(tmp_path, capsys):
    root = _make_example(tmp_path)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 0
    section = _summary(root)["sparse_real_diagnostics"]
    assert "diagnostics only" in section["scope"]
    assert "no trajectory recovery" in section["scope"]
    text = (root / "03_validate" / "output" / "validation.txt").read_text()
    assert "reintegration" not in text.lower()
    assert "truth recovery" not in text.lower()
    capsys.readouterr()


def test_sparse_real_low_point_warning_includes_two_point_series():
    summary = {"kind": "time_series", "point_count": 2}
    warnings: list[str] = []

    validate_example.append_sparse_series_warning(warnings, "two_point", summary)

    assert warnings == ["two_point has low point count: 2 <= 2"]


def test_invalid_simulation_dense_output_fails_in_simulation_mode(
    tmp_path,
    capsys,
):
    root = _make_example(tmp_path)
    simulation_dense_output = _write_simulation_dense_output(
        root,
        "not,a,validated,milestone1,csv\n",
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    assert summary["config"]["values"]["kind"] == "simulation"
    assert summary["structure"]["simulation_dense_output_detected"] is True
    assert summary["files"]["simulation_dense_output"] == str(simulation_dense_output)
    assert summary["simulation_dense_output"]["ok"] is False
    assert summary["dense_event_validation"] == {
        "ok": False,
        "status": "skipped",
        "reason": "simulation dense output validation failed",
        "errors": ["simulation dense output validation failed"],
    }
    capsys.readouterr()


def test_dense_state_close_uses_tolerance_floor_for_small_states():
    assert validate_example.dense_state_close(3.5e-6, 3.0e-6)
    assert not validate_example.dense_state_close(5.0e-6, 3.0e-6)


def test_dense_event_warning_when_pre_event_online_check_never_runs():
    process = _make_process("process_1")
    process.volume.volume_changes["sample"] = SampleVolumeChange(
        name="sample",
        unit="L",
        is_controlled=True,
        is_continuous=False,
        values=TimeSeries(times=jnp.array([1.0]), values=jnp.array([-0.1])),
    )
    rows = (
        validate_example.SimulationDenseOutputRow(
            "process_1",
            0.0,
            "online",
            {"biomass": 0.1},
            1.0,
            2,
        ),
        validate_example.SimulationDenseOutputRow(
            "process_1",
            1.0,
            "pre-event",
            {"biomass": 0.2},
            1.0,
            3,
        ),
        validate_example.SimulationDenseOutputRow(
            "process_1",
            1.0,
            "post-event",
            {"biomass": 0.2},
            0.9,
            4,
        ),
        validate_example.SimulationDenseOutputRow(
            "process_1",
            2.0,
            "online",
            {"biomass": 0.4},
            0.9,
            5,
        ),
    )
    dense = validate_example.SimulationDenseOutput(
        path=Path("simulation_dense_output.csv"),
        state_columns=("biomass",),
        processes={"process_1": rows},
    )

    result = validate_example.validate_process_dense_events(
        "process_1",
        process,
        dense,
    )

    assert result["ok"] is True
    assert result["pre_event_online_checks"] == 0
    assert result["pre_event_online_checks_skipped"] == 1
    assert any(
        "pre-event/online comparison skipped" in warning
        for warning in result["warnings"]
    )


def test_real_config_with_simulation_dense_output_fails_setup(tmp_path, capsys):
    root = _make_example(tmp_path)
    _write_simulation_dense_output(root, VALID_SIMULATION_DENSE_OUTPUT)
    (root / "03_validate" / "config.json").write_text('{"kind": "real"}\n')

    exit_code = validate_example.main(["--check-structure", str(root)])

    assert exit_code == 1
    summary = _summary(root)
    assert summary["config"]["ok"] is False
    assert any("kind 'real'" in error for error in summary["config"]["errors"])
    capsys.readouterr()


def test_valid_simulation_dense_output_parser_mapping_and_repeated_timestamp_pass(
    tmp_path,
    capsys,
):
    root = _make_example(tmp_path)
    _write_simulation_dense_output(root, VALID_SIMULATION_DENSE_OUTPUT)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 0
    summary = _summary(root)
    assert summary["simulation_dense_output"]["ok"] is True
    assert summary["simulation_dense_output"]["row_count"] == 3
    assert summary["simulation_dense_output"]["state_columns"] == ["biomass"]
    assert summary["simulation_dense_output"]["process_ids"] == ["process_1"]
    assert summary["dense_event_validation"]["ok"] is True
    assert summary["dense_event_validation"]["event_checks"] == 0
    assert summary["dense_trajectory_validation"]["ok"] is True
    assert summary["dense_trajectory_validation"]["scope"] == (
        validate_example.DENSE_TRAJECTORY_SCOPE
    )
    assert "sparse_real_diagnostics" not in summary
    text = (root / "03_validate" / "output" / "validation.txt").read_text()
    assert "Simulation dense output:" in text
    assert "Dense event validation:" in text
    assert "Dense trajectory diagnostics:" in text
    capsys.readouterr()


def test_simulation_dense_output_parser_ignores_extra_columns(tmp_path, capsys):
    root = _make_example(tmp_path)
    _write_simulation_dense_output(
        root,
        "process_id,time,row_type,rate,biomass,volume,diagnostic\n"
        "process_1,0.0,online,9.0,0.1,1.0,ignored\n"
        "process_1,1.0,online,9.0,0.2,1.0,ignored\n"
        "process_1,2.0,online,9.0,0.4,1.0,ignored\n",
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 0
    summary = _summary(root)
    assert summary["simulation_dense_output"]["ok"] is True
    assert summary["simulation_dense_output"]["state_columns"] == ["biomass"]
    capsys.readouterr()


@pytest.mark.parametrize(
    ("content", "error_fragment"),
    [
        (
            "process_id,time,row_type,biomass\nprocess_1,0.0,online,0.1\n",
            "missing required columns",
        ),
        (
            "process_id,time,row_type,biomass,biomass,volume\n"
            "process_1,0.0,online,0.1,0.1,1.0\n",
            "duplicate columns",
        ),
        (
            "process_id,time,row_type,,volume\nprocess_1,0.0,online,0.1,1.0\n",
            "empty column names",
        ),
        (
            "process_id,time,row_type,biomass,volume\n"
            "process_1,0.0,online,0.1,1.0,extra\n",
            "fields; expected",
        ),
        (
            "process_id,time,row_type,biomass,volume\nprocess_1,0.0,online,0.1\n",
            "fields; expected",
        ),
        (
            "process_id,time,row_type,biomass,volume\n",
            "no data rows",
        ),
        (
            "process_id,time,row_type,biomass,volume\n,0.0,online,0.1,1.0\n",
            "empty process_id",
        ),
        (
            "process_id,time,row_type,biomass,volume\nprocess_1,0.0,bad,0.1,1.0\n",
            "unknown row_type",
        ),
        (
            "process_id,time,row_type,biomass,volume\nprocess_1,nan,online,0.1,1.0\n",
            "time must be finite",
        ),
        (
            "process_id,time,row_type,biomass,volume\nprocess_1,bad,online,0.1,1.0\n",
            "time must be numeric",
        ),
        (
            "process_id,time,row_type,biomass,volume\nprocess_1,0.0,online,bad,1.0\n",
            "state biomass must be numeric",
        ),
        (
            "process_id,time,row_type,biomass,volume\nprocess_1,0.0,online,0.1,inf\n",
            "volume must be finite",
        ),
        (
            "process_id,time,row_type,biomass,volume\nprocess_1,0.0,online,0.1,bad\n",
            "volume must be numeric",
        ),
        (
            "process_id,time,row_type,biomass,volume\nprocess_1,0.0,online,0.1,0.0\n",
            "volume must be positive",
        ),
        (
            "process_id,time,row_type,biomass,volume\n"
            "process_1,0.0,online,0.1,1.0\n"
            "process_1,0.0,online,0.2,1.0\n",
            "Duplicate simulation dense output row",
        ),
    ],
)
def test_simulation_dense_output_parser_failures(
    tmp_path,
    capsys,
    content,
    error_fragment,
):
    root = _make_example(tmp_path)
    _write_simulation_dense_output(root, content)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    assert summary["simulation_dense_output"]["ok"] is False
    assert any(
        error_fragment in error
        for error in summary["simulation_dense_output"]["errors"]
    )
    capsys.readouterr()


def test_modeled_flow_evaluator_and_sign_checks_cover_controlled_and_modeled_flows():
    process = _flow_process()
    ordering = validate_example.get_process_ordering(process)
    control_splines = validate_example.get_control_splines(process, ordering)
    modeled_flow_evaluator = validate_example.build_modeled_flow_evaluator(
        process,
        ordering,
    )

    u = control_splines(1.0)
    modeled_fvc, modeled_svc = modeled_flow_evaluator(1.0)
    checks, errors = validate_example.evaluate_flow_signs(
        ordering,
        u,
        modeled_fvc,
        modeled_svc,
    )

    assert checks == 3
    assert errors == []
    assert modeled_fvc.tolist() == pytest.approx([0.2])
    assert modeled_svc.tolist() == pytest.approx([-0.05])


def test_modeled_flow_sign_check_fails_for_negative_fvc_slope():
    process = _flow_process(modeled_fvc_values=(0.0, 0.2, -0.2))
    ordering = validate_example.get_process_ordering(process)
    modeled_flow_evaluator = validate_example.build_modeled_flow_evaluator(
        process,
        ordering,
    )
    u = validate_example.get_control_splines(process, ordering)(1.5)
    modeled_fvc, modeled_svc = modeled_flow_evaluator(1.5)

    _, errors = validate_example.evaluate_flow_signs(
        ordering,
        u,
        modeled_fvc,
        modeled_svc,
    )

    assert any("flow is negative" in error for error in errors)


def test_dense_trajectory_reports_rank_deficiency():
    process = BioProcess(
        metadata=BioProcessMetadata(name="rank", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours",
            start=0.0,
            end=2.0,
            time_reference="inoculation",
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.array([0.0, 1.0, 2.0]),
                        values=jnp.array([1.0, 1.1, 1.2]),
                    ),
                )
            },
        ),
        biological_ode=BiologicalOde(
            algebraic={},
            rates={"q1": (None, None), "q2": (None, None)},
            derivatives={"biomass": "q1 * biomass + q2 * biomass"},
        ),
    )
    dense = _simulation_dense_output_for_process("rank", ("biomass",))

    result = validate_example.validate_dense_trajectory(
        dense,
        BioProcessCollection(processes={"rank": process}),
    )

    assert result["ok"] is False
    assert result["processes"]["rank"]["min_rank"] == 1
    assert any("rank deficient" in error for error in result["errors"])


def test_dense_trajectory_reports_nonlinear_rate_expression():
    process = BioProcess(
        metadata=BioProcessMetadata(name="nonlinear", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours",
            start=0.0,
            end=2.0,
            time_reference="inoculation",
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.array([0.0, 1.0, 2.0]),
                        values=jnp.array([1.0, 1.1, 1.2]),
                    ),
                )
            },
        ),
        biological_ode=BiologicalOde(
            algebraic={},
            rates={"q1": (None, None)},
            derivatives={"biomass": "q1 * q1 * biomass"},
        ),
    )
    dense = _simulation_dense_output_for_process("nonlinear", ("biomass",))

    result = validate_example.validate_dense_trajectory(
        dense,
        BioProcessCollection(processes={"nonlinear": process}),
    )

    assert result["ok"] is False
    assert any("not affine" in error for error in result["errors"])


def test_simulation_dense_output_missing_state_column_fails(tmp_path, capsys):
    root = _make_example(
        tmp_path,
        all_processes={
            "process_1": _make_process(
                "process_1",
                component_names=("biomass", "glucose"),
            )
        },
    )
    _write_simulation_dense_output(
        root,
        "process_id,time,row_type,biomass,volume\nprocess_1,0.0,online,0.1,1.0\n",
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    assert summary["simulation_dense_output"]["ok"] is False
    assert any(
        "missing required columns" in error
        for error in summary["simulation_dense_output"]["errors"]
    )
    capsys.readouterr()


def test_simulation_dense_output_process_id_mismatch_fails(tmp_path, capsys):
    root = _make_example(
        tmp_path,
        all_processes={
            "process_1": _make_process("process_1"),
            "process_2": _make_process("process_2"),
        },
    )
    _write_simulation_dense_output(
        root,
        "process_id,time,row_type,biomass,volume\n"
        "process_1,0.0,online,0.1,1.0\n"
        "process_extra,0.0,online,0.1,1.0\n",
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    section = summary["simulation_dense_output"]
    assert section["missing_in_dense"] == ["process_2"]
    assert section["extra_in_dense"] == ["process_extra"]
    assert any("missing process IDs" in error for error in section["errors"])
    assert any("unknown process IDs" in error for error in section["errors"])
    capsys.readouterr()


def test_simulation_dense_output_row_semantics_reject_orphan_event_and_offline_rows(
    tmp_path,
    capsys,
):
    root = _make_example(tmp_path)
    _write_simulation_dense_output(
        root,
        "process_id,time,row_type,biomass,volume\n"
        "process_1,0.0,online,0.1,1.0\n"
        "process_1,1.0,online,0.2,1.0\n"
        "process_1,1.0,pre-event,0.2,1.0\n"
        "process_1,1.0,offline,0.2,1.0\n"
        "process_1,2.0,online,0.4,1.0\n",
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    errors = _summary(root)["simulation_dense_output"]["errors"]
    assert any("non-event time" in error for error in errors)
    assert any("non-sample time" in error for error in errors)
    capsys.readouterr()


def test_invalid_config_unknown_key_or_invalid_kind_fails(tmp_path, capsys):
    invalid_configs = ({"unexpected": True}, {"kind": "experiment"}, [])
    for index, config in enumerate(invalid_configs):
        root = _make_example(tmp_path, name=f"01_example_{index}")
        (root / "03_validate" / "config.json").write_text(json.dumps(config))

        exit_code = validate_example.main(["--check-structure", str(root)])

        assert exit_code == 1
        summary = _summary(root)
        assert summary["config"]["ok"] is False
        assert summary["config"]["errors"]
        capsys.readouterr()


def test_single_process_json_must_contain_exactly_one_process(tmp_path, capsys):
    root = _make_example(
        tmp_path,
        single_processes={
            "process_1": _make_process("process_1"),
            "process_2": _make_process("process_2"),
        },
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    errors = summary["loaded_files"]["single_process"]["errors"]
    assert any("exactly one process" in error for error in errors)
    capsys.readouterr()


def test_single_process_json_must_be_member_of_all_processes(tmp_path, capsys):
    root = _make_example(
        tmp_path,
        single_processes={"process_1": _make_process("process_1")},
        all_processes={"process_2": _make_process("process_2")},
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    errors = summary["loaded_files"]["all_processes"]["errors"]
    assert any("must be present" in error for error in errors)
    capsys.readouterr()


def test_all_process_json_must_not_be_empty(tmp_path, capsys):
    root = _make_example(tmp_path, all_processes={})

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    errors = summary["loaded_files"]["all_processes"]["errors"]
    assert any("one or more processes" in error for error in errors)
    capsys.readouterr()


def test_same_time_dense_events_are_sorted_sample_before_bolus():
    events = [
        {"kind": "bolus_feed"},
        {"kind": "sample"},
    ]

    sorted_kinds = [
        event["kind"]
        for event in sorted(events, key=validate_example.dense_event_sort_key)
    ]

    assert sorted_kinds == ["sample", "bolus_feed"]


def test_structural_validation_failure_fails_and_writes_reports(tmp_path, capsys):
    root = _make_example(
        tmp_path,
        single_processes={"bad": _make_process("bad", valid=False)},
        all_processes={"bad": _make_process("bad", valid=False)},
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary_path = root / "03_validate" / "output" / "validation_summary.json"
    text_path = root / "03_validate" / "output" / "validation.txt"
    assert summary_path.is_file()
    assert text_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["single_process"]["ok"] is False
    assert summary["all_processes"]["ok"] is False
    text = text_path.read_text()
    assert "Reactor medium has no components" in text
    capsys.readouterr()


def test_rhs_skip_helper_reports_exact_message_and_ok():
    process = _make_process("process_1")
    process.biological_ode = None

    result = validate_example.run_rhs_smoke({"process_1": process})

    assert result["ok"] is True
    process_result = result["processes"]["process_1"]
    assert process_result["status"] == "skipped"
    assert process_result["message"] == validate_example.RHS_SKIP_MESSAGE


def test_rhs_smoke_failure_helper_is_failure(monkeypatch):
    def fail_build_rhs_ode(process):
        raise RuntimeError(f"boom for {process.metadata.name}")

    monkeypatch.setattr(validate_example, "build_rhs_ode", fail_build_rhs_ode)

    result = validate_example.run_rhs_smoke({"process_1": _make_process("process_1")})

    assert result["ok"] is False
    process_result = result["processes"]["process_1"]
    assert process_result["status"] == "failed"
    assert "boom for process_1" in process_result["message"]


def test_rhs_smoke_failure_in_normal_run_fails_and_writes_reports(
    tmp_path,
    capsys,
    monkeypatch,
):
    root = _make_example(tmp_path)

    def fail_build_rhs_ode(process):
        raise RuntimeError(f"boom for {process.metadata.name}")

    monkeypatch.setattr(validate_example, "build_rhs_ode", fail_build_rhs_ode)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    assert summary["ok"] is False
    process_result = summary["rhs_smoke"]["single_process"]["processes"]["process_1"]
    assert process_result["status"] == "failed"
    assert "boom for process_1" in process_result["message"]
    text = (root / "03_validate" / "output" / "validation.txt").read_text()
    assert "RHS smoke:" in text
    assert "boom for process_1" in text
    capsys.readouterr()
