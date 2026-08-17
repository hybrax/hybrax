from __future__ import annotations

import dataclasses
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from bp_format.dataclasses import (
    AugmentedBioProcess,
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    FeedVolumeChange,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)

from bp_train.harness import _resolve_estimated_scales
from bp_train.model_api import EstimatedScales
from bp_train.runtime_context import (
    SPLINE_SCALE_SAMPLE_COUNT,
    RuntimeDataContext,
    _series_scale_evidence,
    canonical_training_parents,
    original_parent_processes,
)
from bp_train.training_data import TrainingDataStore


def _process(
    name: str,
    *,
    feed_end: float,
    feed_cin: float,
    temperature: float,
    parent: str | None = None,
) -> BioProcess:
    process_type = AugmentedBioProcess if parent is not None else BioProcess
    kwargs = {"parent_process": parent} if parent is not None else {}
    return process_type(
        metadata=BioProcessMetadata(name=name, process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=1.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed": FeedVolumeChange(
                    name="feed",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(times=[0.0, 1.0], values=[0.0, feed_end]),
                    feed_medium=FeedMedium(
                        name="feed_medium",
                        density=1.0,
                        density_unit="kg/L",
                        components={
                            "biomass": FeedMediumComponent(
                                name="biomass",
                                unit="g/L",
                                concentration=StaticVariable(feed_cin),
                                is_controlled=False,
                            )
                        },
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="reactor_medium",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(times=[0.0, 1.0], values=[1.0, 0.8]),
                )
            },
        ),
        process_variables={
            "temperature": ProcessVariable(
                name="temperature",
                unit="K",
                is_controlled=True,
                values=StaticVariable(temperature),
            )
        },
        **kwargs,
    )


def _runtime_data(
    *,
    p0_feed_end: float = 1.0,
    p0_feed_cin: float = 10.0,
    child_value: float = 1000.0,
    held_out_value: float = 2000.0,
) -> tuple[BioProcessCollection, RuntimeDataContext]:
    collection = BioProcessCollection(
        processes={
            "P0": _process(
                "P0",
                feed_end=p0_feed_end,
                feed_cin=p0_feed_cin,
                temperature=300.0,
            ),
            "P0_aug": _process(
                "P0_aug",
                feed_end=child_value,
                feed_cin=child_value,
                temperature=child_value,
                parent="P0",
            ),
            "P1": _process("P1", feed_end=5.0, feed_cin=20.0, temperature=310.0),
            "P2": _process(
                "P2",
                feed_end=held_out_value,
                feed_cin=held_out_value,
                temperature=held_out_value,
            ),
        }
    )
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    return collection, RuntimeDataContext.from_collection(store, collection)


def test_original_parent_processes_keeps_all_non_augmented_processes():
    order = ("P0", "P0_aug", "P1", "P2")
    parents = (None, "P0", None, None)

    assert original_parent_processes(order, parents) == ("P0", "P1", "P2")


def test_canonical_training_parents_maps_children_and_preserves_parent_order():
    order = ("P0", "P0_aug", "P1", "P2", "P3")
    parents = (None, "P0", None, None, None)

    assert canonical_training_parents(order, parents, ("P1", "P0_aug", "P0")) == (
        "P0",
        "P1",
    )
    assert canonical_training_parents(order, parents, ("P0_aug",)) == ("P0",)
    assert canonical_training_parents(order, parents, ("P3", "P0_aug")) == (
        "P0",
        "P3",
    )
    with pytest.raises(KeyError, match="unknown selected process"):
        canonical_training_parents(order, parents, ("missing",))


def test_selected_parent_context_is_closed_and_parent_aligned():
    collection, runtime_data = _runtime_data()
    selected = runtime_data.select_training_parents(collection, ("P1", "P0_aug", "P0"))

    assert selected.process_order == ("P0", "P1")
    assert tuple(selected.training_parent_collection.processes) == ("P0", "P1")
    assert selected.augmentation_parents == (None, None)
    assert selected.training_data.t_measured.shape[0] == 2
    assert selected.controls_store.control_values.shape[0] == 2
    np.testing.assert_allclose(
        selected.training_data.Cin_controlled_FVCs[:, 0, 0], [10, 20]
    )
    np.testing.assert_allclose(
        selected.rhs_ode.Cin_controlled_FVCs,
        selected.training_data.Cin_controlled_FVCs[0],
    )
    np.testing.assert_allclose(
        selected.rhs_ode.Cin_modeled_FVCs,
        selected.training_data.Cin_modeled_FVCs[0],
    )
    np.testing.assert_allclose(
        selected.controls_store.min_V,
        np.asarray(runtime_data.controls_store.min_V)[[0, 2]],
    )

    with pytest.raises(KeyError, match="unknown process"):
        selected.training_data.get_process("P2")
    with pytest.raises(KeyError, match="unknown process"):
        selected.controls_store.get_controls("P0_aug")


def test_unselected_scale_context_does_not_expose_collection():
    _collection, runtime_data = _runtime_data()

    assert runtime_data.training_parent_collection is None
    with pytest.raises(ValueError, match="unavailable"):
        runtime_data.control_scale_evidence()


def test_selected_scale_context_exposes_only_deep_copied_parents():
    collection, runtime_data = _runtime_data()
    metadata = {
        "bp-train": {
            "process_order": ["P0", "P0_aug", "P1", "P2"],
            "processes": {name: {"marker": name} for name in collection.processes},
            "trusted_setting": {"processes": {"P2": "preserve"}},
        },
        "other": {"trusted": {"held_out_measurement": 123.0}},
    }
    collection = replace(collection, metadata=metadata)
    original_metadata = deepcopy(collection.metadata)

    selected = runtime_data.select_training_parents(collection, ("P0_aug", "P1"))

    parent_collection = selected.training_parent_collection
    assert parent_collection is not None
    assert tuple(parent_collection.processes) == ("P0", "P1")
    assert all(
        parent_collection.processes[name] is not collection.processes[name]
        for name in parent_collection.processes
    )
    assert parent_collection.metadata["bp-train"]["process_order"] == ["P0", "P1"]
    assert tuple(parent_collection.metadata["bp-train"]["processes"]) == (
        "P0",
        "P1",
    )
    assert (
        parent_collection.metadata["bp-train"]["trusted_setting"]
        == (original_metadata["bp-train"]["trusted_setting"])
    )
    assert (
        parent_collection.metadata["bp-train"]["trusted_setting"]
        is not (collection.metadata["bp-train"]["trusted_setting"])
    )
    assert parent_collection.metadata["other"] == original_metadata["other"]
    assert selected.control_scale_evidence().cumulative_FVCs
    assert collection.metadata == original_metadata


@pytest.mark.parametrize(
    "metadata",
    (
        {"bp-train": []},
        {"bp-train": {"processes": []}},
    ),
)
def test_selected_parent_context_rejects_malformed_structural_metadata(metadata):
    collection, runtime_data = _runtime_data()
    collection = replace(collection, metadata=metadata)

    with pytest.raises(ValueError, match="must be a mapping"):
        runtime_data.select_training_parents(collection, ("P0", "P1"))


def test_selected_stores_gather_every_process_aligned_array():
    collection, runtime_data = _runtime_data()
    selected = runtime_data.select_training_parents(collection, ("P0", "P1"))
    source_store = runtime_data.training_data
    selected_store = selected.training_data
    source_controls = runtime_data.controls_store
    selected_controls = selected.controls_store
    source_indices = np.asarray([0, 2])

    expected_training_fields = {
        "Cin_controlled_FVCs",
        "Cin_modeled_FVCs",
        "t_measured",
        "y_measured",
        "mask_measured",
        "n_measured",
        "y0_measured",
    }
    expected_control_fields = {
        "spline_breaks",
        "spline_coeffs",
        "linear_grid",
        "control_values",
        "control_derivatives",
        "jump_ts",
        "grid_lengths",
        "jump_ts_lengths",
        "min_V",
        "sample_event_times",
        "sample_event_volumes",
        "sample_event_mask",
        "bolus_event_times",
        "bolus_event_volumes",
        "bolus_event_Cin",
        "bolus_event_mask",
    }

    def process_aligned_fields(store):
        n_processes = len(store.process_order)
        return {
            field.name
            for field in dataclasses.fields(store)
            if hasattr((value := getattr(store, field.name)), "shape")
            and value.ndim > 0
            and value.shape[0] == n_processes
        }

    def assert_gathered_fields(source, result, field_names):
        for name in field_names:
            actual = np.asarray(getattr(result, name))
            expected = np.asarray(getattr(source, name))[source_indices]
            expected = expected[tuple(slice(0, size) for size in actual.shape)]
            np.testing.assert_array_equal(actual, expected, err_msg=name)

    assert process_aligned_fields(source_store) == expected_training_fields
    assert process_aligned_fields(source_controls) == expected_control_fields
    assert_gathered_fields(source_store, selected_store, expected_training_fields)
    assert_gathered_fields(source_controls, selected_controls, expected_control_fields)

    assert selected_store.t_measured.shape[1] == int(
        np.max(np.asarray(selected_store.n_measured))
    )
    expected_control_widths = {
        "max_grid_length": int(np.max(np.asarray(selected_controls.grid_lengths))),
        "max_spline_breaks": (
            int(
                np.max(
                    np.sum(
                        np.isfinite(np.asarray(selected_controls.spline_breaks)),
                        axis=1,
                    )
                )
            )
            if selected_controls.spline_breaks.shape[1]
            else 0
        ),
        "max_jump_ts_length": int(
            np.max(np.asarray(selected_controls.jump_ts_lengths))
        ),
        "max_sample_events": int(
            np.max(np.sum(np.asarray(selected_controls.sample_event_mask), axis=1))
        ),
        "max_bolus_events": int(
            np.max(np.sum(np.asarray(selected_controls.bolus_event_mask), axis=1))
        ),
    }
    assert {
        name: selected_controls.shape_metadata[name] for name in expected_control_widths
    } == expected_control_widths
    assert (
        selected_controls.linear_grid.shape[1]
        == expected_control_widths["max_grid_length"]
    )
    assert (
        selected_controls.spline_breaks.shape[1]
        == expected_control_widths["max_spline_breaks"]
    )
    assert (
        selected_controls.jump_ts.shape[1]
        == expected_control_widths["max_jump_ts_length"]
    )
    assert (
        selected_controls.sample_event_mask.shape[1]
        == expected_control_widths["max_sample_events"]
    )
    assert (
        selected_controls.bolus_event_mask.shape[1]
        == expected_control_widths["max_bolus_events"]
    )


def test_control_scale_evidence_excludes_children_and_held_out_parents():
    collection, runtime_data = _runtime_data()
    selected = runtime_data.select_training_parents(collection, ("P0_aug", "P1"))
    evidence = selected.control_scale_evidence()

    np.testing.assert_allclose(evidence.cumulative_FVCs[0], [0.0, 1.0, 0.0, 5.0])
    np.testing.assert_allclose(evidence.FVC_rates[0], [1.0, 5.0])
    np.testing.assert_allclose(evidence.PVs[0], [300.0, 310.0])
    np.testing.assert_allclose(evidence.controlled_FVC_Cin[:, 0, 0], [10.0, 20.0])
    assert np.max(evidence.cumulative_FVCs[0]) < 1000.0
    assert np.max(evidence.controlled_FVC_Cin) < 1000.0


def test_unselected_mutations_do_not_change_scale_evidence_but_selected_ones_do():
    def evidence(**kwargs):
        collection, runtime_data = _runtime_data(**kwargs)
        return runtime_data.select_training_parents(
            collection, ("P0_aug", "P1")
        ).control_scale_evidence()

    baseline = evidence()
    child_changed = evidence(child_value=9000.0)
    held_out_changed = evidence(held_out_value=8000.0)
    selected_changed = evidence(p0_feed_end=7.0, p0_feed_cin=70.0)

    for unselected_changed in (child_changed, held_out_changed):
        for baseline_traces, changed_traces in (
            (baseline.cumulative_FVCs, unselected_changed.cumulative_FVCs),
            (baseline.FVC_rates, unselected_changed.FVC_rates),
            (baseline.PVs, unselected_changed.PVs),
        ):
            for baseline_trace, changed_trace in zip(
                baseline_traces, changed_traces, strict=True
            ):
                np.testing.assert_allclose(baseline_trace, changed_trace)
        np.testing.assert_allclose(
            baseline.controlled_FVC_Cin, unselected_changed.controlled_FVC_Cin
        )
    assert not np.array_equal(
        baseline.cumulative_FVCs[0], selected_changed.cumulative_FVCs[0]
    )
    assert not np.array_equal(
        baseline.controlled_FVC_Cin, selected_changed.controlled_FVC_Cin
    )


def test_equivalent_training_selections_produce_identical_scale_evidence():
    collection, runtime_data = _runtime_data()

    from_parent = runtime_data.select_training_parents(collection, ("P0", "P1"))
    from_child = runtime_data.select_training_parents(collection, ("P1", "P0_aug"))

    parent_evidence = from_parent.control_scale_evidence()
    child_evidence = from_child.control_scale_evidence()
    for parent_traces, child_traces in (
        (parent_evidence.cumulative_FVCs, child_evidence.cumulative_FVCs),
        (parent_evidence.FVC_rates, child_evidence.FVC_rates),
        (parent_evidence.PVs, child_evidence.PVs),
    ):
        for parent_trace, child_trace in zip(parent_traces, child_traces, strict=True):
            np.testing.assert_allclose(parent_trace, child_trace)
    np.testing.assert_allclose(
        parent_evidence.controlled_FVC_Cin, child_evidence.controlled_FVC_Cin
    )
    np.testing.assert_allclose(
        parent_evidence.modeled_FVC_Cin, child_evidence.modeled_FVC_Cin
    )


def _with_rich_row_values(
    runtime_data: RuntimeDataContext, values: tuple[float, ...]
) -> RuntimeDataContext:
    """Make every producer-side process-aligned scale input distinguishable."""
    y0 = np.asarray(runtime_data.training_data.y0_measured).copy()
    y0[:] = np.asarray(values)[:, None]
    measured = np.asarray(runtime_data.training_data.y_measured).copy()
    measured[:] = np.asarray(values)[:, None, None]
    training_data = replace(
        runtime_data.training_data,
        y0_measured=y0,
        y_measured=measured,
    )

    def trace(value):
        return (np.asarray([value]), np.asarray([value]))

    return replace(
        runtime_data,
        training_data=training_data,
        process_time_bounds=tuple((value, value + 1.0) for value in values),
        modeled_volume_change_traces=tuple(((trace(value)),) for value in values),
        raw_state_traces=tuple(((trace(value)),) for value in values),
        sample_volume_event_traces=tuple(trace(value) for value in values),
        bound_snapshots=tuple(
            (("biomass", "state", 0, value, value + 1.0),) for value in values
        ),
    )


def _resolve_rich_scales(
    collection: BioProcessCollection,
    runtime_data: RuntimeDataContext,
    selected_processes: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    selected = runtime_data.select_training_parents(collection, selected_processes)

    def estimate_all_scales(data, _target_names, _config):
        evidence = data.control_scale_evidence()
        signature = sum(sum(bounds) for bounds in data.process_time_bounds)
        signature += float(np.sum(data.training_data.y0_measured))
        signature += float(np.sum(data.training_data.y_measured))
        for process_traces in data.modeled_volume_change_traces:
            for times, values in process_traces:
                signature += float(np.sum(times) + np.sum(values))
        for process_traces in data.raw_state_traces:
            for times, values in process_traces:
                signature += float(np.sum(times) + np.sum(values))
        for times, values in data.sample_volume_event_traces:
            signature += float(np.sum(times) + np.sum(values))
        for snapshot in data.bound_snapshots:
            for _, _, axis, lower, upper in snapshot:
                signature += axis
                signature += 0.0 if lower is None else lower
                signature += 0.0 if upper is None else upper
        for traces in (
            evidence.cumulative_FVCs,
            evidence.FVC_rates,
            evidence.PVs,
        ):
            signature += sum(float(np.sum(trace)) for trace in traces)
        signature += float(np.sum(evidence.controlled_FVC_Cin))
        signature += float(np.sum(evidence.modeled_FVC_Cin))
        scale = np.asarray([signature])
        return EstimatedScales(
            **{field.name: scale for field in dataclasses.fields(EstimatedScales)}
        )

    scales = _resolve_estimated_scales(
        custom_module=SimpleNamespace(estimate_all_scales=estimate_all_scales),
        runtime_data=selected,
        custom_cfg=None,
    )
    return tuple(
        np.asarray(getattr(scales, field.name).scale)
        for field in dataclasses.fields(EstimatedScales)
    )


def test_final_scales_use_only_selected_rich_parent_rows():
    def scales(*, rich_values, **process_values):
        collection, runtime_data = _runtime_data(**process_values)
        runtime_data = _with_rich_row_values(runtime_data, rich_values)
        return _resolve_rich_scales(collection, runtime_data, ("P0_aug", "P1"))

    baseline = scales(rich_values=(1.0, 1000.0, 2.0, 2000.0))
    child_changed = scales(rich_values=(1.0, 9000.0, 2.0, 2000.0), child_value=9000.0)
    held_out_changed = scales(
        rich_values=(1.0, 1000.0, 2.0, 8000.0), held_out_value=8000.0
    )
    selected_changed = scales(
        rich_values=(7000.0, 1000.0, 2.0, 2000.0),
        p0_feed_end=7.0,
        p0_feed_cin=70.0,
    )

    for unchanged in (child_changed, held_out_changed):
        for baseline_scale, unchanged_scale in zip(baseline, unchanged, strict=True):
            np.testing.assert_allclose(baseline_scale, unchanged_scale)
    assert any(
        not np.array_equal(baseline_scale, selected_scale)
        for baseline_scale, selected_scale in zip(
            baseline, selected_changed, strict=True
        )
    )


def test_equivalent_parent_and_child_selections_resolve_identical_scales():
    collection, runtime_data = _runtime_data()
    runtime_data = _with_rich_row_values(runtime_data, (1.0, 1000.0, 2.0, 2000.0))

    direct = _resolve_rich_scales(collection, runtime_data, ("P0", "P1"))
    loo_equivalent = _resolve_rich_scales(collection, runtime_data, ("P1", "P0_aug"))

    for direct_scale, loo_scale in zip(direct, loo_equivalent, strict=True):
        np.testing.assert_allclose(direct_scale, loo_scale)


def test_scale_series_uses_raw_samples_before_spline_and_samples_splines_at_200():
    raw_and_spline = TimeSeries(
        times=[0.0, 1.0],
        values=[2.0, 4.0],
        breaks=[0.0, 1.0],
        coeffs=[[100.0, 0.0, 0.0, 0.0]],
        segment_start_piece_idx=[0],
    )
    values, slopes = _series_scale_evidence(raw_and_spline, derivative=True)
    np.testing.assert_allclose(values, [2.0, 4.0])
    np.testing.assert_allclose(slopes, [2.0])

    spline_only = TimeSeries(
        breaks=[0.0, 0.5, 1.0],
        coeffs=[
            [0.75, 1.0, -1.0, 0.0],
            [1.0, 0.0, -1.0, 0.0],
        ],
        segment_start_piece_idx=[0, 1],
    )
    values, slopes = _series_scale_evidence(spline_only, derivative=True)
    grid = np.linspace(0.0, 1.0, SPLINE_SCALE_SAMPLE_COUNT)
    assert values.shape == slopes.shape == (SPLINE_SCALE_SAMPLE_COUNT,)
    np.testing.assert_allclose(values, 0.75 + grid - grid**2)
    np.testing.assert_allclose(slopes, 1.0 - 2.0 * grid)
    assert np.max(values) > max(values[0], values[-1])
