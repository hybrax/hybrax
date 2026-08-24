---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
<!-- LOCK -->
# Overview

> `hybrax-format` collects your bioprocess data in a single `BioProcessCollection` object and validates it for storing, sharing, and model training.

<!-- UNLOCK -->

## Under the hood

<img class="theme-diagram diagram-light" src="../_static/diagram_format_pipeline_light.svg" alt="Your input (ReactorMediumComponent, Volume/Inflow/Outflow, ProcessVariable, BiologicalOde) feeds into hybrax-format's derived objects: ProcessOrdering, ControlSplines, RhsOde, PseudobatchTransform.">
<img class="theme-diagram diagram-dark" src="../_static/diagram_format_pipeline_dark.svg" alt="Your input (ReactorMediumComponent, Volume/Inflow/Outflow, ProcessVariable, BiologicalOde) feeds into hybrax-format's derived objects: ProcessOrdering, ControlSplines, RhsOde, PseudobatchTransform.">

You write the left column once. The right column is generated, and is the single source
of truth for layout everywhere downstream: hybrax.train never re-derives it.

## The pages

| Page | Read it when |
|---|---|
| [The data model](data_model.md) | You are deciding which object holds which measurement. |
| [Loading and saving](load_and_save.md) | You need the on-disk format, or someone sent you a file. |
| [Validating and inspecting](validate_and_inspect.md) | Always. Before modeling anything. |
| [Volume, feeds and events](volume_feeds_events.md) | Your process is not a pure batch. |
| [Time series and splines](time_series_and_splines.md) | You need continuous interpolation of a measurement. |
| [The pseudobatch transform](pseudobatch_transform.md) | Your process has feeds and you need to remove dilution before fitting. |
| [The Bioprocess ODE](bioprocess_ode.md) | You want to see or change the assembled equations. |
| [Limits and gotchas](limits_and_gotchas.md) | Something you expected to work does not. |
| [Further reading](further_reading.md) | You want the exhaustive reference. |

## The shortest possible complete example

```{code-cell} ipython3
import numpy as np
import hybrax.format as hxf

process = hxf.BioProcess(
    metadata=hxf.BioProcessMetadata(name="run_1", process_type="batch"),
    time_axis=hxf.TimeAxis(unit="h", start=0.0, end=10.0,
                          time_reference="inoculation"),
    volume=hxf.Volume(initial_volume=1.0, unit="L"),
    reactor_medium=hxf.ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "biomass": hxf.ReactorMediumComponent(
                name="biomass", unit="g/L",
                concentration=hxf.TimeSeries(
                    times=np.array([0.0, 5.0, 10.0]),
                    values=np.array([0.1, 1.2, 4.0]),
                ),
            )
        },
    ),
)

ok, messages = hxf.validate_process(process)
print("valid:", ok)
print("auto-generated dynamics:", process.biological_ode.derivatives)
```

Four required arguments, one component, and you already have a valid process with a
generated ODE. Everything else in this guide is about the cases where that default is
not what you meant.

## Two conventions to carry with you

**Import as `hxf`.** Every dataclass is re-exported from the package root
(`hxf.BioProcess`, `hxf.TimeSeries`, …). Functions are grouped on module handles instead:
`hxf.serialization.*`, `hxf.validate.*`, `hxf.mechanistic.*`, `hxf.splines.*`. The
inspection helpers are the exception: `hxf.plot_process` and friends are on the root.

**Amounts, not concentrations, are what conserve.** hybrax.format tracks concentrations
because that is what you measure, but every transport term it generates is derived from
an amount balance. When something looks wrong, check the volume first.

## See also

- [Concepts and vocabulary](../start/concepts.md), if any term above was unfamiliar.
- [Tutorial 1](../tutorials/01_your_first_dataset.md): the same material as a walkthrough.
- [API reference](../autoapi/hybrax/format/index): every signature.
- [hybrax.train](../train/index.md) Training a model on your imported data.

