"""Tutorial 2: before modeling anything, make the package tell you what it
understood from your data.

See docs/source/tutorials/02_look_at_it.md for the narrated version.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import hybrax.format as hxf

HERE = Path(__file__).parent

collection = hxf.serialization.load_process_collection(HERE / "data.json")
process = collection.processes["run_1"]

# --- 2.1 Validate ------------------------------------------------------------
ok, messages = hxf.validate_process(process)
print("ok:", ok)
for line in messages:
    print(" ", line)

ok, per_process = hxf.validate_for_publication(collection)
print("ok:", ok)
print("checked:", list(per_process))

# --- 2.2 Print the structure -------------------------------------------------
hxf.print_process_structure(process, verbosity=2)

# --- 2.3 Plot a process -------------------------------------------------------
fig = hxf.plot_process(process)
out = HERE / "out"
out.mkdir(exist_ok=True)
fig.savefig(out / "run_1.png")
print(f"wrote {out / 'run_1.png'}")

# --- 2.4 Print the ODE ---------------------------------------------------------
hxf.print_rhs_ode(process)
