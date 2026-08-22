"""Gallery: pseudobatch splines through a jump.

Recovers a smooth concentration curve from just 5 noisy measurements
straddling a discrete feed jump, checked against a known closed-form ground
truth. hybrax.format only: no reaction module, no training.

The ground-truth constants/function below are duplicated from
docs/source/_data/generate.py's build_demo_spline_jump (this example is
meant to be self-contained, not reach back into docs/).

See docs/source/gallery/pseudobatch_splines.md for the narrated version.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hybrax.format as hxf
from hybrax.format.splines import build_backtransform_spline, build_pseudobatch_transform

HERE = Path(__file__).parent

# Ground-truth constants (must match generate.py's build_demo_spline_jump).
SJ_K = 0.15
SJ_V0 = 1.0
SJ_C0 = 5.0
SJ_T_END = 17.0
SJ_T_JUMP = 10.0
SJ_DELTA_V_BOLUS = 0.15
SJ_C_FEED = 40.0


def spline_jump_truth(t) -> np.ndarray:
    """Exact concentration at time(s) t: closed-form on both sides of the bolus."""
    t = np.asarray(t, dtype=float)
    m_at_jump = SJ_C0 * SJ_V0 * np.exp(-SJ_K * SJ_T_JUMP)
    v_after = SJ_V0 + SJ_DELTA_V_BOLUS
    m_after = m_at_jump + SJ_DELTA_V_BOLUS * SJ_C_FEED
    pre = SJ_C0 * np.exp(-SJ_K * t)
    post = (m_after / v_after) * np.exp(-SJ_K * (t - SJ_T_JUMP))
    return np.where(t < SJ_T_JUMP, pre, post)


dense_t = np.linspace(0.0, SJ_T_END, 400)
truth = spline_jump_truth(dense_t)

collection = hxf.serialization.load_process_collection(HERE / "data.json")
process = collection.processes["run_1"]

solute = process.reactor_medium.components["solute"]
print("measurement times :", np.asarray(solute.concentration.times))
print("measured values   :", np.round(np.asarray(solute.concentration.values), 3))

bundle = build_pseudobatch_transform(process)
process.pseudobatch_transform = bundle
back = build_backtransform_spline(process, "solute")
recovered = np.asarray(back(dense_t))

meas_t = np.asarray(solute.concentration.times)
meas_v = np.asarray(solute.concentration.values)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(dense_t, truth, color="black", lw=1.5, label="ground truth")
ax.plot(dense_t, recovered, color="tab:blue", lw=1.5, ls="--",
        label="recovered (fit + backtransform)")
ax.scatter(meas_t, meas_v, color="tab:red", zorder=5, label="5 measurements")
ax.axvline(SJ_T_JUMP, color="gray", lw=0.8, ls=":")
ax.set_xlabel("t (h)")
ax.set_ylabel("solute (g/L)")
ax.legend()
fig.tight_layout()
out = HERE / "out"
out.mkdir(exist_ok=True)
fig.savefig(out / "recovery.png", dpi=110)
print(f"wrote {out / 'recovery.png'}")

pre = np.linspace(0.0, SJ_T_JUMP - 1.0, 200)
post = np.linspace(SJ_T_JUMP + 1.0, SJ_T_END, 200)
for label, grid in [("pre-jump [0, 9]", pre), ("post-jump [11, 17]", post)]:
    rec = np.asarray(back(grid))
    rel = np.abs(rec - spline_jump_truth(grid)) / spline_jump_truth(grid)
    print(f"{label}: max relative error {rel.max() * 100:4.1f}%   "
          f"mean {rel.mean() * 100:4.1f}%")
