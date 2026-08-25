"""Fit a pole-free rational surrogate for e_coli_core's FBA solution.

Run once, offline, after 01_generate_fba_data.py. Method from Gotsmy &
Guillen-Gosalbez's FBA-Hyb (bioRxiv:10.64898/2026.04.22.720062v1): a
symbolic-regression-discovered rational functional form, its coefficients
re-fit here directly against fresh FBA data via multi-restart L-BFGS
(skipping symbolic regression itself, since the functional form is already
known):

    q_i = qG * (A1.n)(A2.n) / ( (pos(B1.n)+d) * (pos(B2.n)+d) )
    pos(B) = 0.5*(B + sqrt(B^2 + c)),  c = 1.5

Every denominator factor is >= d > 0 for all inputs, by construction: the
surrogate is pole-free, so it cannot blow up inside an ODE solver's adjoint
however far the trained reaction module's own predictions wander. This
script's boundedness certificate (below) checks that directly, over the full
sampling box, before accepting a fit.

n = (n_X, n_M, n_A, n_S). q_glc is analytic (= -qG exactly, by construction of
01_generate_fba_data.py's glucose-bound fixing). Fitted fluxes: biomass, ATPM,
EX_ac_e, EX_succ_e. Its output, 02_surrogate_body.txt, is the literal
`surrogate_fba` body pasted into fba_hyb_custom.py and pls_dfba_custom.py.
"""

import sys
import numpy as np
import pandas as pd
import scipy.optimize
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

C_RAMP = 1.5
VALIDATION_SPLIT = 0.1
N_RESTARTS = int(sys.argv[1]) if len(sys.argv) > 1 else 40

RID = {
    "qX": "BIOMASS_Ecoli_core_w_GAM",
    "qM": "ATPM",
    "qA": "EX_ac_e",
    "qS": "EX_succ_e",
}
FIT_FLUXES = ["qX", "qM", "qA", "qS"]
DELTA = {"qX": 0.05, "qM": 0.05, "qA": 0.05, "qS": 0.05}


def load():
    df = pd.read_csv("01_obj_fba_data.csv", index_col=0)
    avg = df[["qG", "n_X", "n_M", "n_A", "n_S"]].mean(axis=0)
    AVG_QG = float(avg["qG"])
    AVG_N = np.array([avg["n_X"], avg["n_M"], avg["n_A"], avg["n_S"]], float)

    qG_scaled = df["qG"].values / AVG_QG
    n_scaled = df[["n_X", "n_M", "n_A", "n_S"]].values / AVG_N
    Y = {k: df[v].values.astype(float) for k, v in RID.items()}
    return df, AVG_QG, AVG_N, qG_scaled, n_scaled, Y


def canonical_split(n_rows):
    key = jax.random.PRNGKey(13)
    ID = jax.random.choice(
        key, n_rows, shape=(int(n_rows * VALIDATION_SPLIT),), replace=False
    )
    ID = np.asarray(ID)
    valid = np.zeros(n_rows, bool)
    valid[ID] = True
    return ~valid, valid


def pos(B):
    return 0.5 * (B + jnp.sqrt(B * B + C_RAMP))


def aff(w, N):
    return N @ w[:4] + w[4]


def predict(Z, g, N, d):
    A1, A2 = aff(Z[0:5], N), aff(Z[5:10], N)
    B1, B2 = aff(Z[10:15], N), aff(Z[15:20], N)
    return g * A1 * A2 / ((pos(B1) + d) * (pos(B2) + d))


def r2(y, p):
    t = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum((y - p) ** 2) / t if t > 0 else 1.0


def nmae(y, p):
    m = np.mean(np.abs(y))
    return np.mean(np.abs(y - p)) / m if m > 0 else 0.0


def fit_flux(g, N, y, tr, d, n_restarts, seed):
    ytr = jnp.asarray(y[tr])
    gtr = jnp.asarray(g[tr])
    Ntr = jnp.asarray(N[tr])
    ay = jnp.abs(ytr)
    sc = float(np.mean(np.abs(y[tr]))) + 1e-12
    floor = max(float(np.percentile(np.abs(y[tr]), 5)), 0.02 * sc)
    eps2 = (0.01 * sc) ** 2
    REL_WEIGHT = 1.0

    def loss(Z):
        p = predict(Z, gtr, Ntr, d)
        sh = jnp.sqrt((ytr - p) ** 2 + eps2)
        return jnp.mean(sh) / sc + REL_WEIGHT * jnp.mean(sh / (ay + floor))

    vgj = jax.jit(jax.value_and_grad(loss))

    def vg(Z):
        v, gg = vgj(jnp.asarray(Z))
        return float(v), np.asarray(gg, float)

    rng = np.random.default_rng(seed)
    best, best_loss = None, np.inf

    def run(Z0):
        nonlocal best, best_loss
        try:
            r = scipy.optimize.minimize(
                vg,
                Z0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 3000, "maxfun": 6000, "ftol": 1e-15, "gtol": 1e-12},
            )
        except Exception:
            return
        if np.isfinite(r.fun) and r.fun < best_loss:
            best_loss, best = r.fun, r.x

    def num():
        return np.concatenate([rng.normal(0, 1, 4), [rng.normal(0, 0.3)]])

    def den():
        return np.concatenate([rng.normal(0.3, 0.6, 4), [rng.normal(0.5, 0.5)]])

    run(
        np.concatenate(
            [
                np.zeros(4),
                [0.1],
                np.zeros(4),
                [0.1],
                np.zeros(4),
                [1.0],
                np.zeros(4),
                [1.0],
            ]
        )
    )
    for _ in range(n_restarts):
        run(np.concatenate([num(), num(), den(), den()]))
    return best


def main():
    df, AVG_QG, AVG_N, g, N, Y = load()
    n_rows = len(df)
    tr, va = canonical_split(n_rows)
    print(f"rows={n_rows} train={tr.sum()} valid={va.sum()} AVG_QG={AVG_QG:.6f}")

    params = {}
    for nm in FIT_FLUXES:
        Z = fit_flux(
            g, N, Y[nm], tr, DELTA[nm], N_RESTARTS, seed=abs(hash(nm)) % 997 + 1
        )
        params[nm] = Z
        print(f"  fit {nm}")

    def predict_all(mask):
        gg, NN = jnp.asarray(g[mask]), jnp.asarray(N[mask])
        out = {"q_glc": -g[mask] * AVG_QG}
        for nm in FIT_FLUXES:
            out[nm] = np.asarray(predict(jnp.asarray(params[nm]), gg, NN, DELTA[nm]))
        return out

    pv = predict_all(va)
    print("\n" + "=" * 70)
    print(
        f"{'flux':5} | {'R2_all':>8} {'NMAE_all':>9} {'R2_low10':>9} {'NMAE_low10':>11}"
    )
    print("-" * 70)
    for nm in FIT_FLUXES:
        y = Y[nm][va]
        p = pv[nm]
        thr = np.percentile(np.abs(y), 10)
        m = np.abs(y) <= thr
        print(
            f"{nm:5} | {r2(y, p):8.4f} {nmae(y, p):9.4f} "
            f"{r2(y[m], p[m]):9.4f} {nmae(y[m], p[m]):11.4f}"
        )

    # Boundedness certificate over the physical sampling box.
    rng2 = np.random.default_rng(0)
    NB = 200000
    BOX_LO = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
    BOX_HI = np.array([20.0, 2.0, 2.0, 2.0, 2.0])
    U = rng2.uniform(size=(NB, 5)) * (BOX_HI - BOX_LO) + BOX_LO
    gb = jnp.asarray(U[:, 0] / AVG_QG)
    Nb = jnp.asarray(U[:, 1:5] / AVG_N)
    print("-" * 70)
    print(
        f"{'flux':5} | {'max|flux| box':>14} {'max|flux| data':>15} {'overshoot':>11}"
    )
    min_den = np.inf
    all_ok = True
    for nm in FIT_FLUXES:
        Z, d = jnp.asarray(params[nm]), DELTA[nm]
        dd = np.asarray((pos(aff(Z[10:15], Nb)) + d) * (pos(aff(Z[15:20], Nb)) + d))
        min_den = min(min_den, float(dd.min()))
        m_box = float(np.max(np.abs(np.asarray(predict(Z, gb, Nb, d)))))
        m_dat = float(np.max(np.abs(Y[nm])))
        ratio = m_box / m_dat if m_dat > 0 else np.inf
        ok = ratio <= 3.0
        all_ok &= ok
        print(
            f"{nm:5} | {m_box:14.1f} {m_dat:15.1f} "
            f"{ratio:10.1f}x | {'OK' if ok else 'MISS'}"
        )
    print("-" * 70)
    print(f"pole-free: min denominator over box = {min_den:.5f}")
    print(f"ALL PASS: {all_ok}")

    np.savez(
        "02_surrogate_params.npz",
        AVG_QG=AVG_QG,
        AVG_N=AVG_N,
        DELTA=np.array([DELTA[k] for k in FIT_FLUXES]),
        fit_fluxes=np.array(FIT_FLUXES),
        **{f"Z_{k}": params[k] for k in FIT_FLUXES},
    )

    def _fac(w):
        names = ["n_X", "n_M", "n_A", "n_S"]
        terms = [f"{w[j]:+.8g}*{names[j]}" for j in range(4)] + [f"{w[4]:+.8g}"]
        return " ".join(terms).lstrip("+").strip()

    lines = [
        "qG  = x_data[0]",
        "n_X = x_data[1]",
        "n_M = x_data[2]",
        "n_A = x_data[3]",
        "n_S = x_data[4]",
        "pos = lambda B: 0.5 * (B + jnp.sqrt(B * B + 1.5))",
        f"q_glc = {-AVG_QG:.8f} * qG",
    ]
    for nm in FIT_FLUXES:
        Z, d = params[nm], DELTA[nm]
        lines.append(
            f"{nm:4s} = qG * ({_fac(Z[0:5])}) * ({_fac(Z[5:10])})"
            f" / ((pos({_fac(Z[10:15])}) + {d}) * (pos({_fac(Z[15:20])}) + {d}))"
        )
    body = "\n".join(lines)
    with open("02_surrogate_body.txt", "w") as f:
        f.write(body + "\n")
    print("\n" + body)
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
