# %% [markdown]
# # Turing patterns driving surface growth
#
# This tutorial builds a minimal **pattern-driven growth** model: a
# reaction–diffusion system on the surface of a sphere spontaneously breaks
# symmetry into a spotty *Turing pattern*, and the concentration of one of the
# two species is then fed back into a geometric flow so that the surface grows
# *locally*, faster where the activator is high.
#
# The model is taken from B. Kovács, B. Li, C. Lubich, C.A. Power Guerra,
# Convergence of finite elements on an evolving surface driven by diffusion
# on the surface, Numer. Math. 137 (3) (2017) 643–689, doi:10/gh4xvx. and
# reads as follows:
#
# ```
# du/dt + u\nabla\cdot\mathbf{v} - Du * Lap_S u = gamma * (a - u + u^2 v)
# dv/dt + v\nabla\cdot\mathbf{v} - Dv * Lap_S v = gamma * (b - u^2 v)
# ```
#
# where `Lap_S` is the Laplace–Beltrami operator on the surface. The spatially
# uniform steady state is `u* = a + b`, `v* = b / (a+b)^2`. The scaling factor
# `gamma` sets the reaction rate relative to diffusion, and hence the pattern
# wavelength: linearising about `(u*, v*)` gives a critical wavenumber
# `k_c^2 = gamma * (Dv f_u + Du g_v) / (2 Du Dv)`. On the unit sphere the
# admissible modes are spherical harmonics with `-Lap_S Y_l = l(l+1) Y_l`, so
# the selected degree is roughly `l ~ sqrt(k_c^2)`. With the values below
# `k_c^2 ~ 30`, i.e. `l ~ 5` — a handful of spots, which is exactly what a
# `maxh = 0.2` mesh can resolve.
#
# **Parameters here were tuned empirically for this mesh and timescale.** The
# kinetic constants `a, b` are textbook.

# %%
import logging
import time

import matplotlib

logging.basicConfig(level=logging.INFO)

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
from ngsolve import *
from ngsolve.webgui import Draw

from cosmos.core.model import CosmosModel
from cosmos.pde import ADRBoundarySystemBDF1Model, GeometricalFlowModel
from cosmos.utils.generate_meshes import generate_boundary_sphere

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Schnakenberg kinetics
A_K, B_K = 0.1, 0.9
GAMMA = 80.0  # reaction scaling -> pattern wavelength
DU, DV = 1.0, 20.0  # Dv/Du = 20 is comfortably above the Turing threshold
MAXH = 0.2
DT = 0.004

U_STAR = A_K + B_K
V_STAR = B_K / (A_K + B_K) ** 2
SEED = 0

print(f"uniform steady state: u* = {U_STAR:.3f}, v* = {V_STAR:.3f}")

# %% [markdown]
# ## Turing conditions
#
# Before running anything, check the linear-stability conditions on the
# Jacobian of the kinetics at `(u*, v*)`: the well-mixed state must be stable
# (`tr J < 0`, `det J > 0`) *and* diffusion must destabilise it
# (`Dv f_u + Du g_v > 2 sqrt(Du Dv det J)`).

# %%
f_u = -1.0 + 2.0 * B_K / (A_K + B_K)
f_v = (A_K + B_K) ** 2
g_u = -2.0 * B_K / (A_K + B_K)
g_v = -((A_K + B_K) ** 2)

trJ = f_u + g_v
detJ = f_u * g_v - f_v * g_u
turing_lhs = DV * f_u + DU * g_v
turing_rhs = 2.0 * np.sqrt(DU * DV * detJ)
k_c2 = GAMMA * turing_lhs / (2.0 * DU * DV)

print(f"tr J  = {trJ:+.3f}  (must be < 0)")
print(f"det J = {detJ:+.3f}  (must be > 0)")
print(
    f"Dv*f_u + Du*g_v = {turing_lhs:.3f} > 2*sqrt(Du*Dv*detJ) = {turing_rhs:.3f}"
    f"  -> {turing_lhs > turing_rhs}"
)
print(
    f"critical k_c^2 = {k_c2:.1f}  ->  expected spherical-harmonic degree l ~ "
    f"{0.5 * (np.sqrt(1 + 4 * k_c2) - 1):.1f}"
)

# %% [markdown]
# ## Coupling the pattern to surface growth
#
# Put `GeometricalFlowModel` and `ADRBoundarySystemBDF1Model` on the
# same compartment and drive its
# normal velocity with the activator via the additive `rhs` forcing term. In
# `GeometricalFlowModel.Initialize()` the velocity row of the saddle-point
# system reads
# `<V, phi> - alpha*(...) - gamma_f*<kappa, phi> = <rhs, phi> + ...`,
# and `Solve()` refreshes `rhs` from its `Field` each step — so passing a
# *callable* wires the flow directly to the live Turing solution:
#
# ```python
# flow.set_params(rhs=lambda: G_GROW * turing.sol[0])
# ```
#
# `V > 0` means motion along the outward normal, so high `u` produces outward
# growth. Choices for the flow's own coefficients:
#
# * `alpha = 0` — no Willmore (bending) term. With `alpha = 1` the fourth-order
#   bending relaxation of an `l = 4` mode scales like `l^4`, which flattens
#   pattern-induced bumps far faster than they can form; the coupling becomes
#   invisible.
# * `beta = 0`, `gamma_f = 0.075` — a small mean-curvature-flow (surface
#   tension) term. It plays two roles: it regularises the mesh, and since
#   `V_kappa = gamma_f * kappa = -2*gamma_f/R` on a sphere, it *balances the
#   mean* growth `G_GROW * mean(u)`. The result is a shape that stops growing
#   overall and settles into a stationary lobed equilibrium — much more
#   informative than an ever-inflating ball.
#
# **ALE ordering.** A step runs `pdes_pre` -> ALE -> `pdes_post`. The flow is
# registered `ale_type=0` (pre): it computes `V` from the geometry and the `u`
# field that the previous step ended with, the ALE then moves the mesh, and the
# Turing system (`ale_type=1`, post) is solved on the *new* geometry. That is
# the right way round — the reaction–diffusion problem should see the surface
# it actually lives on, and the ADR model already handles the ALE mesh velocity
# internally (its advection field carries a `- ale.W` correction).
#
# Note that surface redistribution is on, so the mesh is not fixed to the material points.
# The ALE moves the mesh to follow the flow, and then the redistribution step repositions
# the mesh nodes to maintain the mesh quality.


# %%
def build_turing(model, comp):
    pde = model.create_pde("turing", ADRBoundarySystemBDF1Model, comp, ale_type=1, dim=2)
    pde.set_params(
        b_1=lambda: model.ale.V,
        b_2=lambda: model.ale.V,
        d_1=CF(DU),
        d_2=CF(DV),
        c_1=CF(GAMMA),
        rhs_1=CF(GAMMA * A_K),
        rhs_2=CF(GAMMA * B_K),
        u0_1=CF(U_STAR),
        u0_2=CF(V_STAR),
    )
    pde.add_nonlinearity(target=1, expression="-1*g*u1**2*u2", map={"g": CF(GAMMA)})
    pde.add_nonlinearity(target=2, expression="g*u1**2*u2", map={"g": CF(GAMMA)})
    return pde


def seed_perturbation(pde, amp=0.01, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(pde.sol[0].vec)
    pde.sol[0].vec.FV().NumPy()[:] = U_STAR + amp * rng.standard_normal(n)
    pde.sol[1].vec.FV().NumPy()[:] = V_STAR + amp * rng.standard_normal(n)


def vertex_points(model):
    c = model.ale.X.vec.FV().NumPy()
    return c.reshape((len(c) // 3, 3), order="F")


# %%
def sphere_views(fig, gs_ids, pts, values, cmap="inferno", label=""):
    """Two hidden-surface-removed 3D scatters + one Mollweide map."""
    r = np.linalg.norm(pts, axis=1)
    lat, lon = np.arcsin(pts[:, 2] / r), np.arctan2(pts[:, 1], pts[:, 0])
    vmin, vmax = values.min(), values.max()
    for gid, (el, az) in zip(gs_ids[:2], [(20, 30), (20, 210)]):
        ax = fig.add_subplot(*gid, projection="3d")
        e, a = np.radians(el), np.radians(az)
        view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        m = pts @ view > 0.0
        ax.scatter(
            pts[m, 0],
            pts[m, 1],
            pts[m, 2],
            c=values[m],
            cmap=cmap,
            s=55,
            vmin=vmin,
            vmax=vmax,
            depthshade=False,
        )
        ax.view_init(elev=el, azim=az)
        lim = 1.05 * np.abs(pts).max()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        ax.set_axis_off()
        ax.set_title(f"azim {az}deg", fontsize=9)
    ax = fig.add_subplot(*gs_ids[2], projection="mollweide")
    s = ax.tripcolor(lon, lat, values, cmap=cmap, shading="gouraud")
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("lon/lat (Mollweide)", fontsize=9)
    fig.colorbar(s, ax=ax, shrink=0.75, label=label)


# %%
G_GROW = 0.15  # normal velocity per unit of u
ALPHA_F = 0.0  # Willmore/bending: off (see above)
GAMMA_F = 0.075  # mean-curvature flow: regularises + balances mean growth

mesh = generate_boundary_sphere(maxh=MAXH, R=1.0)
model = CosmosModel(
    "turing_growth",
    mesh,
    t0=0.0,
    t1=3,
    dt=DT,
    t=Parameter(0.0),
    coupling_type="implicit",
    redistribute=True,
)
comp = model.create_compartment("surface", boundary="default", bboundary="")

flow = model.create_pde("flow", GeometricalFlowModel, comp, ale_type=0)
turing = build_turing(model, comp)

flow.set_params(
    rhs=lambda: G_GROW * turing.sol[0], alpha=CF(ALPHA_F), beta=CF(0.0), gamma=CF(GAMMA_F)
)

ale = model.create_ale("ale", compartment=comp)
ale.set_normal_velocity(lambda: flow.V_h)
ale.set_tangential_velocity(CF((0.0, 0.0, 0.0)))

t_start = time.time()
gen = model()
next(gen)
seed_perturbation(turing)

plot = Draw(turing.sol[0], mesh=model.parentmesh)
t_b, rng_u, r_min, r_mean, r_max = [], [], [], [], []
for _ in gen:
    plot.Redraw()
    u = turing.sol[0].vec.FV().NumPy()
    r = np.linalg.norm(vertex_points(model), axis=1)
    t_b.append(model.t.Get())
    rng_u.append(u.max() - u.min())
    r_min.append(r.min())
    r_mean.append(r.mean())
    r_max.append(r.max())

print(f"Part B: {len(t_b)} steps in {time.time() - t_start:.1f} s")

pts_b = vertex_points(model)
u_b = turing.sol[0].vec.FV().NumPy().copy()
r_b = np.linalg.norm(pts_b, axis=1)
corr = np.corrcoef(u_b, r_b)[0, 1]

print(
    f"final radius: min {r_b.min():.4f}, mean {r_b.mean():.4f}, max {r_b.max():.4f}"
    f"  (peak-to-valley {r_b.max() - r_b.min():.4f})"
)
print(f"final u:      min {u_b.min():.4f}, max {u_b.max():.4f}")
print(f"Pearson correlation between vertex u and vertex radius: {corr:+.4f}")

# %%
fig = plt.figure(figsize=(13, 6.5))
sphere_views(fig, [(2, 3, 1), (2, 3, 2), (2, 3, 3)], pts_b, u_b, label="u")

ax = fig.add_subplot(2, 3, 4)
ax.plot(t_b, r_min, label="min r")
ax.plot(t_b, r_mean, label="mean r")
ax.plot(t_b, r_max, label="max r")
ax.set_xlabel("t")
ax.set_ylabel("vertex radius")
ax.set_title("the sphere becomes lobed, not bigger", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = fig.add_subplot(2, 3, 5)
ax.plot(t_b, rng_u, lw=1.8, color="C3")
ax.set_xlabel("t")
ax.set_ylabel("max(u) - min(u)")
ax.set_title("pattern amplitude under growth", fontsize=9)
ax.grid(alpha=0.3)

ax = fig.add_subplot(2, 3, 6)
ax.scatter(u_b, r_b - r_b.mean(), s=12, alpha=0.65)
ax.set_xlabel("u at vertex")
ax.set_ylabel("r - mean(r) at vertex")
ax.set_title(f"local activator vs. local bulge (rho = {corr:+.3f})", fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle(f"Pattern-driven growth: V_normal = {G_GROW}*u + {GAMMA_F}*kappa")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "turing_growth_coupling.png"), dpi=110)
plt.close(fig)
print("saved figures/turing_growth_coupling.png")

# %% [markdown]
# ## What actually happened
#
# With `V = 0.15*u + 0.075*kappa` the surface bulges outward under
# every spot and pulls in between them. The mean radius barely moves (growth
# and surface tension balance), while the peak-to-valley radius spread reaches
# roughly 0.28 on an `R = 1` sphere — clearly visible lobes. The final scatter
# of vertex `u` against vertex radial displacement is close to a straight line
# (`rho ~ 0.99`), which is the quantitative statement of "the pattern drives
# the shape".
#
# ### Caveats, and what did not work
#
# * **This is a proxy, not biology.** There is no cell population, no
#   mechanics, no nutrient limitation, no volume constraint. Growth is an
#   ad-hoc normal velocity proportional to a chemical concentration. Nothing
#   here is validated against a real system, and the experiment has no closed-form
#   solution to check against — the success criterion is purely qualitative.
# * **`alpha = 1` (Willmore) destroys the effect.** The bending relaxation of
#   the selected mode is orders of magnitude faster than the growth forcing, so
#   the sphere stays round. Bending had to be switched off entirely.
# * **Unbalanced growth degrades the correlation.** With `gamma_f = 0` and pure
#   outward growth the sphere inflates; `k_c` is fixed, so as `R` increases the
#   surface admits a higher harmonic degree and the pattern *reorganises*
#   mid-run. The bumps then record where the spots *used to be*, and the
#   correlation drops. Balancing the mean growth against surface tension keeps
#   the domain size — and therefore the selected mode — fixed. This is a real
#   effect of growing-domain pattern formation, not a numerical artefact, but
#   it does mean the tidy `rho ~ 0.99` here is partly a consequence of choosing
#   a size-preserving regime.
# * **The reaction is explicit (IMEX).** `dt` must satisfy roughly
#   `dt * gamma < 1`; raising `gamma` to sharpen the pattern forces `dt` down
#   proportionally. A finer mesh is also needed as `gamma` grows, since the
#   selected wavelength shrinks like `1/sqrt(gamma)`.
# * **The pattern that forms depends on the random seed** (and on the mesh).
#   Turing systems on a sphere have many degenerate `m`-modes at the selected
#   `l`; a different seed rotates or reshuffles the spots. The *number* of
#   spots is robust; their placement is not.
