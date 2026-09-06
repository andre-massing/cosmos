# %% [markdown]
# # 04 — Distance function on a disk
#
# `cosmos.pde.DistanceVolumeModel` computes a *smoothed approximation to the
# distance from a prescribed boundary* on a volume compartment. In a
# surface/level-set FEM framework this is a workhorse geometric quantity:
#
# * its zero level set is the prescribed boundary, and its level sets foliate the
#   compartment, so it is a ready-made level-set function for an existing domain;
# * its normalised gradient extends the boundary normal into the bulk, which is
#   exactly what you need to build a *normal velocity field* for an ALE motion or
#   to couple a surface PDE to a volume PDE;
# * it gives a cheap "how deep am I inside the domain" coordinate for
#   depth-dependent coefficients, boundary layers, or penalisation.
#
# The model exposes both of those: `pde.sol[0]` (`pde.gfu`) is the scalar
# distance field, `pde.sol[1]` (`pde.gfu2`) is the normalised-gradient
# ("velocity") field.
#
# ## The test case
#
# On a disk of radius `R` centred at the origin, the distance to the boundary is
# known in closed form,
#
# $$ d_{\text{exact}}(x, y) = R - \sqrt{x^2 + y^2}, $$
#
# zero on the boundary circle and positive inside — which is the sign convention
# this model produces (verified numerically below).
#
# ## A caveat worth stating up front: the solution is *not* smooth
#
# $\sqrt{x^2+y^2}$ has a **cone-point singularity at the origin**. It is Lipschitz
# but not differentiable there: $|\nabla d| = 1$ everywhere except at $r = 0$,
# where the gradient jumps direction by $180°$ across the point, and
# $\Delta d = -1/r$ blows up. This is the interior analogue of the classical
# corner singularity in elliptic problems, and it means we should *not* walk into
# this study expecting the textbook $O(h^{p+1})$ rate of a smooth solution. We
# will measure whatever rate actually comes out and then explain it.

# %% [markdown]
# ## How the model works, and what `dt` really is
#
# `DistanceVolumeModel.Solve()` is three linear solves, not a time loop
# (the model is fully stationary — registering it with `ale_type=-1` puts it in
# the "solved once, before everything else" list):
#
# 1. **Screened Poisson / Varadhan step.** Solve
#    $u - \delta\,\Delta u = 0$ with $u = 1$ on the zero-boundary. The solution is
#    a boundary layer $u \approx e^{-d/\sqrt{\delta}}$, whose level sets are
#    (to leading order) the level sets of the distance function.
# 2. **$L^2$ projection.** Project $\nabla u / |\nabla u|$ into a `VectorH1`
#    space, with the outward normal imposed on the boundary. This is `gfu2`, the
#    unit "outward direction" field — a smoothed $-\nabla d$.
# 3. **Poisson recovery.** Solve $-\Delta d_h = \nabla\!\cdot\!\texttt{gfu2}$ with
#    $d_h = 0$ on the boundary, which undoes the gradient and returns the
#    distance itself.
#
# In `Initialize()` the regularisation parameter is set as
#
# ```python
# dt = specialcf.mesh_size**2
# ```
#
# The name `dt` is a **red herring**: there is no time evolution anywhere in this
# model. It is the penalty/regularisation coefficient $\delta$ of step 1, and it
# has units of *length squared*. So the width of the boundary layer in step 1 is
# $\sqrt{\delta} = h$, the **local element size**.
#
# That choice is the single most important thing to understand about this model,
# and we will come back to it after the refinement study: because the smoothing
# length is tied to $h$, refining the mesh shrinks the layer *in lockstep* with
# the elements, so the layer is always resolved by about one element — no better
# on a fine mesh than on a coarse one.

# %%
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ngsolve import (
    CF,
    VOL,
    BilinearForm,
    GridFunction,
    Grad,
    H1,
    IfPos,
    InnerProduct,
    Integrate,
    LinearForm,
    Normalize,
    Trace,
    dx,
    grad,
    sqrt,
    x,
    y,
)

from cosmos.core.model import CosmosModel
from cosmos.pde import DistanceVolumeModel
from cosmos.utils.generate_meshes import generate_volume_circle

R = 1.0
GEO_ORDER = 3
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

d_exact = CF(R) - sqrt(x * x + y * y)
r_cf = sqrt(x * x + y * y)

t_start = time.time()


# %% [markdown]
# ## Driver
#
# `t0 = t1 = 0.0` with any positive `dt` gives exactly one pass through
# `CosmosModel`'s time loop (`while t <= t1` fires once at `t = 0`, then `t`
# advances past `t1`), which is all a stationary model needs. `ale_type=-1`
# registers the PDE in `pdes_init`, solved at the top of that single step.

# %%
def solve_distance(mesh, pde_class=DistanceVolumeModel, name="dist"):
    model = CosmosModel(name, mesh, t0=0.0, t1=0.0, dt=1.0)
    comp = model.create_compartment("disk", material="default", boundary="boundary")
    pde = model.create_pde(name, pde_class, comp, ale_type=-1, zero_bnd="boundary")
    model.run()
    return pde


def l2_error(pde, mesh, region=None):
    e2 = InnerProduct(pde.sol[0] - d_exact, pde.sol[0] - d_exact)
    if region is not None:
        e2 = IfPos(region, e2, 0.0)
    return np.sqrt(Integrate(e2, mesh, VOL))


def l2_norm_exact(mesh, region=None):
    n2 = InnerProduct(d_exact, d_exact)
    if region is not None:
        n2 = IfPos(region, n2, 0.0)
    return np.sqrt(Integrate(n2, mesh, VOL))


# %%
mesh = generate_volume_circle(maxh=0.15, order_g=GEO_ORDER, R=R, bnd_name="boundary")
pde = solve_distance(mesh)
gfu, gvel = pde.sol[0], pde.sol[1]

err = l2_error(pde, mesh)
print(f"baseline maxh=0.15, {mesh.ne} elements")
print(f"  d_h(0,0)   = {gfu(mesh(0.0, 0.0)):.5f}   (exact 1.00000)")
print(f"  d_h(0.5,0) = {gfu(mesh(0.5, 0.0)):.5f}   (exact 0.50000)")
print(f"  d_h(0,-0.9)= {gfu(mesh(0.0, -0.9)):.5f}   (exact 0.10000)")
print(f"  L2 error   = {err:.4e}  ({100 * err / l2_norm_exact(mesh):.2f} % relative)")
print(f"  || |grad-field| - 1 ||_L2 = "
      f"{np.sqrt(Integrate((sqrt(InnerProduct(gvel, gvel)) - 1) ** 2, mesh, VOL)):.4e}")

# %% [markdown]
# The field is positive inside and vanishes on the boundary, confirming the sign
# convention $d_h \approx R - r$.
#
# ## Baseline picture
#
# The mesh only covers the disk, so we sample `gfu(mesh(px, py))` on a Cartesian
# grid masked to $r < 0.97R$ and leave the rest as `nan`.

# %%
def sample(gf, n=220, rmax=0.97, comp=None):
    g = np.linspace(-R, R, n)
    X, Y = np.meshgrid(g, g)
    mask = X**2 + Y**2 < (rmax * R) ** 2
    vals = np.asarray(gf(mesh(X[mask], Y[mask])))
    out = np.full(X.shape, np.nan)
    out[mask] = vals[:, comp] if comp is not None else vals.ravel()
    return X, Y, out


X, Y, Zh = sample(gfu)
Ze = R - np.sqrt(X**2 + Y**2)
Ze[np.isnan(Zh)] = np.nan

fig, ax = plt.subplots(2, 2, figsize=(11, 9.5))

levels = np.linspace(0.0, R, 26)
c0 = ax[0, 0].contourf(X, Y, Zh, levels=levels, cmap="viridis")
gq = np.linspace(-R, R, 17)
Xq, Yq = np.meshgrid(gq, gq)
mq = Xq**2 + Yq**2 < (0.9 * R) ** 2
vq = np.asarray(gvel(mesh(Xq[mq], Yq[mq])))
ax[0, 0].quiver(Xq[mq], Yq[mq], vq[:, 0], vq[:, 1], color="w", scale=22, width=0.004)
ax[0, 0].set_title("computed $d_h$ (pde.sol[0])\nwhite arrows: pde.sol[1], unit outward field")
fig.colorbar(c0, ax=ax[0, 0])

c1 = ax[0, 1].contourf(X, Y, Ze, levels=levels, cmap="viridis")
ax[0, 1].set_title("exact $d = R - r$")
fig.colorbar(c1, ax=ax[0, 1])

c2 = ax[1, 0].contourf(X, Y, Zh - Ze, levels=25, cmap="coolwarm")
ax[1, 0].set_title("pointwise error $d_h - d$")
fig.colorbar(c2, ax=ax[1, 0])

for a in ax.ravel()[:3]:
    a.set_aspect("equal")
    a.set_xlabel("x")
    a.set_ylabel("y")

rr = np.linspace(0.0, 0.96 * R, 300)
prof = np.asarray(gfu(mesh(rr, 0.0 * rr))).ravel()
ax[1, 1].plot(rr, R - rr, "k--", lw=2, label="exact $R-r$ (kink at $r=0$)")
ax[1, 1].plot(rr, prof, "C1", lw=1.8, label="computed $d_h$")
ax[1, 1].set_xlabel("r")
ax[1, 1].set_ylabel("d")
ax[1, 1].set_title("radial profile along $y=0$")
axe = ax[1, 1].twinx()
axe.plot(rr, prof - (R - rr), "C3", lw=1.2, label="error (right axis)")
axe.axhline(0.0, color="0.7", lw=0.8)
axe.set_ylabel("$d_h - d$", color="C3")
axe.tick_params(axis="y", colors="C3")
h1, l1 = ax[1, 1].get_legend_handles_labels()
h2, l2 = axe.get_legend_handles_labels()
ax[1, 1].legend(h1 + h2, l1 + l2, fontsize=9, loc="lower left")

fig.suptitle(f"DistanceVolumeModel on a disk, maxh = 0.15 ({mesh.ne} elements)")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "distance_function_baseline.png"), dpi=130)
plt.close(fig)
print("wrote figures/distance_function_baseline.png")

# %% [markdown]
# ## Where does the error live?
#
# The cone point predicts that the error should be *denser* near the origin. We
# check by splitting the $L^2$ error into a disk $r < R/4$ and the rest, and
# normalising each by the area it covers to get an RMS error density.

# %%
a_in = Integrate(IfPos(0.25 * R - r_cf, CF(1.0), 0.0), mesh, VOL)
a_out = Integrate(IfPos(r_cf - 0.25 * R, CF(1.0), 0.0), mesh, VOL)
e_in = l2_error(pde, mesh, region=0.25 * R - r_cf)
e_out = l2_error(pde, mesh, region=r_cf - 0.25 * R)
print(f"RMS error density,  r < R/4 : {e_in / np.sqrt(a_in):.4e}")
print(f"RMS error density,  r > R/4 : {e_out / np.sqrt(a_out):.4e}")
print(f"ratio (centre / bulk)       : {(e_in / np.sqrt(a_in)) / (e_out / np.sqrt(a_out)):.2f}")

# %% [markdown]
# ## Refinement study
#
# We refine a single base mesh uniformly (`mesh.Refine()`, followed by
# `Curve()` so the new boundary nodes are projected back onto the circle). A
# *nested* family is used deliberately: independently generated Netgen meshes at
# `maxh = 0.4 / 1.5**k` give the same error magnitudes but with ±50 % mesh-to-mesh
# scatter, which at these rates is as large as the signal and makes successive
# two-level rates meaningless. Uniform refinement removes that variability, so
# whatever trend is left is real.
#
# Alongside the shipped model we run a variant that is identical except that the
# regularisation $\delta$ is held **fixed** instead of being tied to $h^2$. That
# is the controlled experiment for the "`dt` is a mesh-scaled smoothing length"
# claim above.

# %%
DELTA_FIXED = 0.15**2


class FixedRegularisationDistance(DistanceVolumeModel):
    """DistanceVolumeModel with delta held fixed instead of delta = mesh_size**2."""

    def Initialize(self):
        delta = CF(DELTA_FIXED)
        fes1 = H1(
            self.model.parentmesh,
            order=self.params["fes_order"],
            definedon=self.compartment.domain,
            dirichlet=self.params["zero_bnd"],
        )
        u1, v1 = fes1.TnT()
        self.A1 = BilinearForm(fes1)
        self.A1 += (u1 * v1 + delta * grad(u1) * grad(v1)) * dx(deformation=self.model.ale.Y)
        self.A1.Assemble()
        self.invA1 = self.A1.mat.Inverse(freedofs=fes1.FreeDofs())
        self.gfu1 = GridFunction(fes1)

        u2, v2 = self.fes2.TnT()
        self.A2 = BilinearForm(self.fes2)
        self.A2 += u2 * v2 * dx(deformation=self.model.ale.Y)
        self.A2.Assemble()
        self.invA2 = self.A2.mat.Inverse(freedofs=self.fes2.FreeDofs())
        self.F2 = LinearForm(self.fes2)
        self.F2 += Normalize(grad(self.gfu1)) * v2 * dx(deformation=self.model.ale.Y)

        u3, v3 = self.fes.TnT()
        self.A3 = BilinearForm(self.fes)
        self.A3 += grad(u3) * grad(v3) * dx(deformation=self.model.ale.Y)
        self.A3.Assemble()
        self.invA3 = self.A3.mat.Inverse(freedofs=self.fes.FreeDofs())
        self.F3 = LinearForm(self.fes)
        self.F3 += Trace(Grad(self.gfu2)) * v3 * dx(deformation=self.model.ale.Y)


# %%
NLEV = 5
rmesh = generate_volume_circle(maxh=0.35, order_g=GEO_ORDER, R=R, bnd_name="boundary")

hs, errs_h2, errs_fix = [], [], []
print(f"\n{'h':>8}{'ne':>8}{'L2 err (delta=h^2)':>22}{'rel %':>9}"
      f"{'|grad|-1':>12}{'L2 err (delta fixed)':>23}")
for lev in range(NLEV):
    if lev:
        rmesh.Refine()
        rmesh.Curve(GEO_ORDER)
    area = Integrate(CF(1.0), rmesh, VOL)
    h = np.sqrt(2.0 * area / rmesh.ne)

    p1 = solve_distance(rmesh, DistanceVolumeModel, "dh2")
    e1 = l2_error(p1, rmesh)
    v1 = p1.sol[1]
    gdef = np.sqrt(Integrate((sqrt(InnerProduct(v1, v1)) - 1) ** 2, rmesh, VOL))
    e2 = l2_error(solve_distance(rmesh, FixedRegularisationDistance, "dfix"), rmesh)
    rel = 100 * e1 / l2_norm_exact(rmesh)

    hs.append(h)
    errs_h2.append(e1)
    errs_fix.append(e2)
    print(f"{h:8.4f}{rmesh.ne:8d}{e1:22.4e}{rel:9.2f}{gdef:12.3e}{e2:23.4e}")

hs = np.array(hs)
errs_h2 = np.array(errs_h2)
errs_fix = np.array(errs_fix)

rates_h2 = np.log(errs_h2[:-1] / errs_h2[1:]) / np.log(2.0)
rates_fix = np.log(errs_fix[:-1] / errs_fix[1:]) / np.log(2.0)
print("\nfitted two-level rates, delta = h^2 (shipped): "
      + "  ".join(f"{r:+.2f}" for r in rates_h2))
print("fitted two-level rates, delta fixed         : "
      + "  ".join(f"{r:+.2f}" for r in rates_fix))
print(f"least-squares slope, delta = h^2 : {np.polyfit(np.log(hs), np.log(errs_h2), 1)[0]:+.2f}")
print(f"least-squares slope, delta fixed : "
      f"{np.polyfit(np.log(hs[1:]), np.log(errs_fix[1:]), 1)[0]:+.2f}   (levels 1..4)")

# %%
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.loglog(hs, errs_h2, "o-", lw=2, label=r"shipped model, $\delta = h^2$")
ax.loglog(hs, errs_fix, "s-", lw=2, color="C2", label=r"same model, $\delta$ fixed $= 0.15^2$")
ax.loglog(hs, errs_h2[0] * (hs / hs[0]) ** 1, "k--", lw=1, label=r"$O(h)$")
ax.loglog(hs, errs_h2[0] * (hs / hs[0]) ** 2, "k:", lw=1, label=r"$O(h^2)$")
ax.set_xlabel("h")
ax.set_ylabel(r"$\|d_h - d\|_{L^2(\Omega)}$")
ax.set_title("Distance function: convergence under uniform refinement")
ax.invert_xaxis()
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "distance_function_convergence.png"), dpi=130)
plt.close(fig)
print("wrote figures/distance_function_convergence.png")
print(f"\ntotal runtime: {time.time() - t_start:.1f} s")

# %% [markdown]
# ## What we actually observed
#
# **The shipped model does not converge to the exact distance function.** The
# $L^2$ error starts around 3–4 % relative on the coarsest mesh, improves by only
# a few percent on the first refinement, and is then flat — in fact very slightly
# *increasing* — across the remaining levels. The fitted two-level rates hover
# around $0$ (and go mildly negative), and the least-squares slope over the whole
# family is close to zero. This is not $O(h^2)$, and it is not even $O(h)$.
#
# It is also not a broken setup: the field has the right sign, vanishes on the
# boundary to machine accuracy, gives $d_h(0.5, 0) = 0.5002$ against an exact
# $0.5$, and its unit-gradient defect $\|\,|\texttt{sol[1]}| - 1\,\|_{L^2}$ *does*
# fall monotonically under refinement (0.20 → 0.027 across the five levels). The
# distance field is *accurate to a fixed few percent* — it just stops getting
# better.
#
# ### Why: `delta = mesh_size**2` is a mesh-scaled smoothing length
#
# The green curve is the controlled experiment. The *only* change is holding the
# regularisation $\delta$ fixed instead of tying it to $h^2$. With $\delta$ fixed
# the same three-solve algorithm converges cleanly at very nearly $O(h^2)$ over
# three orders of magnitude, because the boundary layer of step 1 now has a fixed
# physical width and the mesh eventually resolves it. (The first level is
# pre-asymptotic: there $h > \sqrt{\delta}$, the layer is not resolved at all, and
# the error is large.)
#
# With $\delta = h(x)^2$ the layer width is *identically the local element size*.
# Refining shrinks the layer exactly as fast as it shrinks the elements, so the
# layer is perpetually resolved by about one element and the relative
# discretisation error of step 1 is frozen. Worse, `specialcf.mesh_size` varies
# from element to element, so $\delta$ is a discontinuous, element-wise
# coefficient: neighbouring elements give the layer slightly different decay
# lengths, which tilts its level sets away from the true offset curves. Step 2
# normalises that gradient, so those tilts become $O(1)$-in-$h$ direction errors
# in `pde.sol[1]`, and step 3 integrates them into the distance. That is the
# mechanism behind the plateau, and it is why the plateau level depends on mesh
# quality (independently meshed families plateau anywhere between ~1 % and ~4 %).
#
# This is a deliberate engineering trade-off, not a bug: the field is meant to be
# a *smoothed* distance, cheap (three SPD solves, no iteration, no reinitialisation)
# and robust on any mesh, whose smoothing scale automatically follows the mesh.
# For its actual job — extending a boundary normal into the bulk to drive an ALE
# velocity — a few percent is fine. It is simply not a converging approximation of
# the exact distance, and should not be used as one.
#
# ### And the cone singularity
#
# The point singularity at $r = 0$ is a real, separate effect, and it is what
# limits the green curve. Three pieces of evidence:
#
# * The error-density split printed above: the RMS error in $r < R/4$ is about
#   **3x** the RMS error in the bulk, even though that disk holds only ~6 % of the
#   area. The error is concentrated at the cone point.
# * The radial-profile panel shows the computed field rounding off the kink of
#   $R - r$ at the origin — the computed $d_h(0,0)$ undershoots by several percent
#   while $d_h(0.5, 0)$ is accurate to $10^{-4}$.
# * The green curve converges at $O(h^2)$, one full order short of the $O(h^3)$
#   that continuous $P_2$ elements give for a smooth solution. Verified separately:
#   raising `fes_order` to 3 or 4 lowers the error constant but leaves the rate at
#   $\approx 2$. The cap is therefore a property of the solution, not of the
#   discretisation order — exactly the signature of a point singularity, the
#   interior analogue of the corner singularities that cap rates in elliptic
#   problems on non-convex polygons. Recovering a higher rate would require
#   grading the mesh towards the origin.
#
# So $O(h^2)$, not $O(h^{p+1})$, is the best this test case can do — and even that
# is only visible once the regularisation is decoupled from $h$.
#
# The practical takeaway: **the singularity caps the achievable rate at $O(h^2)$,
# but with the shipped $\delta = h^2$ it never gets the chance — the $h$-scaled
# regularisation freezes the accuracy at a few percent first.**
