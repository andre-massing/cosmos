"""Application 1: single-cell migration driven by front/back polarity.

The cell outline is a closed 1D curve (the boundary of a 2D cross-section)
carrying two competing, mutually-inhibiting chemical species:
``u1`` ("A", a front/protrusive marker) and ``u2`` ("B", a back/retraction
marker). Their local concentrations set the membrane's normal velocity
through a Hill-function competition in ``V()``, so wherever A dominates the
membrane advances and wherever B dominates it retreats -- a minimal
front-back polarity model of a migrating cell, in the spirit of
activator/inhibitor (e.g. Rac/Rho- or PIP3/PTEN-like) polarity models.

Each reaction term also couples back to the interface geometry through
``kappa*V*ns*u`` -- the local curvature-weighted normal velocity feeds into
the species' own kinetics, which is what lets a moving boundary and its
chemistry co-evolve self-consistently.

The script runs the same model twice over a 21x21 grid of the two
cross-inhibition strengths ``a12``/``a21``:

1. First loop -- fixed activation threshold ``A0``: a baseline polarity
   model with no mechanochemical feedback from cell shape.
2. Second loop -- ``Tension()`` computes a Hill function of the current
   enclosed area relative to the rest area ``Area0``, and that "tension"
   rescales the front species' activation threshold (``A0*T`` instead of a
   fixed ``A0``). This is the paper's membrane-tension-mediated feedback:
   as the cell spreads (area grows), rising tension raises the bar for
   protrusion, closing a shape -> signaling -> shape loop. Note this loop
   also swaps in the non-stabilised ADR solver (``SOLVER`` uses the
   gradient-jump-stabilised variant only for the first, feedback-free loop).

In both loops, a short unstimulated run (``t < 10*dt``) lets the two species
relax, after which ``B`` is knocked down on one half of the cell
(``x > 0``, via the smoothed step ``1 - 1/(1+exp(-5x))``) to break the
front/back symmetry -- the standard way to seed spontaneous polarization in
this class of model rather than relying on numerical noise alone.
"""


import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from ngsolve import *
from cosmos import *
from cosmos.utils.generate_meshes import generate_boundary_1D_circle
from cosmos.core.model import CosmosModel
from cosmos.pde import (
    ADRBoundarySystemBDF1Model,
    ADRBoundarySystemBDF1StabModel
)

SOLVER = ADRBoundarySystemBDF1StabModel  # gradient-jump stabilisation for the feedback-free sweep below
ROOT = '.'

# 20x20 grid of cross-inhibition strengths (A inhibiting B, B inhibiting A);
# endpoint 0 is dropped since a12=a21=0 decouples the two species entirely.
a12s = np.round(np.linspace(0, 3, 21, endpoint=True), 2)[1:]
a21s = np.round(np.linspace(0, 3, 21, endpoint=True), 2)[1:]

# --- Sweep 1: baseline polarity model, fixed activation threshold A0 -------
for a12 in a12s:
    for a21 in a21s:
        dt = 5e-2
        t0 = 0
        t1 = 100

        kappa1 = 0.4  # strength of the curvature/velocity feedback into A's kinetics
        kappa2 = 0.4  # strength of the curvature/velocity feedback into B's kinetics
        m = 4  # Hill coefficient for the A/B competition driving V()
        s = 8  # Hill coefficient for Tension() (used only in the feedback sweep below)
        r1 = 1  # linear decay rate of A
        r2 = 1  # linear decay rate of B
        klim_A = 1  # self-limiting (quadratic) production rate of A
        klim_B = 1  # self-limiting (quadratic) production rate of B
        A0 = 2.2  # half-activation threshold for the front species A
        B0 = 2.2  # half-activation threshold for the back species B

        R = 10  # resting radius of the idealized circular cell outline
        mesh = generate_boundary_1D_circle(r=R, N=100)
        ns = specialcf.normal(2)

        model_name = f"Application_1_a12_{a12}_a21_{a21}"
        model = CosmosModel(
            name=model_name,
            parentmesh=mesh,
            dt=dt,
            t0=t0,
            t1=t1,
            root=ROOT,
            samples=100,
            coupling_type="implicit",
            redistribute=True,
            adaptive_timestep=True,
        )

        comp1 = model.create_compartment(name="compartment", boundary="default", bboundary="")

        adr_system = model.create_pde(
            name="adr_system",
            pde_model=SOLVER,
            compartment=comp1,
            ale_type=1,
            dim=2,
        )

        ale = model.create_ale(name="ale", compartment=comp1)

        # Membrane normal velocity: a smooth "winner take all" competition
        # between the front marker A and the back marker B. V > 0 (A wins)
        # advances the membrane, V < 0 (B wins) retracts it.
        def V():
            term1 = (adr_system.sol[0]) ** m / ((adr_system.sol[0]) ** m + A0**m)
            term2 = (adr_system.sol[1]) ** m / ((adr_system.sol[1]) ** m + B0**m)
            return term1 - term2

        ale.set_normal_velocity(V)
        ale.set_tangential_velocity(CF((0, 0)))

        # Reaction kinetics for A: quadratic self-production, cross-inhibition
        # from B (a12), and a curvature/velocity feedback term that couples
        # A's own dynamics to how fast/curved the membrane it sits on is moving.
        adr_system.add_nonlinearity(
            target=1,
            expression="klim_A*u1**2 + a12*u1*u2 - kappa1*V*ns*u1",
            map={
                "kappa1": kappa1,
                "klim_A": klim_A,
                "a12": a12,
                "V": model.ale.V,
                "ns": specialcf.normal(2),
            },
        )
        # Mirror image for B, with the opposite sign on the velocity feedback
        # term (kappa2 has a "+" here vs. "-" for A) so the two species are
        # pushed in opposite directions by the same interface motion.
        adr_system.add_nonlinearity(
            target=2,
            expression="klim_B*u2**2 + a21*u1*u2 + kappa2*V*ns*u2",
            map={
                "kappa2": kappa2,
                "klim_B": klim_B,
                "a21": a21,
                "V": model.ale.V,
                "ns": specialcf.normal(2),
            },
        )

        # Linear part of the ADR system: both species diffuse (d=0.1), decay
        # linearly (c=-r), are advected by the ALE mesh velocity itself
        # (b=model.ale.V, i.e. they are carried along with the moving
        # membrane), and are kept non-negative (bounds=[0, inf)).
        adr_system.set_params(
            u0_1=0.1,
            d_1=0.1,
            c_1=-r1,
            b_1=lambda: model.ale.V,
            bounds_1=[0, 1e100],
            u0_2=1,
            d_2=0.1,
            c_2=-r2,
            b_2=lambda: model.ale.V,
            bounds_2=[0, 1e100],
            printing=True,
        )

        # Per-step diagnostics: total mass of each species, pointwise samples,
        # the cell's centroid (x_bary/y_bary track net migration distance),
        # enclosed area/volume, and the concentration range of each species.
        output_callables = {
            "mass_A": lambda: Integrate(adr_system.sol[0], mesh, VOL_or_BND=BND),
            "mass_B": lambda: Integrate(adr_system.sol[1], mesh, VOL_or_BND=BND),
            "A_point": lambda: adr_system.sol[0].vec[0],
            "B_point": lambda: adr_system.sol[1].vec[0],
            "normal_velocity": lambda: ale.gfu_norm_vel.vec[0],
            "x_bary": lambda: (
                Integrate(x, mesh, VOL_or_BND=BND) / Integrate(1, mesh, VOL_or_BND=BND)
            ),
            "y_bary": lambda: (
                Integrate(y, mesh, VOL_or_BND=BND) / Integrate(1, mesh, VOL_or_BND=BND)
            ),
            "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
            "volume": lambda: Integrate(CF((x, 0)) * ns, mesh, VOL_or_BND=BND),
            "min_A": lambda: np.min(adr_system.sol[0].vec.FV().NumPy()),
            "max_A": lambda: np.max(adr_system.sol[0].vec.FV().NumPy()),
            "min_B": lambda: np.min(adr_system.sol[1].vec.FV().NumPy()),
            "max_B": lambda: np.max(adr_system.sol[1].vec.FV().NumPy()),
        }

        model.set_params(output_callables=output_callables)

        # After a brief relaxation (10 steps), knock down B on the right half
        # of the cell (x > 0) to break front/back symmetry and seed
        # polarization -- otherwise the symmetric initial condition would
        # stay symmetric (up to numerical noise) and the cell would not move.
        depleted = False
        for _ in model():
            if model.t.Get() >= 10 * dt and not depleted:
                adr_mod = GridFunction(H1(mesh))
                adr_mod.Set(adr_system.sol[1], definedon=comp1.domain)
                adr_system.sol[1].Set(
                    (1 - 1 / (1 + exp(-5 * x))) * adr_mod,
                    dual=True,
                    definedon=comp1.domain,
                )
                arr = adr_system.sol[1].vec.FV().NumPy()
                adr_system.sol[1].vec.data[:] = np.clip(arr, a_min=0, a_max=1e100)
                depleted = True


# --- Sweep 2: same polarity model, now with membrane-tension feedback -----
# Identical to sweep 1 except Tension() feeds an area-dependent rescaling of
# A's activation threshold back into V(), closing the shape -> signaling loop
# that the paper reports as the source of tension-mediated regulation.
for a12 in a12s:
    for a21 in a21s:
        dt = 5e-2
        t0 = 0
        t1 = 100

        kappa1 = 0.4  # strength of the curvature/velocity feedback into A's kinetics
        kappa2 = 0.4  # strength of the curvature/velocity feedback into B's kinetics
        m = 4  # Hill coefficient for the A/B competition driving V()
        s = 8  # Hill coefficient for Tension() (used only in the feedback sweep below)
        r1 = 1  # linear decay rate of A
        r2 = 1  # linear decay rate of B
        klim_A = 1  # self-limiting (quadratic) production rate of A
        klim_B = 1  # self-limiting (quadratic) production rate of B
        A0 = 2.2  # half-activation threshold for the front species A
        B0 = 2.2  # half-activation threshold for the back species B

        R = 10  # resting radius of the idealized circular cell outline
        mesh = generate_boundary_1D_circle(r=R, N=100)
        ns = specialcf.normal(2)
        Area0 = Integrate(CF((x, 0)) * ns, mesh, VOL_or_BND=BND)  # rest area, the tension baseline

        from cosmos.core.model import CosmosModel

        model_name = f"Application_1_feedback_a12_{a12}_a21_{a21}"
        model = CosmosModel(
            name=model_name,
            parentmesh=mesh,
            dt=dt,
            t0=t0,
            t1=t1,
            root=ROOT,
            samples=100,
            coupling_type="implicit",
            redistribute=True,
            adaptive_timestep=True,
        )

        comp1 = model.create_compartment(name="compartment", boundary="default", bboundary="")

        adr_system = model.create_pde(
            name="adr_system",
            pde_model=ADRBoundarySystemBDF1Model,
            compartment=comp1,
            ale_type=1,
            dim=2,
        )

        ale = model.create_ale(name="ale", compartment=comp1)

        # Tension() is a switch-like (Hill, order s) function of the current
        # enclosed area relative to the rest area Area0: T ~ 1 near rest area,
        # rising sharply once the cell has spread beyond it. This stands in
        # for membrane tension increasing as the cell stretches.
        def Tension():
            Area = Integrate(CF((x, 0)) * ns, mesh, VOL_or_BND=BND)
            T = (Area ** (s + 1)) / (Area**s + Area0**s)
            return T

        # Same front/back competition as sweep 1, but A's effective threshold
        # is now A0*T instead of the fixed A0: rising tension makes it harder
        # for A to win, i.e. harder for the membrane to keep protruding --
        # the mechanochemical feedback loop closing shape back onto signaling.
        def V():
            T = Tension()
            term1 = (adr_system.sol[0]) ** m / ((adr_system.sol[0]) ** m + (A0 * T) ** m)
            term2 = (adr_system.sol[1]) ** m / ((adr_system.sol[1]) ** m + B0**m)
            return term1 - term2

        ale.set_normal_velocity(V)
        ale.set_tangential_velocity(CF((0, 0)))

        adr_system.add_nonlinearity(
            target=1,
            expression="klim_A*u1**2 + a12*u1*u2 - kappa1*V*ns*u1",
            map={
                "kappa1": kappa1,
                "klim_A": klim_A,
                "a12": a12,
                "V": model.ale.V,
                "ns": specialcf.normal(2),
            },
        )
        adr_system.add_nonlinearity(
            target=2,
            expression="klim_B*u2**2 + a21*u1*u2 + kappa2*V*ns*u2",
            map={
                "kappa2": kappa2,
                "klim_B": klim_B,
                "a21": a21,
                "V": model.ale.V,
                "ns": specialcf.normal(2),
            },
        )

        adr_system.set_params(
            u0_1=0.1,
            d_1=0.1,
            c_1=-r1,
            b_1=lambda: model.ale.V,
            bounds_1=[0, 1e100],
            u0_2=1,
            d_2=0.1,
            c_2=-r2,
            b_2=lambda: model.ale.V,
            bounds_2=[0, 1e100],
            printing=True,
        )

        output_callables = {
            "mass_A": lambda: Integrate(adr_system.sol[0], mesh, VOL_or_BND=BND),
            "mass_B": lambda: Integrate(adr_system.sol[1], mesh, VOL_or_BND=BND),
            "tension": lambda: Tension(),
            "A_point": lambda: adr_system.sol[0].vec[0],
            "B_point": lambda: adr_system.sol[1].vec[0],
            "normal_velocity": lambda: ale.gfu_norm_vel.vec[0],
            "x_bary": lambda: (
                Integrate(x, mesh, VOL_or_BND=BND) / Integrate(1, mesh, VOL_or_BND=BND)
            ),
            "y_bary": lambda: (
                Integrate(y, mesh, VOL_or_BND=BND) / Integrate(1, mesh, VOL_or_BND=BND)
            ),
            "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
            "volume": lambda: Integrate(CF((x, 0)) * ns, mesh, VOL_or_BND=BND),
            "min_A": lambda: np.min(adr_system.sol[0].vec.FV().NumPy()),
            "max_A": lambda: np.max(adr_system.sol[0].vec.FV().NumPy()),
            "min_B": lambda: np.min(adr_system.sol[1].vec.FV().NumPy()),
            "max_B": lambda: np.max(adr_system.sol[1].vec.FV().NumPy()),
        }

        model.set_params(output_callables=output_callables)

        depleted = False
        for _ in model():
            if model.t.Get() >= 10 * dt and not depleted:
                adr_mod = GridFunction(H1(mesh))
                adr_mod.Set(adr_system.sol[1], definedon=comp1.domain)
                adr_system.sol[1].Set(
                    (1 - 1 / (1 + exp(-5 * x))) * adr_mod,
                    dual=True,
                    definedon=comp1.domain,
                )
                arr = adr_system.sol[1].vec.FV().NumPy()
                adr_system.sol[1].vec.data[:] = np.clip(arr, a_min=0, a_max=1e100)
                depleted = True
