"""Application 3 (idealized geometry): actin remodeling drives spine shape.

Couples a three-species actin nucleation/severing reaction-network (inside
the spine volume) to Willmore-flow shape evolution of the spine membrane, on
a synthetic "idealized" spine geometry (an axisymmetric neck+head profile
from ``dendritic_spine_geom.generate_synapse3d``, as opposed to the
segmented real geometry used in ``Application_3_realistic_spine.py`` -- the
paper's direct comparison of the two is what motivates having both files).

The three ADR species (``u1=A``, ``u2=B``, ``u3=C``) implement a minimal
actin turnover cycle, read off the parameter names:

- ``A`` -- a nucleation-promoting factor (e.g. an Arp2/3-activating signal).
- ``B`` -- the G-actin (monomeric actin) pool, hence its concentration
  (``B0``) is ~1000x larger than A or C.
- ``C`` -- newly-severed actin (e.g. cofilin-severed fragments/monomers
  released back from filaments).

``add_nonlinearity`` wires them together: A catalyses "nucleation" that
consumes B (target 1), that same nucleation plus a switch-like
(Hill, order N) "severing" reaction consume more B (target 2), and severing
converts that consumed B directly into C (target 3) -- a
nucleation/polymerization/severing cycle rather than a simple decay chain.

The membrane's Willmore flow (``geom_flow``) is forced by ``B`` itself
(``rhs=adr_sys.sol[1]*1e-2``): wherever the G-actin pool locally accumulates,
the spine membrane is pushed outward, giving actin dynamics direct control
over spine shape. ``dist_fct`` (a smoothed signed-distance-to-membrane field)
is used to bias B's advective transport toward the membrane within a thin
boundary layer via ``sinh/cosh`` (a smooth sign function of distance) --
standing in for actin being preferentially delivered/retained near the cell
surface rather than diffusing freely through the bulk.

Time is set up as an equilibration-then-stimulus-then-relaxation protocol,
mimicking a synaptic (LTP-like) stimulation experiment: the simulation
starts at t=-60 and runs unperturbed until t=0 (letting the reaction network
settle near its own steady state), a stimulus impulse
(``I_S*`` terms, active only for ``0 < t < 60`` via the ``impulse`` gate)
transiently boosts all three source terms, and the run continues to t1=70 to
observe how the spine shape relaxes afterwards. (The commented-out
alternative T1 values were used to stop right at/before t=0, to inspect the
pre-stimulus equilibrium on its own.)
"""

import logging

logging.basicConfig(level=logging.INFO)

from dendritic_spine_geom import generate_synapse3d
from ngsolve import *

from cosmos import *
from cosmos.core.model import CosmosModel
from cosmos.pde import ADRVolumeSystemBDF1Model, DistanceVolumeModel, GeometricalFlowStationaryModel

INITIAL_TIMESTEP = 0.01  # INITIAL_TIMESTEP = 0.001
T1 = 70  # T1 = -59.75 or T1 = -59.999
MAXH = 0.02  # MAXH = 0.04
ROOT = "."

mesh = generate_synapse3d(maxh=MAXH)

###################  PARAMETERS  ##################################

A0 = 20  # initial concentration of the nucleation-promoting factor A
B0 = 3000 * 3.6  # initial concentration of the G-actin pool B (dominant species)
C0 = 40  # initial concentration of severed actin C
K_A = 0.0013  # linear decay rate of A
K_B = 0.0081  # linear decay rate of B
K_C = 0.0006  # linear decay rate of C
I_A = 0.0255  # baseline (homeostatic) source term for A
I_B = 24.4284  # baseline source term for B
I_C = 0.0237  # baseline source term for C
I_SA = 0.0293  # stimulus-evoked extra source for A (active during the impulse window)
I_SB = 25.6684  # stimulus-evoked extra source for B
I_SC = 0.4384  # stimulus-evoked extra source for C
K_nuc = 0.0153  # nucleation rate constant (A-catalysed consumption of B)
K_sev = 0.012  # severing rate constant (C-triggered consumption of B -> production of C)
K_n = 0.6  # half-saturation constant for the severing Hill function
psi0 = 3.6  # stoichiometric conversion factor, B loss -> C/severing balance
psi1 = 0.02  # stoichiometric conversion factor for the nucleation term
N = 3  # Hill coefficient (cooperativity) of the severing reaction

dt = Parameter(INITIAL_TIMESTEP)
t = Parameter(-60)


model_name = f"Application_3_MAXH_{MAXH}_DT_{INITIAL_TIMESTEP}_T1_{T1}"
model = CosmosModel(
    parentmesh=mesh,
    dt=dt,
    t=t,
    t0=t.Get(),
    t1=T1,
    root=ROOT,
    samples=400,
    name=model_name,
    coupling_type="implicit",
    redistribute=True,
    adaptive_timestep=True,
)

model.print_model_data()

# "bulk" is the spine's cytoplasmic volume, bounded by the membrane surface.
comp1 = model.create_compartment("bulk", material="default", boundary="membrane|default")
# A one-off (ale_type=-1: solved once at init, not every step) signed-distance
# field to the membrane, used below both to seed the initial condition away
# from the membrane and to bias transport toward it (see b_2 below).
dist_fct = model.create_pde(
    "distance_function",
    pde_model=DistanceVolumeModel,
    compartment=comp1,
    zero_bnd="membrane|default",
    ale_type=-1,
)
dist_fct.set_params(printing=True)
adr_sys = model.create_pde(
    "adr_system", pde_model=ADRVolumeSystemBDF1Model, compartment=comp1, ale_type=1, dim=3
)

# "surface" is the spine membrane itself; membrane_bnd (where the membrane
# meets the dendritic shaft) is clamped, matching the idealized geometry's
# open axisymmetric profile.
comp2 = model.create_compartment(
    "surface", boundary="membrane", bboundary="membrane_bnd", clamped_bbnd="membrane_bnd"
)
geom_flow = model.create_pde(
    "willmore", pde_model=GeometricalFlowStationaryModel, compartment=comp2, ale_type=0
)
# Forced purely by the local G-actin concentration (sol[1] = B); alpha=1 is
# pure bending resistance (no extra tension/area term here).
geom_flow.set_params(rhs=lambda: adr_sys.sol[1] * 1e-2, alpha=1, printing=True)

ale = model.create_ale("ale", compartment=comp2)
ale.set_normal_velocity(geom_flow.V_h)
ale.set_tangential_velocity(CF((0, 0, 0)))

# Nucleation: A catalyses conversion of B (bimolecular in u1*u2).
adr_sys.add_nonlinearity(
    target=1, expression="K_nuc*psi1*u1*u2", map={"K_nuc": K_nuc, "psi1": psi1}
)

# B is consumed by both nucleation and a switch-like (Hill, order N)
# severing reaction gated by C -- the "-1*psi0*(...)" is a pure loss term.
adr_sys.add_nonlinearity(
    target=2,
    expression="-1*psi0*(K_nuc*psi1*u1*u2 + K_sev*u3**N/(K_n + u3**N)*psi1*u2)",
    map={"K_nuc": K_nuc, "psi1": psi1, "psi0": psi0, "K_sev": K_sev, "K_n": K_n, "N": N},
)

# The severing reaction above converts that same consumed B into C.
adr_sys.add_nonlinearity(
    target=3,
    expression="K_sev*u3**N/(K_n + u3**N)*psi1*u2",
    map={"psi1": psi1, "K_sev": K_sev, "K_n": K_n, "N": N},
)

model.initialize()
dist_fct.Initialize()
dist_fct.Solve()
# Confine the initial species pools to the spine head (z > 0.4, away from the
# neck/dendrite shaft) and away from the membrane itself (distance > 0.02) --
# actin is not initialised sitting exactly at the boundary.
id_funct = IfPos(dist_fct.sol[0] - 0.02, 1, 0) * IfPos(z - 0.4, 1, 0)
# Stimulus gate: 1 for 0 < t < 60, 0 otherwise. Combined with t starting at
# -60, this gives equilibration (t in [-60, 0]) -> stimulus (t in [0, 60])
# -> relaxation (t in [60, 70]).
impulse = IfPos(t, 1, 0) * IfPos(60 - t, 1, 0)

adr_sys.set_params(
    Neu_bnd="membrane|default",
    u0_1=A0 * id_funct,
    b_1=lambda: model.ale.V,
    c_1=K_A,
    rhs_1=I_A + I_SA * impulse,
    gradu_bnd_1=CF((0, 0, 0)),
    bounds_1=[0, 1e100],
    u0_2=B0 * id_funct,
    c_2=K_B,
    # B's advection velocity is the mesh velocity plus a small extra term
    # along dist_fct's gradient direction (sol[1]); sinh(50 d)/cosh(50 d) is
    # a smoothed sign(d) that is ~0 except within a thin boundary layer near
    # the membrane -- biasing G-actin transport toward the surface without
    # affecting bulk diffusion far from it.
    b_2=lambda: (
        model.ale.V
        + dist_fct.sol[1] * 1e-3 * sinh(50 * dist_fct.sol[0]) / cosh(50 * dist_fct.sol[0])
    ),
    rhs_2=psi0 * (I_B + I_SB * impulse),
    gradu_bnd_2=CF((0, 0, 0)),
    bounds_2=[0, 1e100],
    u0_3=C0 * id_funct,
    b_3=lambda: model.ale.V,
    c_3=K_C,
    rhs_3=I_C + I_SC * impulse,
    gradu_bnd_3=CF((0, 0, 0)),
    bounds_3=[0, 1e100],
    printing=True,
)

output_callables = {
    "mass_A": lambda: Integrate(adr_sys.sol[0], mesh),
    "mass_B": lambda: Integrate(adr_sys.sol[1], mesh),
    "mass_C": lambda: Integrate(adr_sys.sol[2], mesh),
    # Bending (Willmore) energy relative to the spine's own evolving
    # spontaneous-curvature target -- tracks how far the shape is being
    # pushed from its mechanical equilibrium by the actin forcing.
    "energy": lambda: Integrate(
        0.5 * (geom_flow.kappa_h - geom_flow.sp_curv_h) ** 2, mesh, VOL_or_BND=BND
    ),
    "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
    "volume": lambda: Integrate(1, mesh, VOL_or_BND=VOL),
}
model.set_params(output_callables=output_callables)

model.run()
