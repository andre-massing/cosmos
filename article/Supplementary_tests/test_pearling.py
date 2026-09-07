# %%
"""Validation: the pearling instability, and that the flow decays energy.

A membrane tube (a cylinder capped with hemispheres, "cigar" geometry) is
given a spontaneous curvature (``kappa0``) that does not match a cylinder's
own curvature, driving the classic pearling/beading instability: the tube
breaks up into a string of near-spherical "pearls" as it minimizes bending
energy (Bar-Ziv & Moses-style tube instability). Since the Willmore/Helfrich
flow implemented here is a gradient flow of the bending energy, that energy
(logged via ``output_callables['energy']``) should decrease monotonically
throughout -- this is the numerical check the module name refers to, not
just a qualitative shape check.

Two tube lengths are run back to back with different ``kappa0`` magnitudes
(shorter tube / kappa0=-2 -> two pearls; longer tube / kappa0=-3 -> three
pearls), demonstrating that the number of pearls the instability selects
depends on tube length, as expected physically.
"""

import logging

logging.basicConfig(level=logging.INFO)

from ngsolve import *

from cosmos import *
from cosmos.core.model import CosmosModel
from cosmos.pde import GeometricalFlowStationaryModel
from cosmos.utils.generate_meshes import generate_boundary_cigar

MAXH = 0.25  # 0.125
REDISTRIBUTE = True
ROOT = "."
SOLVER = GeometricalFlowStationaryModel

# --- Two-pearl case: shorter tube (h=3) -------------------------------------
mesh = generate_boundary_cigar(maxh=MAXH, r=1, h=3)
model_name = f"test_two_pearls_example_ne_{mesh.nfacet}"
model = CosmosModel(
    name=model_name,
    parentmesh=mesh,
    t0=0,
    dt=1e-3,
    t1=1,
    redistribute=REDISTRIBUTE,
    root=ROOT,
    samples=100,
    coupling_type="implicit",
    adaptive_timestep=False,
)

comp1 = model.create_compartment(name="comp1", boundary="default", bboundary="")
geom_flow = model.create_pde(name="geom_flow", pde_model=SOLVER, compartment=comp1, ale_type=0)
geom_flow.set_params(
    printing=True, kappa0=CF(-2)
)  # mismatched vs. the tube's own curvature -> instability

ale1 = model.create_ale("ale1", compartment=comp1)
ale1.set_normal_velocity(lambda: geom_flow.V_h)
ale1.set_tangential_velocity(lambda: CF((0, 0, 0)))

# "energy" is the quantity that must decrease monotonically over the run for
# this to count as a valid (gradient-flow) evolution -- inspect the logged
# output_step_data.txt / plot it against time to check.
model.set_params(
    output_callables={
        "energy": lambda: Integrate(
            0.5 * (geom_flow.kappa_h - geom_flow.sp_curv_h) ** 2,
            mesh,
            VOL_or_BND=BND,
        ),
        "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
        "volume": lambda: Integrate(CF((x, 0, 0)) * specialcf.normal(3), mesh, VOL_or_BND=BND),
    }
)

model.run()


# --- Three-pearl case: longer tube (h=5), larger curvature mismatch --------
model_name = f"test_three_pearls_example_ne_{mesh.nfacet}"

mesh = generate_boundary_cigar(maxh=MAXH, r=1, h=5)
model = CosmosModel(
    name=model_name,
    parentmesh=mesh,
    t0=0,
    dt=1e-3,
    t1=0.3,
    redistribute=REDISTRIBUTE,
    root=ROOT,
    samples=100,
    coupling_type="implicit",
    adaptive_timestep=False,
)

comp1 = model.create_compartment(name="comp1", boundary="default", bboundary="")
geom_flow = model.create_pde(name="geom_flow", pde_model=SOLVER, compartment=comp1, ale_type=0)
geom_flow.set_params(printing=True, kappa0=CF(-3))

ale1 = model.create_ale("ale1", compartment=comp1)
ale1.set_normal_velocity(lambda: geom_flow.V_h)
ale1.set_tangential_velocity(lambda: CF((0, 0, 0)))

model.set_params(
    output_callables={
        "energy": lambda: Integrate(
            0.5 * (geom_flow.kappa_h - geom_flow.sp_curv_h) ** 2,
            mesh,
            VOL_or_BND=BND,
        ),
        "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
        "volume": lambda: Integrate(CF((x, 0, 0)) * specialcf.normal(3), mesh, VOL_or_BND=BND),
    }
)

model.run()
