# %%
"""Validation: volume-preserving Willmore flow of a torus decays energy.

A torus with major/minor radius ratio R/r=2 (not the Willmore-energy-
minimizing "Clifford torus" ratio of sqrt(2)) is evolved under pure bending
(Willmore) flow with the enclosed volume held fixed
(``volume_preserving=True``). As a genus-1 (topologically non-trivial)
surface, this is a different validation case from the sphere/tube/spine
tests elsewhere: it checks that both the bending-energy gradient flow and
the volume-preservation Lagrange-multiplier constraint behave correctly
together, and that bending energy (``0.5*kappa_h**2``, here against zero
rather than a nonzero spontaneous curvature) keeps decreasing as the shape
relaxes. Uses explicit coupling (unlike the other supplementary tests,
which use implicit) -- exercising that code path too.
"""

import logging

logging.basicConfig(level=logging.INFO)

from ngsolve import *

from cosmos import *
from cosmos.core.model import CosmosModel
from cosmos.pde import GeometricalFlowModel
from cosmos.utils.generate_meshes import generate_boundary_torus

ROOT = "."
MAXH = 0.2
SOLVER = GeometricalFlowModel  # GeometricalFlowStationaryModel

t = Parameter(0)
model_name = f"test_torus_evolution_is_energy_decaying_maxh_{MAXH}"
dt = 0.001

mesh = generate_boundary_torus(maxh=MAXH, R=2, r=1)
model = CosmosModel(
    name=model_name,
    parentmesh=mesh,
    t0=0,
    dt=dt,
    t1=10,
    t=t,
    redistribute=True,
    root=ROOT,
    samples=100,
    coupling_type="explicit",
)

comp1 = model.create_compartment(name="comp1", boundary="default", bboundary="")
geom_flow = model.create_pde(name="geom_flow", pde_model=SOLVER, compartment=comp1, ale_type=0)
ns = specialcf.normal(3)
geom_flow.set_params(
    printing=True, volume_preserving=True
)  # alpha=1 (default): pure bending, no tension term

ale1 = model.create_ale("ale1", compartment=comp1)
ale1.set_normal_velocity(lambda: geom_flow.V_h)
ale1.set_tangential_velocity(lambda: CF((0, 0)))

model.set_params(
    output_callables={
        "energy": lambda: Integrate(0.5 * (geom_flow.kappa_h) ** 2, mesh, VOL_or_BND=BND),
        "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
        "volume": lambda: Integrate(CF((x, 0, 0)) * specialcf.normal(3), mesh, VOL_or_BND=BND),
    }
)

model.run()
