# %%
"""Validation: robustness of a real, irregular spine mesh under a point-like
transient external force.

Loads a real segmented spine plasma-membrane mesh ("open": it has an
artificial cut boundary, ``bboundary1``, where it was sliced out of the full
dendrite/ER segmentation, hence clamped there rather than left free) and
applies a body force that is a Gaussian pulse in time
(``exp(-5*(t-2)**2)``, peaking at t=2 and decaying away on either side) and
a fixed direction (+z), e.g. mimicking a localized transient push such as an
AFM/optical-tweezer perturbation. Unlike the idealized-geometry tests, the
point here is numerical robustness: does the solver stay stable (mesh
doesn't tangle or blow up) when the underlying triangulation is irregular,
real segmented data rather than a smooth analytic surface.
"""

import logging

logging.basicConfig(level=logging.INFO)

from ngsolve import *

from cosmos import *
from cosmos.core.model import CosmosModel
from cosmos.pde import GeometricalFlowStationaryModel

DT = 0.01  # 0.001
ROOT = "."

t = Parameter(0)
model_name = f"test_realistic_mesh_under_external_force_dt_{DT}"

mesh = Mesh("../vol/spine_sliced_coarse/open/spine_coarse_sliced_PM_fixed.vol")
model = CosmosModel(
    name=model_name,
    parentmesh=mesh,
    t0=0,
    dt=DT,
    t1=10,
    t=t,
    redistribute=True,
    root=ROOT,
    samples=100,
    coupling_type="implicit",
    adaptive_timestep=True,
)

comp1 = model.create_compartment(
    name="comp1", boundary="default", bboundary="bboundary1", clamped_bbnd="bboundary1"
)
geom_flow = model.create_pde(
    name="geom_flow", pde_model=GeometricalFlowStationaryModel, compartment=comp1, ale_type=0
)
ns = specialcf.normal(3)
# No explicit kappa0/sp_curv here, so both default to the mesh's own initial
# curvature (mechanical equilibrium) -- the only thing perturbing the shape
# is the transient forcing below, isolating the force's effect from any
# curvature-mismatch relaxation.
geom_flow.set_params(printing=True, rhs=10 * exp(-5 * (t - 2) ** 2) * ns * CF((0, 0, 1)))

ale1 = model.create_ale("ale1", compartment=comp1)
ale1.set_normal_velocity(lambda: geom_flow.V_h)
ale1.set_tangential_velocity(lambda: CF((0, 0)))

model.set_params(
    output_callables={
        "energy": lambda: Integrate(
            0.5 * (geom_flow.kappa_h - geom_flow.sp_curv_h) ** 2, mesh, VOL_or_BND=BND
        ),
        "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
        "volume": lambda: Integrate(CF((x, 0, 0)) * specialcf.normal(3), mesh, VOL_or_BND=BND),
    }
)

model.run()
