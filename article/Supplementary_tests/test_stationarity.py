# %%
"""Validation: a real spine mesh at its own resting curvature does not move.

Self-consistency check for the geometric-flow solver on real, irregular
segmented meshes: with no ``kappa0``/``sp_curv`` override and no external
forcing, the flow's target curvature defaults to the mesh's own initial
curvature, so a mechanically-correct implementation should leave the shape
essentially stationary (all of the reported "energy" should stay ~0). This
is run on both a fully "closed" spine membrane (no boundary at all) and an
"open" one (with the sliced/clamped ``bboundary1`` rim also used in
``test_external_force.py``), covering both boundary-condition cases the
main applications actually use.
"""

import logging

logging.basicConfig(level=logging.INFO)

from ngsolve import *

from cosmos import *
from cosmos.core.model import CosmosModel
from cosmos.pde import GeometricalFlowStationaryModel

DT = 0.1  # 0.01, 0.001
ROOT = "."

# --- Closed membrane: no boundary at all ------------------------------------
model_name = f"test_realistic_closed_mesh_is_stationary_dt_{DT}"
mesh = Mesh("../vol/spine_sliced_coarse/closed/spine_coarse_sliced_PM_closed_fixed.vol")
model = CosmosModel(
    name=model_name,
    parentmesh=mesh,
    t0=0,
    dt=DT,
    t1=1,
    redistribute=True,
    root=ROOT,
    samples=100,
    coupling_type="implicit",
    adaptive_timestep=True,
)

comp1 = model.create_compartment(name="comp1", boundary="default", bboundary="")
geom_flow = model.create_pde(
    name="geom_flow", pde_model=GeometricalFlowStationaryModel, compartment=comp1, ale_type=0
)
geom_flow.set_params(printing=True)

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


# --- Open membrane: clamped cut boundary ------------------------------------
# NB: model_name below is identical to the closed-mesh run above (both stem
# from the same "..._closed_mesh_is_stationary_dt_{DT}" string), and
# CosmosIOManager writes to root/model_name -- with the same ROOT, this run
# writes into the same output folder as the closed-mesh case above and will
# overwrite/mix with it. Likely a copy-paste leftover ("open" was probably
# intended here); worth a distinct name (e.g. "test_realistic_open_mesh...")
# before relying on both runs' saved output.
model_name = f"test_realistic_open_mesh_is_stationary_dt_{DT}"
mesh = Mesh("../vol/spine_sliced_coarse/open/spine_coarse_sliced_PM_fixed.vol")
model = CosmosModel(
    name=model_name,
    parentmesh=mesh,
    t0=0,
    dt=DT,
    t1=1,
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
geom_flow.set_params(printing=True)

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
