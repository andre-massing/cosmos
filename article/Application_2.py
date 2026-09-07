"""Application 2: neutrophil protrusion under micropipette aspiration.

Models a neutrophil (PMN, "polymorphonuclear leukocyte") partly aspirated
into a micropipette -- the classic experimental setup used to load a defined
membrane tension onto a cell. The axisymmetric geometry is built by
revolving a 2D profile (two circular arcs plus the straight segment sealing
the aspirated end) around the cell's long axis: ``free_bnd`` is the
unconstrained membrane exposed outside the pipette, ``pipette_bnd`` is the
portion held inside it, and ``bboundary`` is the rim where the membrane
meets the pipette wall (clamped, since a real pipette fixes that contact
line in place).

Three coupled PDEs run on this geometry:

- ``adr_bnd`` (surface): a single "indicator" species initialised as a
  sigmoid cap centred at polar position ``R_PMN*PHI_CAP`` -- a localized
  patch of e.g. a nucleator/receptor marking where protrusion will start.
- ``adr_vol`` (volume): a diffusing cytosolic species produced at the
  membrane in proportion to the indicator (``gradu_bnd_1``), decaying in
  the bulk, and only produced while ``t < 50`` (``IfPos(t-50, 0, 1)`` -- a
  transient stimulus). This stands in for a diffusible protrusive agent
  (e.g. polymerizing actin/force-generating machinery).
- ``geom_flow`` (surface, Willmore/mean-curvature flow): the membrane shape
  equation. Its forcing (``rhs``) is proportional to the local cytosolic
  concentration, so wherever that species accumulates the membrane is
  pushed outward; ``alpha``/``gamma`` (bending/tension, both scaled by the
  drag coefficient ``gamma_drag``) resist that push, and
  ``volume_preserving=True`` keeps the cytoplasm incompressible.

The membrane's own normal velocity (``geom_flow.V_h``) then drives the ALE
mesh motion, closing the loop from local signaling to global cell shape --
this is the paper's second demonstration of tension-mediated mechanochemical
feedback, this time for a single protrusion event rather than persistent
migration.
"""


import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from ngsolve import *
import netgen.occ as occ
from cosmos import *
from cosmos.core.model import CosmosModel
from cosmos.pde import (
    GeometricalFlowModel,
    ADRVolumeSystemBDF1Model,
    ADRBoundarySystemBDF1Model
)

GAMMA_TENSION = 100  # membrane tension (area-elasticity) coefficient, before /gamma_drag
GAMMA_CURV = 10  # bending rigidity coefficient, before /gamma_drag
PHI_CAP = 0.99  # initial cap position for the indicator species, as a fraction of R_PMN
ROOT = '.'

R_PMN = 4.25  # resting radius of the (roughly spherical) neutrophil, in um
k_deg = 1  # degradation rate of the diffusing cytosolic species
k_prod = 10  # membrane production rate of the diffusing cytosolic species
F0 = 100  # protrusive force coefficient (scales geom_flow's forcing)
D_m = 1  # diffusion coefficient of the cytosolic species
gamma_drag = 1000  # drag coefficient normalising the mechanical parameters below

angle2_deg = 20  # half-angle (degrees) of the pipette opening
angle2 = angle2_deg/180*pi

# Axisymmetric profile of a partly-aspirated cell: a near-full sphere
# (pnt1->pnt2->pnt3) capped by the portion drawn into the pipette
# (pnt3->pnt4->pnt5), sealed by a straight segment along the axis.
pnt1 = occ.Pnt(0, 0, R_PMN)
pnt2 = occ.Pnt(R_PMN, 0, 0)
pnt3 = occ.Pnt(R_PMN*sin(angle2) , 0, -1*R_PMN*cos(angle2))
pnt4 = occ.Pnt(R_PMN*sin(angle2/2) , 0, -1*R_PMN*cos(angle2/2))
pnt5 = occ.Pnt(0 , 0, -R_PMN)

arc1 = occ.ArcOfCircle(pnt1, pnt2, pnt3)
arc2 = occ.ArcOfCircle(pnt3, pnt4, pnt5)
seg1 = occ.Segment(pnt1, pnt5)

w = occ.Wire([arc1, arc2, seg1])
w.edges[1].name = 'pipette_bnd'  # membrane held inside the micropipette
w.edges[0].name = 'free_bnd'  # unconstrained membrane where protrusion occurs
w.edges[0].maxh = 0.4
f = occ.Face(w)
# Revolve the 2D profile into the full axisymmetric 3D surface.
body = f.Revolve(occ.Axis((0,0,0),occ.Z), 360).Rotate(occ.Axis((0,0,0),occ.Y), 90)
body.edges[1].name = "bboundary"  # rim where the membrane meets the pipette wall (clamped)
body.edges[1].maxh = 0.1

geo = occ.OCCGeometry(body)
ngmesh = geo.GenerateMesh(maxh=R_PMN, uselocalh=True, optsteps2d=3)
mesh = Mesh(ngmesh)

###################  Solver  ##################################
t = Parameter(0)
dt = Parameter(0.02)
model_name = f"Application_2_phi_cap{PHI_CAP}_alpha{GAMMA_CURV}_gamma{GAMMA_TENSION}"
model = CosmosModel(name=model_name, parentmesh=mesh, t0 = 0, t1 = 100,
                    dt = dt, t = t,
                    coupling_type = 'implicit', adaptive_timestep = False,
                    redistribute = True,
                    root = ROOT, samples = 100)

################### GRADIENT FLOW  ##################################
# Surface "indicator" species: a sharp sigmoid cap around polar position
# R_PMN*PHI_CAP marks the initial protrusion site; zero-flux (Neumann) at
# the pipette rim since nothing crosses the clamped contact line.
comp1 = model.create_compartment(name = 'comp1', boundary = 'free_bnd', bboundary = 'bboundary', clamped_bbnd = "bboundary")
adr_bnd = model.create_pde(name = 'indicator', pde_model=ADRBoundarySystemBDF1Model, compartment=comp1, ale_type = 1, dim = 1)
adr_bnd.set_params(
    Neu_bnd = 'bboundary',
    b_1 = lambda: model.ale.Vo,
    u0_1 = 1/(1+exp(-100*(x-R_PMN*PHI_CAP))),
    printing = True
)

###################  VOLUME ADR  ##################################
# Diffusing cytosolic species (e.g. a protrusive/force-generating agent):
# produced at the membrane wherever the indicator is positive
# (gradu_bnd_1, a Neumann flux proportional to adr_bnd.sol[0]), but only
# while t < 50 -- IfPos(t-50, 0, 1) switches production off, modeling a
# transient rather than sustained stimulus. Decays in the bulk at rate k_deg.
comp2 = model.create_compartment(name = 'comp2', material = 'default', boundary = 'free_bnd|pipette_bnd')
adr_vol = model.create_pde(name = 'concentration', pde_model=ADRVolumeSystemBDF1Model, compartment=comp2, ale_type = 1, dim = 1)
ns = specialcf.normal(3)
adr_vol.set_params(
    Neu_bnd = 'free_bnd|pipette_bnd',
    d_1 = D_m,
    c_1 = k_deg,
    gradu_bnd_1 = lambda: IfPos(adr_bnd.sol[0], adr_bnd.sol[0], 0)*k_prod/D_m*ns*IfPos(t-50, 0, 1),
    b_1 = lambda: model.ale.Vo,
    printing = True
)

# Membrane shape equation: starts at the sphere's own curvature
# (kappa0=sp_curv=-2/R_PMN, so it begins at mechanical equilibrium), is
# pushed outward by the local cytosolic concentration (rhs), and resisted
# by bending (alpha) and tension (gamma) -- both normalised by gamma_drag,
# turning the Willmore-flow rate into a force-balance-with-drag statement.
# volume_preserving=True enforces an incompressible cytoplasm.
geom_flow = model.create_pde(name = 'geom_flow', pde_model=GeometricalFlowModel, compartment=comp1, ale_type = 0)
geom_flow.set_params(
    kappa0 = CF(-2/R_PMN),
    sp_curv = CF(-2/R_PMN),
    rhs = lambda: F0*adr_vol.sol[0]/gamma_drag,
    alpha = GAMMA_CURV/gamma_drag,
    gamma = GAMMA_TENSION/gamma_drag,
    volume_preserving = True,
    printing = True
)

###################  ALE  ##################################
# Mesh motion is driven purely by the geometric flow's own normal velocity
# -- this is what actually deforms the cell in response to the signaling.
ale1 = model.create_ale('ale1', compartment=comp1, printing = False)
ale1.set_normal_velocity(lambda: geom_flow.V_h)
ale1.set_tangential_velocity(lambda: CF((0,0,0)))

model.set_params(
    output_callables = {'energy': lambda: Integrate(0.5*(geom_flow.kappa_h - geom_flow.params['sp_curv'])**2, mesh, VOL_or_BND = BND),
                        'area': lambda: Integrate(1, mesh, VOL_or_BND = BND),
                        'volume': lambda: Integrate(1, mesh, VOL_or_BND = VOL),
                        'u_mass': lambda: Integrate(adr_vol.sol[0], mesh, VOL_or_BND = VOL),
                        'i_mass': lambda: Integrate(adr_bnd.sol[0], mesh, VOL_or_BND = BND),
                        'x_bary': lambda: Integrate(x, mesh, VOL_or_BND = VOL),
                        'y_bary': lambda: Integrate(y, mesh, VOL_or_BND = VOL),
                        'z_bary': lambda: Integrate(z, mesh, VOL_or_BND = VOL),
                        'x_max': lambda: np.max(model.ale.X.components[0].vec.FV().NumPy()),
                        'y_max': lambda: np.max(model.ale.X.components[1].vec.FV().NumPy()),
                        'z_max': lambda: np.max(model.ale.X.components[2].vec.FV().NumPy())
                        }
)

model.run()