# %%

from cosmos.utils.generate_surface_meshes import generate_sphere
from ngsolve.webgui import Draw

mesh, _ = generate_sphere(maxh=0.1)
Draw(mesh)

from cosmos.pdes.pde_adr_vol import VolADR
from cosmos.pdes.coupling_strong import StrongCoupling
from cosmos.pdes.multiplier import Multiplier
from ngsolve import *

###################  SPECIES  ##################################
A0 = CF(0.5)
dA = CF(0.2)
flux = {'.*': }
sp1 = VolADR(c = CF(1), d = dA, u0 = A0, neu_d = flux , name = "A")

###################  Normal velocity  ##################################
v_normal = Multiplier(space = H1(mesh, definedon = mesh.Boundaries('.*')), name = 'normal_velocity')
def lhs(data, trial, test, ale):
    return trial[0]*test[0]*ds(deformation=ale.deformation)
v_normal.AddLHS(f=lhs)

###################   SYSTEM    ##################################
system = StrongCoupling()
system.AddPDEs(sp1, v_normal)

###################  COUPLINGS  ##################################
factor = 0.1
def coupling(data, trial, test, ale):
    cpl = (-1*factor*trial[0])*test[1]*ds(deformation=ale.deformation)
    return cpl
system.AddCoupling(f = coupling)

###################  Solver  ##################################
from cosmos.solvers.schemes import BDF1
from cosmos.solvers.solver_unsteady import UnsteadySolver
dt = Parameter(0.01)
T = 100
t = Parameter(0)
solver = UnsteadySolver(mesh = mesh, t = t, T = T, dt = dt)
solver.AddPDE(system, BDF1())

###################  ALE deformation  ##################################
from cosmos.pdes.pde_neohook import NeoHook
from cosmos.pdes.pde_elastic import Elastic
n = specialcf.normal(mesh.dim)
bnd_funct = GridFunction(VectorH1(mesh))
dir_bnd = {'.*': bnd_funct}
ale_ext = NeoHook(lam = 1, mu = 1, rho = 1, steady = True, dir=dir_bnd)
solver.AddPDE(ale_ext, BDF1())
solver.ale.SetMeshDeformation(ale_ext.d_h)

###################  RUN  ##################################
scene = Draw(sp1.gfu_save[0], mesh, deformation = solver.ale.deformation)
for _ in solver():
    scene.Redraw()