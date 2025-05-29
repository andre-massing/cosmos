# %%

from cosmos.utils.generate_surface_meshes import generate_circle
from ngsolve import *
from ngsolve.webgui import Draw

mesh, _ = generate_circle(maxh=0.08)

from cosmos.pdes.pde_adr_bnd import BndADR
from cosmos.pdes.coupling_strong import StrongCoupling
from cosmos.pdes.multiplier import Multiplier
from cosmos.pdes.pde_tools import SaveSolution

a_12 = 1
a_21 = 1
kappa1 = 0.4
kappa2 = 0.4
m = 4
s = 8

###################  SPECIES 1  ##################################
A0 = CF(1)
dA = CF(0.1)
sp1 = BndADR(c = CF(-1), d = dA, u0 = A0, name = "A")

###################  SPECIES 2  ##################################
B0 = CF(1)
dB = CF(0.1)
sp2 = BndADR(c = CF(-1), d = dB, u0 = B0, name = "B")
sol_save = SaveSolution(folderpath = './results',
                        filename = 'specie_B',
                        sample_rate = 10)
sp2.SaveSol(sol_save)

###################  Normal velocity  ####
###############################
v_normal = Multiplier(space = H1(mesh, definedon = mesh.Boundaries('.*')), name = 'normal_velocity')
def lhs(data, trial, test, ale):
    return trial[0]*test[0]*ds(deformation=ale.deformation)
v_normal.AddLHS(f=lhs)

###################   SYSTEM    ##################################
system = StrongCoupling()
system.AddPDEs(sp1, sp2, v_normal)

###################  COUPLINGS  ##################################

def coupling1(data, trial, test, ale):
    cpl1 = (trial[0]**2 
            + a_12*trial[0]*trial[1]
            -kappa1*trial[2]*trial[0])*test[0]*ds(deformation=ale.deformation)
    return cpl1
system.AddCoupling(f = coupling1)

def coupling2(data, trial, test, ale):
    cpl2 = (trial[1]**2 
            + a_21*trial[0]*trial[1]
            +kappa2*trial[2]*trial[1])*test[1]*ds(deformation=ale.deformation)
    return cpl2
system.AddCoupling(f = coupling2)

Tension = GridFunction(H1(mesh))
Area0 = Integrate(CF(1), mesh, VOL_or_BND=BND)
Tension.Set(Area0/2)
def coupling3(data, trial, test, ale):
    cpl3 = (-trial[0]**m/(trial[0]**m +Tension**m)
            +trial[1]**m/(trial[1]**m + 1))*test[2]*ds(deformation=ale.deformation)
    return cpl3
system.AddCoupling(f = coupling3)

###################  Solver  ##################################
from cosmos.solvers.schemes import BDF1, Steady
from cosmos.solvers.solver_unsteady import UnsteadySolver
dt = Parameter(0.1)
T = 50
t = Parameter(0)
solver = UnsteadySolver(mesh = mesh, t = t, T = T, dt = dt)
solver.AddPDE(system, BDF1())

from cosmos.pdes.pde_elastic import Elastic
from cosmos.pdes.pde_neohook import NeoHook
bnd_funct = GridFunction(VectorH1(mesh))
ale_ext = NeoHook(lam = 1, mu = 1, dir = {'.*': bnd_funct}, steady = True)
solver.AddPDE(ale_ext, Steady())
solver.ale.SetMeshDeformation(ale_ext.d_h)

n = specialcf.normal(mesh.dim)
modified = False

aux = GridFunction(H1(mesh))
aux.Set(v_normal.gfu, definedon = mesh.Boundaries('.*'))
###################  RUN  ##################################
scene = Draw(aux, mesh, deformation = solver.ale.deformation)
for _ in solver():
    mesh.SetDeformation(solver.ale.deformation)
    # solver.ale.deformation.Set(v_normal.gfu*n*dt + solver.ale.deformation, definedon=mesh.Boundaries('.*'))
    bnd_funct.Set(v_normal.gfu*n*dt + ale_ext.d_h, definedon=mesh.Boundaries('.*'))
    Area = Integrate(CF(1), mesh, VOL_or_BND=BND)
    Tension.Set(Area**(s+1)/(Area**s + Area0**s))
    if t>10 and not modified:
        angle = 30
        ratio = 0.1
        sp2.gfu.Set(IfPos(x/Norm(CF((x,y))) - cos(angle/180*pi), ratio*sp2.gfu_save[0], sp2.gfu_save[0]), definedon=mesh.Boundaries('.*'))
        modified = True
    mesh.UnsetDeformation()

    aux.Set(v_normal.gfu, definedon = mesh.Boundaries('.*'))
    scene.Redraw()
# %%

# %%
