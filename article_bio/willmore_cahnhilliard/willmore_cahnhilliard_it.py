# %%

from cosmos.utils.generate_surface_meshes import generate_sphere, generate_box
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np

import os
folderpath = './prova'
os.makedirs(folderpath, exist_ok = True)

dt = 2e-4
T = 1.5
n = 100
sample_rate = np.maximum(int(T/dt/n), 1)

mesh, _ = generate_sphere(maxh = 0.07, R = 1)

def ComputeW(solverdata):

    mesh = solverdata.mesh
    order = solverdata.mesh.GetCurveOrder()

    n = specialcf.normal(3)
    t = specialcf.tangential(3)
    mu = Cross(n,t)
    
    # Average normal vector
    gfF = GridFunction(VectorFacetSurface(mesh,order=order-1))
    gfF.Set(n, dual=True, definedon=mesh.Boundaries(".*"))
    
    fes = HDivDivSurface(mesh,order=order-1)
    sigma,tau = fes.TnT()
    sigma,tau = sigma.Trace(),tau.Trace()
    
    a = BilinearForm(fes, symmetric=True)
    a += InnerProduct(sigma,tau)*ds
    
    # Grad(n) = specialcf.Weingarten(3)
    f = LinearForm(fes)
    f += -1*(InnerProduct(Grad(n),tau))*ds \
            + -1*((pi/2-acos(Normalize(gfF)*mu))*tau*mu*mu)*ds(element_boundary=True)
    
    gflift = GridFunction(fes)
    
    with TaskManager():
        a.Assemble()
        f.Assemble()
        gflift.vec.data = a.mat.Inverse(fes.FreeDofs(),inverse="sparsecholesky")*f.vec
        
    return gflift


from cosmos.pdes.pde_ch_bnd import CahnHilliardBnd
from cosmos.pdes.pde_willmore_dziuk_stab import WillmoreDziukStab
from cosmos.pdes.pde_willmore_dziuk import WillmoreDziuk
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_tools import SaveSolution

###################  CAHN-HILLIARD  ##################################
theta = 0.001
gamma = 0.5
sigma = 2
c0 = (exp(2*x/sqrt(2))-1)/(exp(2*x/sqrt(2))+1)
ch = CahnHilliardBnd(c0 = c0, theta = theta, gamma=gamma, sigma = sigma, 
                     time_scheme = BDF2(), BP = [-1, 1], MP = True)
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'cahn_hilliard',
                        sample_rate = sample_rate)
ch.SaveSol(sol_save)

###################  WILLMORE  ##################################
kappa = 0.2
willmore = WillmoreDziuk(time_scheme=BDF1(), postprocess=True)
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'willmore',
                        sample_rate = sample_rate)
willmore.SaveSol(sol_save)

###################  Solver  ##################################
from cosmos.solvers.solvers import Dynamic
dt = Parameter(dt)
t = Parameter(0)
solver = Dynamic(mesh = mesh, t = t, T = T, dt = dt)

###################  ALE  ##################################

###################  Couplings  ##################################
from cosmos.pdes.coupling_weak import WeakCoupling
cpl = WeakCoupling(tol = 1e-5, type = 'explicit')
cpl.AddPDEs(ch, willmore)
solver.AddPDE(cpl)

solver.ale.deformation_field = lambda: willmore.displacement
solver.ale.velocity_field = lambda : willmore.displacement/solver.dt
def mat_velocity():
    n = specialcf.normal(3)
    Q = OuterProduct(n,n)
    P = Id(3) - Q
    field = Q*willmore.displacement/solver.dt + P*(ch.potential*grad(ch.phase).Trace()) 
    return field
solver.ale.mat_velocity_field = mat_velocity

def w_rhs():
    solver.mesh.SetDeformation(solver.ale.deformation)
    W = ComputeW(solver)
    n = specialcf.normal(3)
    Q = OuterProduct(n,n)
    P = Id(3) - Q
    term1 = -1*sigma*gamma*W*grad(ch.phase).Trace()*grad(ch.phase).Trace()*n
    term2 = sigma*willmore.mean_curvature*(gamma/2*InnerProduct(grad(ch.phase).Trace(), grad(ch.phase).Trace())
                                     +1/(4*gamma)*(ch.phase**2-1)**2)
    solver.mesh.UnsetDeformation()
    return term1 + term2
willmore.rhs.value = w_rhs
# willmore.sp_curv.value = lambda: ch.phase

import numpy as np
np.random.seed(42)
scene = Draw(ch.phase, mesh, deformation = solver.ale.deformation)
for i, sol in enumerate(solver()):
    # if i == 0:
    #     n = len(ch.phase.vec.data)
    #     ch.phase.vec.data = np.clip(np.random.normal(0, 0.1, n), -1, 1)
    scene.Redraw()
# %%

# %%
