# %%

from cosmos.utils.generate_surface_meshes import generate_sphere, generate_box
from ngsolve import *
from ngsolve.webgui import Draw

mesh, _ = generate_sphere(maxh = 0.08, R = 1)
# mesh, _ = generate_box(maxh = 0.07, vol_or_bnd='BND')

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
from cosmos.pdes.pde_mc_bgn_stab import MCBGNStab
from cosmos.pdes.pde_tools import SaveSolution

###################  CAHN-HILLIARD  ##################################
theta = 0.001
gamma = 0.02
sigma = 3
aux = IfPos(sin(1e7*(x+y*y))-1, 1, sin(1e7*(x+y*y)))
c0 = IfPos(-aux-1, -1, aux)

c0 = sinh(x/sqrt(2))/cosh(x/sqrt(2))

ch = CahnHilliardBnd(c0 = c0, theta = theta, gamma=gamma, sigma = sigma, 
                     time_scheme = BDF1(), MP = True, BP = [-1, 1])
# sol_save = SaveSolution(folderpath = './results',
#                         filename = 'cahn_hilliard',
#                         sample_rate = 1)
# ch.SaveSol(sol_save)

###################  WILLMORE  ##################################
kappa = 0.02
willmore = WillmoreDziuk(mc_autoupdate = False, time_scheme=BDF1(),
                         postprocess= False, kappa = kappa)
# willmore = MCBGNStab()
# sol_save = SaveSolution(folderpath = './results',
#                         filename = 'willmore',
#                         sample_rate = 1)
# willmore.SaveSol(sol_save)

###################  Solver  ##################################
from cosmos.solvers.solvers import Dynamic
T = 1.5
dt = Parameter(1e-4)
t = Parameter(0)
solver = Dynamic(mesh = mesh, t = t, T = T, dt = dt)
solver.AddPDE(willmore)
solver.AddPDE(ch)

###################  ALE  ##################################
solver.ale.deformation_field = willmore.displacement
def velocity():
    return (willmore.displacement - solver.ale.deformation)/solver.dt
solver.ale.velocity_field = velocity
def mat_velocity():
    n = specialcf.normal(3)
    field = InnerProduct((willmore.displacement - solver.ale.deformation)/solver.dt,  n)*n
    return field
solver.ale.mat_velocity_field = mat_velocity

###################  Couplings  ##################################
def w_rhs():
    n = specialcf.normal(3)
    solver.mesh.SetDeformation(solver.ale.deformation)
    W = ComputeW(solver)
    solver.mesh.UnsetDeformation()
    term1 = - sigma*gamma*InnerProduct(W*grad(ch.phase).Trace(), grad(ch.phase).Trace())*n
    term2 = sigma*willmore.mean_curvature*(gamma/2*InnerProduct(grad(ch.phase).Trace(), grad(ch.phase).Trace())
                                     +1/(4*gamma)*(ch.phase**2-1)**2)
    return term1 + term2
willmore.rhs.value = w_rhs

scene = Draw(ch.phase, mesh, deformation = solver.ale.deformation)
for i, sol in enumerate(solver()):
    scene.Redraw()

# %%
