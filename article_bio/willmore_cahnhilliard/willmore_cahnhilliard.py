# %%

from cosmos.utils.generate_surface_meshes import generate_sphere
from ngsolve import *
from ngsolve.webgui import Draw

R = 1
mesh, _ = generate_sphere(maxh = 0.1, R = R)


from cosmos.pdes.pde_ch_bnd import CahnHilliardBnd
from cosmos.pdes.pde_willmore_dziuk_stab import WillmoreDziukStab
from cosmos.pdes.pde_mc_bgn_stab import MCBGNStab
from cosmos.pdes.pde_tools import SaveSolution

###################  CAHN-HILLIARD  ##################################
D = 1
gamma = 1e-2
epsilon = 1
u0 = [ 0.5*sin(1e7*(x+y*y)), CF(0)]
ch = CahnHilliardBnd(u0 = u0, D = D, gamma = gamma, epsilon = epsilon,
                  BP = [-1, 1], MP = True)
# sol_save = SaveSolution(folderpath = './results',
#                         filename = 'cahn_hilliard',
#                         sample_rate = 1)
# ch.SaveSol(sol_save)

###################  WILLMORE  ##################################
willmore = WillmoreDziukStab(postprocess = 1, mc_autoupdate = True)
# willmore = MCBGNStab()
# sol_save = SaveSolution(folderpath = './results',
#                         filename = 'willmore',
#                         sample_rate = 1)
# willmore.SaveSol(sol_save)

###################  Solver  ##################################
from cosmos.solvers.schemes import BDF1
from cosmos.solvers.solver_unsteady import UnsteadySolver
T = 5
dt = Parameter(0.005)
t = Parameter(0)
solver = UnsteadySolver(mesh = mesh, t = t, T = T, dt = dt)
solver.AddPDE(willmore, BDF1(conservative=False))
solver.AddPDE(ch, BDF1())
solver.ale.SetMeshDeformation(willmore.gfu.components[0])

###################  RUN  ##################################
def UpdateDef():
    solver.data.mesh.SetDeformation(solver.ale.deformation)
    n = specialcf.normal(solver.data.mesh.dim)
    willmore.params['rhs'] = [CF(0), ch.gfu.components[0]*n]
    solver.data.mesh.UnsetDeformation()
UpdateDef()

for i, sol in enumerate(solver()):

    UpdateDef()

    if i == 0:
        scene = Draw(ch.gfu.components[0], mesh, deformation = solver.ale.deformation)
        mass0 = Integrate(ch.gfu.components[0], mesh, VOL_or_BND=BND)
        mesh.UnsetDeformation()
    else:
        
        scene.Redraw()
# %%

# %%
