# %%
from cosmos.solvers.solver_moving import MovingSolver
from cosmos.solvers.schemes import BackwardEulerDispl
from cosmos.solvers.coupling_schemes import DisplacementCoupling
from cosmos.pdes.pde_adr_bnd import BndADR
from cosmos.pdes.pde_adr_bnd_ad import AdBndADR
from cosmos.pdes.pde_tools import SaveError, SaveSolution
from cosmos.pdes.pde_tools import gradient, sphere_transformation
from cosmos.utils.generate_surface_meshes import generate_half_sphere
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
from ngsolve.webgui import Draw

dt = 0.01
mesh, _ = generate_half_sphere(maxh = 0.1)
t = Parameter(0.0)
dt = Parameter(dt)
T = 0.1

############# Transformation
def AandB(t):
    A = CF((1, 0, 0,\
                    0, 1, 0,\
                        0, 0, 1), dims = (3,3))
    B = CF((0, 0, 0))
    return A, B

A_d, B_d = AandB(t+dt)
displ_ex = A_d*CF((x,y,z)) + B_d - CF((x,y,z))
n_ex = CF((x, y, z))/Norm(CF((x, y, z)))
P_ex = Id(3) - OuterProduct(n_ex, n_ex)

############# Manufactured solution
u_ex = exp(-3*(x**2 + y**2))
b = P_ex*CF((z,0,-x))
Fneub = {'.*': CF((0, 0, 0))}


############# Boundary conditions type and solver solution
dir_b = {'.*': u_ex}
pde = AdBndADR(b = b, Fneu_b = Fneub,
                    u0=u_ex,
                    BP = [0, 1e5], MP = True)

coupling = DisplacementCoupling()
def displ():
    return displ_ex
coupling.AddDisplacement(displ, '.*', BND, total = True)

solver = MovingSolver(mesh = mesh, dt=dt, T=T, t=t, coupling=coupling)
scheme = BackwardEulerDispl()
solver.AddPDE(pde, scheme)

gfu = GridFunction(H1(mesh))
gfu.Set(u_ex)
max = []

for sol in solver():
    max_i = np.max(pde.gfu_comp[0].vec.FV().NumPy())
    max.append(max_i)

plt.plot(np.array(max))
Draw(pde.gfu_comp[0])