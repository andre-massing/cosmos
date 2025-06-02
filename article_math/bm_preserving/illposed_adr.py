# %%
from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import BDF1
from cosmos.pdes.pde_adr_bnd import BndADR
from cosmos.pdes.pde_adr_bnd_ad import AdBndADR
from cosmos.pdes.pde_tools import SaveError, SaveSolution
from cosmos.pdes.pde_tools import gradient, sphere_transformation
from ngsolve import *
import numpy as np

def solve_illposed_adr(mesh, dt, folderpath, filename, option):

    filename = filename + '_option' + str(option)

    t = Parameter(0.0)
    dt = Parameter(dt)
    T = 1

    ############# Transformation
    def AandB(t):
        A = CF((cos(t), -sin(t), 0,\
                        sin(t), cos(t), 0,\
                            0, 0, 1), dims = (3,3))
        B = CF((0, 0, 0))
        return A, B

    A, B = AandB(t)
    phi, inv_phi, w_phi, detJ, n_ex = sphere_transformation(A, B, t)
    P_ex = Id(3) - OuterProduct(n_ex, n_ex)
    A_d, B_d = AandB(t+dt)
    displ_ex = A_d*CF((x,y,z)) + B_d - CF((x,y,z))

    ############# Manufactured solution
    u_ex = exp(-3*(x**2 + y**2))
    b = CF((z,0,-x))*(1-exp(-10*z))
    neub = {'bottom': CF((0, 0, 0))}


    ############# Boundary conditions type and solver solution
    if option == 0:
        pde = BndADR(b = b, neu_b = neub,
                     u0=u_ex, name = filename, time_scheme = BDF1())
    elif option == 1:
        pde = AdBndADR(b = b, neu_b = neub,
                     u0=u_ex, name = filename, time_scheme = BDF1())
    elif option == 2:
        pde = AdBndADR(b = b, neu_b = neub,
                     u0=u_ex, name = filename,
                     BP = [0, 1e5], time_scheme = BDF1())
    elif option == 3:
        pde = AdBndADR(b = b, neu_b = neub,
                     u0=u_ex, name = filename,
                     BP = [0, 1e5], MP = True, time_scheme = BDF1())
    sol_save = SaveSolution(folderpath = folderpath,
                        filename = filename,
                        sample_rate = 1)
    pde.SaveSol(sol_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(pde)
    solver.ale.deformation_field = displ_ex

    gfu = GridFunction(H1(mesh))
    gfu.Set(u_ex)
    mass = []

    for sol in solver():
        mass_i = Integrate(pde.gfu_save[0], mesh, VOL_or_BND=BND)
        mass.append(mass_i)
    
    mass = np.array(mass)
    mass = (mass - mass[0])/mass[0]

    return mass