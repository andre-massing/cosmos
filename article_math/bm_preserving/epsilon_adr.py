# %%
from cosmos.solvers.solver_unsteady import UnsteadySolver
from cosmos.solvers.schemes import BDF1
from cosmos.pdes.pde_adr_bnd import BndADR
from cosmos.pdes.pde_adr_bnd_ad import AdBndADR
from cosmos.pdes.pde_tools import SaveError, SaveSolution
from cosmos.pdes.pde_tools import gradient, sphere_transformation
from ngsolve import *

def solve_epsilon_adr(mesh, dt, folderpath, filename, option):

    filename = filename + '_option' + str(option)

    t = Parameter(0.0)
    dt = Parameter(dt)
    T = 1

    ############# Transformation
    def AandB(t):
        A = CF((*cos(t), -sin(t), 0,\
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
    r = sqrt(x**2+y**2)
    epsilon = 0.001
    u_ex = (1-exp((r-1)/epsilon))/(1-exp(-2/epsilon)) + 0.5*sin(pi*r)
    b = P_ex*CF((1,0,0))
    flux_b = (b*u_ex).Compile()
    rel_flux = u_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_ex, Id(3))
    flux = flux_b
    rhs = (u_ex.Diff(t) + rel_flux + Trace(gradient(flux, P_ex))).Compile()

    ############# Boundary conditions type and solver solution
    dir_b = {'.*': u_ex}
    if option == 0:
        pde = BndADR(b = b, dir_b = dir_b,
                     rhs = rhs, u0=u_ex, name = filename)
    elif option == 1:
        pde = AdBndADR(b = b, dir_b = dir_b,
                     rhs = rhs, u0=u_ex, name = filename)
    elif option == 2:
        pde = AdBndADR(b = b, dir_b = dir_b,
                     rhs = rhs, u0=u_ex, name = filename,
                     BP = [0, 1.5])
    elif option == 3:
        pde = AdBndADR(b = b, dir_b = dir_b,
                     rhs = rhs, u0=u_ex, name = filename,
                     BP = [0, 1.5], MP = True)

    err_save = SaveError(ex_sol = u_ex, 
                    norm = 'L2', 
                    folderpath = folderpath,
                    filename = filename)
    pde.SaveErr(err_save)
    sol_save = SaveSolution(folderpath = folderpath,
                        filename = filename,
                        sample_rate = 50)
    pde.SaveSol(sol_save)

    solver = UnsteadySolver(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(pde, BDF1())
    solver.ale.SetMeshDeformation(displ_ex)

    gfu = GridFunction(H1(mesh))
    gfu.Set(u_ex)
    mass = []

    for sol in solver():
        mass_i = Integrate(pde.gfu_comp[0], mesh, VOL_or_BND=BND)
        mass.append(mass_i)

    return mass