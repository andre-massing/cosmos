# %%
from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import BDF1
from cosmos.pdes.pde_adr_bnd import BndADR
from cosmos.pdes.pde_adr_bnd_ad import AdBndADR
from cosmos.pdes.pde_tools import SaveError
from cosmos.pdes.pde_tools import gradient, sphere_transformation
from ngsolve import *

def solve_moving_surf_bmp_adr(mesh, dt, folderpath, filename):

    t = Parameter(0.0)
    dt = Parameter(dt)
    T = 1

    ############# Transformation
    def AandB(t):
        A = CF(((1+0.25*sin(t))*cos(t), -sin(t), 0,\
                        sin(t), (1-0.25*sin(t))*cos(t), 0,\
                            0, 0, 1), dims = (3,3))
        B = CF((0.2*t, 0.1*t, 0))
        return A, B

    A, B = AandB(t)
    phi, inv_phi, w_phi, detJ, n_ex = sphere_transformation(A, B, t)
    P_ex = Id(3) - OuterProduct(n_ex, n_ex)
    A_d, B_d = AandB(t+dt)
    displ_ex = A_d*CF((x,y,z)) + B_d - CF((x,y,z))

    ############# Manufactured solution
    u_ex = cos(pi*x*y)**2*cos(t)**2
    b = P_ex*CF((2,-x,0))
    flux_b = (b*u_ex).Compile()
    rel_flux = u_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_ex, Id(3))
    flux = flux_b
    rhs = (u_ex.Diff(t) + rel_flux + Trace(gradient(flux, P_ex))).Compile()

    ############# Boundary conditions type and solver solution
    bnd = 'dir'
    if bnd == 'dir':
        dir_b = {'.*': u_ex}
        pde = AdBndADR(b = b, dir_b = dir_b,
                     rhs = rhs, u0=u_ex, name = filename,
                     BP = [0, 1], time_scheme = BDF1())
    else:
        neu_b = {'.*': flux_b} 
        pde = AdBndADR(b = b, neu_b = neu_b,
                     rhs = rhs, u0 = u_ex, name = filename,
                     BP = [0, 1], time_scheme = BDF1())

    err_save = SaveError(ex_sol = u_ex, 
                    norm = 'L2', 
                    folderpath = folderpath,
                    filename = filename)
    pde.SaveErr(err_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(pde)
    solver.ale.deformation_field = displ_ex

    solver.Solve()