# %%
from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import BDF1, CN, BDF2
from cosmos.pdes.pde_adr_vol import VolADR
from cosmos.pdes.pde_adr_vol_ad import AdVolADR
from cosmos.pdes.pde_tools import SaveError
from cosmos.pdes.pde_tools import gradient
from ngsolve import *

def solve_moving_bulk_adr(mesh, dt, folderpath, filename):

    t = Parameter(0.0)
    dt = Parameter(dt)
    T = 1

    ############# Transformation
    def AandB(t):
        A = CF(((1+0.25*sin(t))*cos(t), -sin(t),\
                    sin(t), (1-0.25*sin(t))*cos(t)), dims = (2,2))
        B = CF((0.2*t, 0.1*t))
        return A, B

    A, B = AandB(t)
    A_d, B_d = AandB(t+dt)
    detJ = Det(A)
    invA = Cof(A).trans/detJ
    inv_phi = invA*(CF((x,y)) - B) # Inverse of deformation map
    w_phi = A.Diff(t)*inv_phi + B.Diff(t) # velocity of the moving domain
    displ_ex = A_d*CF((x,y))+B_d - CF((x,y))

    ############# Manufactured solution
    u_ex = cos(pi*x)*sin(pi*y)*cos(t) # Exact solution
    c = 1 + t**2 
    b = CF((2,-x))
    flux_b = b*u_ex
    rhs = (u_ex.Diff(t) + Trace(gradient(flux_b + w_phi*u_ex, Id(2))) + c*u_ex).Compile()

    ############# Boundary conditions type and solver solution
    bnd = 'dir'
    if bnd == 'dir':
        # dir_d = {'.*': u_ex}
        dir_b = {'.*': u_ex}
        pde = AdVolADR(c = c, b = b, dir_b = dir_b, 
                     rhs = rhs, u0=u_ex, name = filename, time_scheme=CN())
    else:
        # neu_d = {'.*': flux_d} 
        neu_b = {'.*': flux_b} 
        pde = AdVolADR(c = c, b = b, neu_b = neu_b,
                     rhs = rhs, u0=u_ex, name = filename, time_scheme=CN())

    err_save = SaveError(ex_sol = u_ex, 
                    norm = 'L2', 
                    folderpath = folderpath,
                    filename = filename)
    pde.SaveErr(err_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(pde)
    solver.ale.deformation_field = displ_ex

    solver.Solve()