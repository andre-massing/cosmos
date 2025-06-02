# %%
from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import BDF2, CN
from cosmos.pdes.pde_adr_vol import VolADR
from cosmos.pdes.pde_adr_vol_ad import AdVolADR
from cosmos.pdes.pde_adr_bnd import BndADR
from cosmos.pdes.pde_adr_bnd_ad import AdBndADR
from cosmos.pdes.pde_tools import gradient
from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.pdes.pde_tools import SaveError
from cosmos.pdes.coupling_strong import StrongCoupling

def solve_moving_surfbulk_adr(mesh, dt, folderpath, filename):

    t = Parameter(0)
    T = 1
    dt = Parameter(dt)

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

    u_ex = x*sin(pi*y)*cos(t)
    c1 = 1
    b1 = CF((2,t))
    flux_b1 = (b1*u_ex).Compile()
    flux1 = flux_b1 + w_phi*u_ex
    rhs1 = (u_ex.Diff(t) + Trace(gradient(flux1, Id(2))) + c1*u_ex + u_ex**3).Compile()

    bnd = 'dir'
    if bnd == 'dir':
        dir_b = {'.*': u_ex}
        adr1 = AdVolADR(c = c1, b = b1, dir_b = dir_b, rhs = rhs1,
                        u0 = u_ex, name = filename + 'bulk')
    else:
        neu_b = {'.*': flux_b1} 
        adr1 = AdVolADR(c = c1, b = b1, neu_b = neu_b, rhs = rhs1,
                        u0 = u_ex, name = filename + 'bulk')

    err_save = SaveError(ex_sol = u_ex, 
                    norm = 'L2', 
                    folderpath = folderpath,
                    filename = filename + 'bulk')
    adr1.SaveErr(err_save)

    n_ex = Cof(A)*inv_phi/Norm(Cof(A)*inv_phi)
    P_ex = Id(2) - OuterProduct(n_ex, n_ex)
    v_ex = cos(pi*x)*cos(2*t)
    c2 = 1
    b2 = P_ex*CF((2,x))
    flux_b2 = (b2*v_ex).Compile()
    flux2 = flux_b2
    rel_flux = v_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(v_ex, Id(2))
    rhs2 = (v_ex.Diff(t) + rel_flux + Trace(gradient(flux2, P_ex)) + c2*v_ex + u_ex*sin(v_ex**2)).Compile()

    adr2 = AdBndADR(c = c2, b = b2, rhs = rhs2,
                  u0 = v_ex, name = filename + 'surf')

    err_save = SaveError(ex_sol = v_ex, 
                    norm = 'L2', 
                    folderpath = folderpath,
                    filename = filename + 'surf')
    adr2.SaveErr(err_save)

    cpl = StrongCoupling(time_scheme=BDF2())
    cpl.AddPDEs(adr1, adr2)
    def f0(solverdata, trial, test):
        return trial[0]**3*test[0]*dx(deformation=solverdata.ale.deformation)
    cpl.AddCoupling(f = f0)
    def f1(solverdata, trial, test):
        return trial[0]*sin(trial[1]**2)*test[1]*ds(deformation=solverdata.ale.deformation)
    cpl.AddCoupling(f = f1)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.ale.deformation_field = displ_ex
    solver.AddPDE(cpl)

    solver.Solve()