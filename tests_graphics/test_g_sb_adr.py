from cosmos.solvers.sb_adr import SurfaceBulkADRSolver
from cosmos.solvers.simulation import Simulation
from cosmos.utils.manufactured_solution_tools import gradient
from cosmos.utils.generate_surface_meshes import generate_circle, generate_sphere
from ngsolve import *
import numpy as np


def test_g_sb_adr_vol():

    mesh, _ = generate_circle(maxh = 0.1)

    T = 1
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = Simulation(mesh=mesh, dt=dt, t=t, T=T)
    sb_sol = SurfaceBulkADRSolver(fes_order=1)
    simulation.AddSolver(sb_sol)

    ## Volume part

    u_v_ex = cos(pi*x)*sin(3*y)*cos(t)
    d_v = 1+x**2
    c_v = cos(x)
    b_v = CF((2,1))
    flux1_v = (b_v*u_v_ex).Compile()
    flux2_v = (- d_v*gradient(u_v_ex, Id(2))).Compile()
    flux_v = flux1_v + flux2_v
    grad_u_v = -flux_v
    rhs_v = (u_v_ex.Diff(t) + Trace(gradient(flux_v, Id(2))) + c_v*u_v_ex).Compile()
    sb_sol.attach_bulk_adr(rhs = rhs_v,
                           advection = b_v,
                           diffusion = d_v,
                           reaction = c_v,
                           u0 = u_v_ex,
                           neu_bc = grad_u_v)

    ## Coupling parameters

    alpha = 1.0
    beta = 1.0

    ## Surface part
    normal = CF((x,y))/Norm(CF((x,y)))
    P = Id(2) - OuterProduct(normal, normal)

    u_s_ex = sin(pi*x)*cos(3*y)*cos(t)
    d_s = 1+y**2
    c_s = cos(y)
    b_s = P*CF((2,1))
    flux1_s = (b_s*u_s_ex).Compile()
    flux2_s = (- d_s*gradient(u_s_ex, P)).Compile()
    flux_s = flux1_s + flux2_s
    rhs_s = (u_s_ex.Diff(t) + Trace(gradient(flux_s, P)) + c_s*u_s_ex + beta*u_s_ex - alpha*u_v_ex**3).Compile()
    sb_sol.attach_surface_adr(rhs = rhs_s,
                              advection = b_s,
                              diffusion = d_s,
                              reaction = c_s,
                              u0 = u_s_ex)

    ## Define non-linear coupling
    def f2(u1, u2):

        return beta*u2 - alpha*u1**3
    sb_sol.add_coupling(mrk1 = 0, mrk2 = 1, f2 = f2)

    simulation.run()

    sb_sol.save_solution(filename='./results/sb_adr/sb_adr_vol')

    assert True

def test_f_sb_adr_bnd():

    mesh, _ = generate_sphere()

    T = 1
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = Simulation(mesh=mesh, dt=dt, t=t, T=T)
    sb_sol = SurfaceBulkADRSolver(fes_order=1)
    simulation.AddSolver(sb_sol)

    ## Surface part

    u_v_ex = cos(pi*x)*sin(3*y)*cos(t)
    d_v = 1+x**2
    c_v = cos(x)
    b_v = CF((2,1,0))
    flux1_v = (b_v*u_v_ex).Compile()
    flux2_v = (- d_v*gradient(u_v_ex, Id(3))).Compile()
    flux_v = flux1_v + flux2_v
    grad_u_v = -flux_v
    rhs_v = (u_v_ex.Diff(t) + Trace(gradient(flux_v, Id(3))) + c_v*u_v_ex).Compile()
    sb_sol.attach_surface_adr(rhs = rhs_v,
                           advection = b_v,
                           diffusion = d_v,
                           reaction = c_v,
                           u0 = u_v_ex)


    ## Coupling parameters

    alpha = 1.0
    beta = 1.0

    ## Surface part
    normal = CF((x,y,z))/Norm(CF((x,y,z)))
    P = Id(3) - OuterProduct(normal, normal)

    u_s_ex = sin(pi*x)*cos(3*y)*cos(t)
    d_s = 1+y**2
    c_s = cos(y)
    b_s = P*CF((2,1,0))
    flux1_s = (b_s*u_s_ex).Compile()
    flux2_s = (- d_s*gradient(u_s_ex, P)).Compile()
    flux_s = flux1_s + flux2_s
    rhs_s = (u_s_ex.Diff(t) + Trace(gradient(flux_s, P)) + c_s*u_s_ex + beta*u_s_ex - alpha*u_v_ex**3).Compile()
    sb_sol.attach_surface_adr(rhs = rhs_s,
                              advection = b_s,
                              diffusion = d_s,
                              reaction = c_s,
                              u0 = u_s_ex)

    ## Define non-linear coupling
    def f2(u1, u2):

        return beta*u2 - alpha*u1**3
    sb_sol.add_coupling(mrk1 = 0, mrk2 = 1, f2 = f2)

    simulation.run()

    sb_sol.save_solution(filename='./results/sb_adr/sb_adr_bnd')

    assert True