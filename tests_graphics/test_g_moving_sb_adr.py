from cosmos.solvers.moving_sb_adr import MovingSurfaceBulkADRSolver
from cosmos.solvers.base_problem import MovingProblem
from cosmos.utils.manufactured_solution_tools import gradient, get_lin_trans_params
from cosmos.utils.generate_surface_meshes import generate_circle, generate_sphere
from ngsolve import *
import numpy as np


def test_g_moving_sb_adr_vol():

    mesh, _ = generate_circle(maxh = 0.1)

    T = 1
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

    # General affine transformation (sphere to ellipse)
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t),\
                sin(t), (1-0.25*sin(t))*cos(t)), dims = (2,2))
    B = CF((0.2*t, 0.1*t))

    phi = A*CF((x,y)) + B
    detJ = Det(A)
    invA = Cof(A).trans/detJ
    inv_phi = invA*(CF((x,y)) - B)
    w_phi = A.Diff(t)*inv_phi + B.Diff(t)
    displ_ex = phi - CF((x,y))

    def displacement():
        return displ_ex

    simulation.set_displacement(displacement)
    sb_sol = MovingSurfaceBulkADRSolver(fes_order=1)
    simulation.attach_solver(sb_sol)

    ## Surface part 1
    normal = CF((x,y))/Norm(CF((x,y)))
    P = Id(2) - OuterProduct(normal, normal)

    u_s1_ex = sin(pi*x)*cos(3*y)*cos(2*t)
    d_s1 = 1+y**2
    c_s1 = cos(t) +1
    b_s1 = CF((-y,x))
    flux1_s1 = (b_s1*u_s1_ex).Compile()
    flux2_s1 = (- d_s1*gradient(u_s1_ex, P)).Compile()
    rel_flux_s1 = u_s1_ex*Trace(gradient(w_phi, P)) + w_phi*gradient(u_s1_ex, Id(2))
    flux_s1 = flux1_s1 + flux2_s1
    rhs_s1 = (u_s1_ex.Diff(t) + rel_flux_s1 + Trace(gradient(flux_s1, P)) + 
             c_s1*u_s1_ex).Compile()
    sb_sol.attach_surface_adr(rhs = rhs_s1,
                              advection = b_s1, 
                              diffusion = d_s1,
                              reaction = c_s1,
                              u0 = u_s1_ex)

    ## Coupling parameters
    alpha = 1.0

    ## Surface part 2

    u_s2_ex = cos(pi*x)*sin(3*y)*cos(2*t)
    d_s2 = 1+t**2
    c_s2 = cos(t) +1
    b_s2 = CF((-y,x))
    flux1_s2 = (b_s2*u_s2_ex).Compile()
    flux2_s2 = (- d_s2*gradient(u_s2_ex, P)).Compile()
    rel_flux_s2 = u_s2_ex*Trace(gradient(w_phi, P)) + w_phi*gradient(u_s2_ex, Id(2))
    flux_s2 = flux1_s2 + flux2_s2
    rhs_s2 = (u_s2_ex.Diff(t) + rel_flux_s2 + Trace(gradient(flux_s2, P)) + 
             c_s2*u_s2_ex - alpha*(u_s1_ex**2 - u_s2_ex**3)).Compile()
    sb_sol.attach_surface_adr(rhs = rhs_s2,
                              advection = b_s2, 
                              diffusion = d_s2,
                              reaction = c_s2,
                              u0 = u_s2_ex)

    ## Define non-linear coupling

    def f2(u1, u2):

        return -alpha*(u1**2 - u2**3)
    sb_sol.add_coupling(mrk1 = 0, mrk2 = 1, f2 = f2)

    simulation.run()

    sb_sol.save_solution(filename='./results/m_sb_adr/m_sb_adr_vol')

    assert True

def test_g_moving_sb_adr_bnd():

    mesh, _ = generate_sphere()

    T = 1
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

    # General affine transformation (sphere to ellipse)
    # General affine transformation (sphere to ellipse)
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t), 0,\
                sin(t), (1-0.25*sin(t))*cos(t), 0,\
                    0, 0, 1), dims = (3,3))
    B = CF((0.2*t, 0.1*t, 0))

    phi, inv_phi, w_phi, detJ, n_ex = get_lin_trans_params(A, B, t)
    P = Id(3) - OuterProduct(n_ex, n_ex)
    displ_ex = phi - CF((x,y,z))

    def displacement():
        return displ_ex

    simulation.set_displacement(displacement)
    sb_sol = MovingSurfaceBulkADRSolver(fes_order=1)
    simulation.attach_solver(sb_sol)

    ## Surface part 1
    normal = CF((x,y,z))/Norm(CF((x,y,z)))
    P = Id(3) - OuterProduct(normal, normal)

    u_s1_ex = sin(pi*x)*cos(3*y)*cos(2*t)
    d_s1 = 1+y**2
    c_s1 = cos(t) +1
    b_s1 = CF((-y,x,0))
    flux1_s1 = (b_s1*u_s1_ex).Compile()
    flux2_s1 = (- d_s1*gradient(u_s1_ex, P)).Compile()
    rel_flux_s1 = u_s1_ex*Trace(gradient(w_phi, P)) + w_phi*gradient(u_s1_ex, Id(3))
    flux_s1 = flux1_s1 + flux2_s1
    rhs_s1 = (u_s1_ex.Diff(t) + rel_flux_s1 + Trace(gradient(flux_s1, P)) + 
             c_s1*u_s1_ex).Compile()
    sb_sol.attach_surface_adr(rhs = rhs_s1,
                              advection = b_s1, 
                              diffusion = d_s1,
                              reaction = c_s1,
                              u0 = u_s1_ex)

    ## Coupling parameters
    alpha = 1.0

    ## Surface part 2

    u_s2_ex = cos(pi*x)*sin(3*y)*cos(2*t)
    d_s2 = 1+t**2
    c_s2 = cos(t)+1
    b_s2 = CF((-z, 0, x))
    flux1_s2 = (b_s2*u_s2_ex).Compile()
    flux2_s2 = (- d_s2*gradient(u_s2_ex, P)).Compile()
    rel_flux_s2 = u_s2_ex*Trace(gradient(w_phi, P)) + w_phi*gradient(u_s2_ex, Id(3))
    flux_s2 = flux1_s2 + flux2_s2
    rhs_s2 = (u_s2_ex.Diff(t) + rel_flux_s2 + Trace(gradient(flux_s2, P)) + 
             c_s2*u_s2_ex - alpha*(u_s1_ex**2 - u_s2_ex**3)).Compile()
    sb_sol.attach_surface_adr(rhs = rhs_s2,
                              advection = b_s2, 
                              diffusion = d_s2,
                              reaction = c_s2,
                              u0 = u_s2_ex)

    ## Define non-linear coupling

    def f2(u1, u2):

        return -alpha*(u1**2 - u2**3)
    sb_sol.add_coupling(mrk1 = 0, mrk2 = 1, f2 = f2)

    simulation.run()

    sb_sol.save_solution(filename='./results/m_sb_adr/m_sb_adr_bnd')

    assert True