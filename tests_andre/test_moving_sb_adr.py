# %%

from cosmos.solvers.moving_sb_adr import MovingSurfaceBulkADRSolver
from cosmos.solvers.base_problem import MovingProblem
from cosmos.utils.manufactured_solution_tools import gradient
from cosmos.utils.generate_surface_meshes import generate_circle
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw

def solve_moving_sb_adr(dh, dt):

    mesh, _ = generate_circle(maxh = dh)

    T = 1
    dt = Parameter(dt)
    t = Parameter(0)

    simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

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

    u_v_ex = sin(pi*x)*cos(3*y)*cos(2*t)
    d_v = 1+y**2
    c_v = cos(t) +1
    b_v = CF((-y,x))
    flux1_v = (b_v*u_v_ex).Compile()
    flux2_v = (- d_v*gradient(u_v_ex, P)).Compile()
    rel_flux_v = u_v_ex*Trace(gradient(w_phi, P)) + w_phi*gradient(u_v_ex, Id(2))
    flux_v = flux1_v + flux2_v
    rhs_v = (u_v_ex.Diff(t) + rel_flux_v + Trace(gradient(flux_v, P)) + 
             c_v*u_v_ex).Compile()
    sb_sol.attach_surface_adr(rhs = rhs_v,
                              advection = b_v, 
                              diffusion = d_v,
                              reaction = c_v,
                              u0 = u_v_ex)

    ## Coupling parameters
    alpha = 1.0

    ## Surface part 2
    u_s_ex = cos(pi*x)*sin(3*y)*cos(2*t)
    d_s = 1+t**2
    c_s = cos(t)+1
    b_s = CF((-y,x))
    flux1_s = (b_s*u_s_ex).Compile()
    flux2_s = (- d_s*gradient(u_s_ex, P)).Compile()
    rel_flux_s = u_s_ex*Trace(gradient(w_phi, P)) + w_phi*gradient(u_s_ex, Id(2))
    flux_s = flux1_s + flux2_s
    rhs_s = (u_s_ex.Diff(t) + rel_flux_s + Trace(gradient(flux_s, P)) + 
             c_s*u_s_ex - alpha*(u_v_ex**2 - u_s_ex**3)).Compile()
    sb_sol.attach_surface_adr(rhs = rhs_s,
                              advection = b_s, 
                              diffusion = d_s,
                              reaction = c_s,
                              u0 = u_s_ex)

    ## Define non-linear coupling

    def f2(u1, u2):
        return -alpha*(u1**2 - u2**3)
    sb_sol.add_coupling(mrk1 = 0, mrk2 = 1, f2 = f2)

    simulation.run()

    err = sb_sol.compute_error([u_v_ex, u_s_ex], 'L2')

    for i in range(2):
        print(np.sqrt(np.sum(simulation.dt.Get()*np.array(err[i])**2)))

    # sb_sol.draw_solution(simulation.data)

solve_moving_sb_adr(0.2, 0.2)
solve_moving_sb_adr(0.1, 0.1)
solve_moving_sb_adr(0.05, 0.05)
# %%

from cosmos.solvers.moving_sb_adr import MovingSurfaceBulkADRSolver
from cosmos.solvers.base_problem import MovingProblem
from cosmos.utils.manufactured_solution_tools import gradient, get_lin_trans_params
from cosmos.utils.generate_surface_meshes import generate_circle
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw

def solve_moving_sb_adr(dh, dt):

    mesh, _ = generate_circle(maxh = dh)

    T = 1
    dt = Parameter(dt)
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

    n_ex = invA.trans*inv_phi/Norm(invA.trans*inv_phi)
    P_ex = Id(2)-OuterProduct(n_ex, n_ex)

    def displacement():
        return displ_ex

    simulation.set_displacement(displacement)
    sb_sol = MovingSurfaceBulkADRSolver(fes_order=1)
    simulation.attach_solver(sb_sol)

    ## Volume

    u_v_ex = sin(pi*x)*cos(3*y)*cos(2*t)
    d_v = 1+y**2
    c_v = cos(t) +1
    b_v = CF((2, 1))
    flux1_v = (b_v*u_v_ex).Compile()
    flux2_v = (- d_v*gradient(u_v_ex, Id(2))).Compile()
    rel_flux_v = (w_phi*u_v_ex).Compile()
    flux_v = flux1_v + flux2_v
    rhs_v = (u_v_ex.Diff(t) + Trace(gradient(flux_v + rel_flux_v, Id(2))) + 
             c_v*u_v_ex).Compile()
    sb_sol.attach_bulk_adr(rhs = rhs_v,
                              advection = b_v, 
                              diffusion = d_v,
                              reaction = c_v,
                              u0 = u_v_ex,
                              neu_bc = -flux_v)

    ## Coupling parameters
    alpha = 1.0

    ## Surface part 2
    u_s_ex = cos(pi*x)*sin(3*y)*cos(2*t)
    d_s = 1+t**2
    c_s = cos(t)+1
    b_s = CF((-y,x))
    flux1_s = (b_s*u_s_ex).Compile()
    flux2_s = (- d_s*gradient(u_s_ex, P_ex)).Compile()
    rel_flux_s = u_s_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_s_ex, Id(2))
    flux_s = flux1_s + flux2_s
    rhs_s = (u_s_ex.Diff(t) + rel_flux_s + Trace(gradient(flux_s, P_ex)) + 
             c_s*u_s_ex - alpha*(u_v_ex**2 - u_s_ex**3)).Compile()
    sb_sol.attach_surface_adr(rhs = rhs_s,
                              advection = b_s, 
                              diffusion = d_s,
                              reaction = c_s,
                              u0 = u_s_ex)

    ## Define non-linear coupling

    def f2(u1, u2):
        return -alpha*(u1**2 - u2**3)
    sb_sol.add_coupling(mrk1 = 0, mrk2 = 1, f2 = f2)

    simulation.run()

    err = sb_sol.compute_error([u_v_ex, u_s_ex], 'L2')

    for i in range(2):
        print(np.sqrt(np.sum(simulation.dt.Get()*np.array(err[i])**2)))

    # sb_sol.draw_solution(simulation.data)

solve_moving_sb_adr(0.2, 0.2)
solve_moving_sb_adr(0.1, 0.1)
solve_moving_sb_adr(0.05, 0.05)