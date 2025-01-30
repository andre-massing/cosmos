# %%
 
from cosmos.solvers.moving_bulk_adr import MovingBulkADRSolver
from cosmos.solvers.base_problem import MovingProblem
from cosmos.utils.manufactured_solution_tools import gradient
from cosmos.utils.generate_surface_meshes import *
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw

def solve_m_bulk_adr(dh, dt, neu = False):

    mesh, _ = generate_circle(maxh=dh, R = 1, order_g =2)
    T = 1.0
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

    def displacement():
        return displ_ex

    simulation.set_displacement(displacement)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + t**2
    c = 1
    b = CF((sin(x), 1))
    flux = b*u_ex - d*gradient(u_ex, Id(2)) + w_phi*u_ex
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, Id(2))) + c*u_ex).Compile()
    fes_order = 1
    flux_c = b*u_ex
    flux_d = - d*gradient(u_ex, Id(2))
    flux_c_bc = {'.*': flux_c}
    flux_d_bc = {'.*': flux_d}
    dirichlet_bc = {'.*': u_ex}

    if neu:

        adr_sol = MovingBulkADRSolver(fes_order=fes_order,
                              rhs = rhs,
                              advection = b,
                              diffusion = d,
                              reaction = c,
                              u0 = u_ex,
                              flux_c_bc=flux_c_bc,
                              flux_d_bc=flux_d_bc)
    else:
        adr_sol = MovingBulkADRSolver(fes_order=fes_order,
                              rhs = rhs,
                              advection = b,
                              diffusion = d,
                              reaction = c,
                              u0 = u_ex,
                              dir_bc=dirichlet_bc)

    simulation.attach_solver(adr_sol)

    simulation.run()

    err = adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    print(err)

    # adr_sol.draw_solution(simulation.data)

neu = False
solve_m_bulk_adr(0.2, 0.2, neu)
solve_m_bulk_adr(0.1, 0.1, neu)
solve_m_bulk_adr(0.05, 0.05, neu)
# solve_m_bulk_adr(0.025, 0.025, neu)
# %%
