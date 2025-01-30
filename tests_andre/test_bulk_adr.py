# %%

from cosmos.solvers.bulk_adr import BulkADRSolver
from cosmos.solvers.base_problem import UnsteadyProblem
from cosmos.utils.manufactured_solution_tools import gradient
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw

def solve_bulk_adr(dh, dt, neu = False):

    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))

    T = 1.0
    dt = Parameter(dt)
    t = Parameter(0)

    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + x**2
    c = y**2
    b = CF((2,cos(x)))
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, Id(2))).Compile()
    flux = flux1 + flux2
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, Id(2))) + c*u_ex).Compile()
    fes_order = 1
    flux_c_bc = {'.*': flux1}
    flux_d_bc = {'.*': flux2}
    dirichlet_bc = {'.*': u_ex}

    if neu:

        adr_sol = BulkADRSolver(fes_order=fes_order,
                                rhs = rhs,
                                advection = b,
                                diffusion = d,
                                reaction = c,
                                u0 = u_ex,
                                flux_c_bc=flux_c_bc,
                                flux_d_bc=flux_d_bc)
        
    else:

        adr_sol = BulkADRSolver(fes_order=fes_order,
                                rhs = rhs,
                                advection = b,
                                diffusion = d,
                                reaction = c,
                                u0 = u_ex,
                                dir_bc = dirichlet_bc)
        
    # Adding the solver to the simulation
    simulation.attach_solver(adr_sol)

    simulation.run()

    err = adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    print(err)

    # adr_sol.draw_solution()

    # adr_sol.save_solution(filename = 'results/bulk_adr')

neu = True
solve_bulk_adr(0.2, 0.2)
solve_bulk_adr(0.1, 0.1)
solve_bulk_adr(0.05, 0.05)
# %%
