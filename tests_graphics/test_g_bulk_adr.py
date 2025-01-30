from glapypack.solvers.bulk_adr import BulkADRSolver
from glapypack.solvers.base_problem import UnsteadyProblem
from glapypack.utils.test_tools import gradient
from ngsolve import *

def test_g_bulk_adr_2D_mixed():
    
    dh = 0.05
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + t
    c = cos(y)
    b = CF((2,1))
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, Id(2))).Compile()
    flux = flux1 + flux2
    grad_u = -flux
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, Id(2))) + c*u_ex).Compile()
    fes_order = 1
    flux_c_bc = {'right': flux1,
                    'left': flux1}
    flux_d_bc = {'right': flux2,
                    'left': flux2}
    dirichlet_bc = {'top': u_ex,
                    'bottom': u_ex}

    bulk_adr_sol = BulkADRSolver(fes_order=fes_order,
                                 rhs = rhs,
                                 advection = b,
                                 diffusion = d,
                                 reaction = c,
                                 u0 = u_ex,
                                 flux_c_bc=flux_c_bc,
                                 flux_d_bc=flux_d_bc,
                                 dir_bc = dirichlet_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(bulk_adr_sol)

    simulation.run()

    bulk_adr_sol.save_solution(filename='./results/bulk_adr/bulk_adr')

    assert True