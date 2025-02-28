from cosmos.solvers.poisson import PoissonSolver
from cosmos.solvers.simulation import SteadyProblem
from cosmos.utils.manufactured_solution_tools import gradient

from ngsolve import *
import numpy as np
import math

def test_g_poisson_2D_mixed():

    dh = 0.05
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=1)

    u_ex = cos(4*pi*x)*cos(6*pi*y)
    grad_u = CF((u_ex.Diff(x), u_ex.Diff(y)))
    rhs = -u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y)
    fes_order = 2
    dirichlet_bc = {'top': u_ex,
                    'bottom': u_ex}
    neumann_bc = {'right': grad_u,
                    'left': grad_u}

    p_sol = PoissonSolver(fes_order=fes_order, rhs = rhs, dir_bc=dirichlet_bc , neu_bc=neumann_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(p_sol)

    simulation.run()

    p_sol.save_solution(filename='./results/poisson/poisson')

    assert True
