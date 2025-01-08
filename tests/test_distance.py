from cosmos.solvers.distance_solver import DistanceSolver
from cosmos.utils.generate_surface_meshes import *
from cosmos.utils.manufactured_solution_tools import ErrorTools

from ngsolve import *
import math

def test_distance_2D():

    R = 1
    mesh, geo = generate_circle(maxh=0.1)

    d_ex = R - Norm(CF((x,y)))
    v_ex = CF((x,y))/Norm(CF((x,y)))

    params = {'zero_bnd': '.*'}
    
    fes_order = 1
    
    solver = DistanceSolver(mesh, fes_order=fes_order, params = params)

    sol_generator = solver()
    next(sol_generator)

    params = {'exact_solutions': [d_ex, v_ex],
              'vol_or_bnd': [VOL, VOL],
              'norms': ['L2', 'L2']}
    
    errors = ErrorTools(solver, params=params)

    ERR = errors.compute_errors()

    assert ERR[0]<7.1e-3 and ERR[1]<8e-2