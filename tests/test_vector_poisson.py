from cosmos.solvers.vector_poisson_solver import VectorPoissonSolver
from cosmos.utils.generate_surface_meshes import *
from cosmos.utils.manufactured_solution_tools import ErrorTools, gradient

from ngsolve import *
import numpy as np
import math

def test_vector_poisson_2D_dirichlet():

    u_ex = CF((cos(4*pi*x)*cos(6*pi*y),sin(2*pi*x)*sin(2*pi*y))).Compile()

    rhs = CF((-u_ex[0].Diff(x).Diff(x) - u_ex[0].Diff(y).Diff(y), -u_ex[1].Diff(x).Diff(x) - u_ex[1].Diff(y).Diff(y))).Compile() 

    ## Case with mixed boundary conditions
    dirichlet_bc = [['top', u_ex],
                        ['bottom', u_ex],
                        ['right', u_ex],
                        ['left', u_ex]]
    bnd_cond={'dirichlet': dirichlet_bc}
    fes_order = 2

    params = {'rhs': rhs,
              'boundary_c': bnd_cond}
    
    dh = 0.05
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    solver = VectorPoissonSolver(mesh, fes_order=fes_order, params = params)

    sol_generator = solver()
    next(sol_generator)

    params = {'exact_solutions': [u_ex],
              'vol_or_bnd': [VOL],
              'norms': ['L2']}
    errors = ErrorTools(solver, params=params)

    ERR_u = errors.compute_errors()

    assert ERR_u[0]<2.3e-3

def test_vector_poisson_2D_neumann():

    u_ex = CF((cos(4*pi*x)*cos(6*pi*y),sin(2*pi*x)*sin(2*pi*y))).Compile()
    grad_u = CF((u_ex[0].Diff(x), u_ex[0].Diff(y), u_ex[1].Diff(x), u_ex[1].Diff(y)), dims = (2,2)).Compile()

    rhs = CF((-u_ex[0].Diff(x).Diff(x) - u_ex[0].Diff(y).Diff(y), -u_ex[1].Diff(x).Diff(x) - u_ex[1].Diff(y).Diff(y))).Compile()

    ## Case with mixed boundary conditions
    neumann_bc = [['left', grad_u],
                    ['right', grad_u],
                    ['top|bottom', grad_u]]

    bnd_cond={'neumann': neumann_bc}
    fes_order = 2

    params = {'rhs': rhs,
              'boundary_c': bnd_cond}
    
    dh = 0.05
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    solver = VectorPoissonSolver(mesh, fes_order=fes_order, params = params)

    sol_generator = solver()
    next(sol_generator)
    
    params = {'exact_solutions': [u_ex],
              'vol_or_bnd': [VOL],
              'norms': ['L2']}
    errors = ErrorTools(solver, params=params)

    ERR_u = errors.compute_errors()

    assert ERR_u[0]<2.3e-3

def test_vector_poisson_2D_mixed():

    u_ex = CF((cos(4*pi*x)*cos(6*pi*y),sin(2*pi*x)*sin(2*pi*y))).Compile()
    grad_u = CF((u_ex[0].Diff(x), u_ex[0].Diff(y), u_ex[1].Diff(x), u_ex[1].Diff(y)), dims = (2,2)).Compile()

    rhs = CF((-u_ex[0].Diff(x).Diff(x) - u_ex[0].Diff(y).Diff(y), -u_ex[1].Diff(x).Diff(x) - u_ex[1].Diff(y).Diff(y))).Compile() 

    ## Case with mixed boundary conditions
    neumann_bc = [['left', grad_u],
                    ['right', grad_u]]
    dirichlet_bc = [['top', u_ex],
                        ['bottom', u_ex]]
    bnd_cond={'neumann': neumann_bc,
            'dirichlet': dirichlet_bc}
    fes_order = 2

    params = {'rhs': rhs,
              'boundary_c': bnd_cond}
    
    dh = 0.05
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    solver = VectorPoissonSolver(mesh, fes_order=fes_order, params = params)

    sol_generator = solver()
    next(sol_generator)
    
    params = {'exact_solutions': [u_ex],
              'vol_or_bnd': [VOL],
              'norms': ['L2']}
    errors = ErrorTools(solver, params=params)

    ERR_u = errors.compute_errors()

    assert ERR_u[0]<2.3e-3


def test_vector_poisson_3D_dirichlet():

    u_ex = CF((cos(4*pi*x)*cos(6*pi*y),sin(2*pi*x)*sin(2*pi*y), sin(2*pi*x)*cos(4*pi*y))).Compile()

    rhs = CF((-u_ex[0].Diff(x).Diff(x) - u_ex[0].Diff(y).Diff(y) - u_ex[0].Diff(z).Diff(z),\
               -u_ex[1].Diff(x).Diff(x) - u_ex[1].Diff(y).Diff(y) - u_ex[1].Diff(z).Diff(z),\
                 -u_ex[2].Diff(x).Diff(x) - u_ex[2].Diff(y).Diff(y) - u_ex[2].Diff(z).Diff(z) )).Compile()

    ## Case with mixed boundary conditions
    dirichlet_bc = [['top', u_ex],
                        ['bottom', u_ex],
                        ['right', u_ex],
                        ['left', u_ex],
                        ['front', u_ex],
                        ['back', u_ex]]
    bnd_cond={'dirichlet': dirichlet_bc}
    fes_order = 2

    params = {'rhs': rhs,
              'boundary_c': bnd_cond}
    
    dh = 0.1
    geo = unit_cube
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    solver = VectorPoissonSolver(mesh, fes_order=fes_order, params = params)

    sol_generator = solver()
    next(sol_generator)
    
    params = {'exact_solutions': [u_ex],
              'vol_or_bnd': [VOL],
              'norms': ['L2']}
    errors = ErrorTools(solver, params=params)

    ERR_u = errors.compute_errors()

    assert ERR_u[0]<3.4e-2

def test_vector_poisson_3D_neumann():

    u_ex = CF((cos(4*pi*x)*cos(6*pi*y),sin(2*pi*x)*sin(2*pi*y), sin(2*pi*x)*cos(4*pi*y))).Compile()
    grad_u = CF((u_ex[0].Diff(x), u_ex[0].Diff(y), u_ex[0].Diff(z),\
                  u_ex[1].Diff(x), u_ex[1].Diff(y), u_ex[1].Diff(z),\
                    u_ex[2].Diff(x), u_ex[2].Diff(y), u_ex[2].Diff(z)), dims = (3,3)).Compile()

    rhs = CF((-u_ex[0].Diff(x).Diff(x) - u_ex[0].Diff(y).Diff(y) - u_ex[0].Diff(z).Diff(z),\
               -u_ex[1].Diff(x).Diff(x) - u_ex[1].Diff(y).Diff(y) - u_ex[1].Diff(z).Diff(z),\
                 -u_ex[2].Diff(x).Diff(x) - u_ex[2].Diff(y).Diff(y) - u_ex[2].Diff(z).Diff(z) )).Compile()

    ## Case with mixed boundary conditions
    neumann_bc = [['left', grad_u],
                    ['right', grad_u],
                    ['top|bottom', grad_u],
                    ['front|back', grad_u]]

    bnd_cond={'neumann': neumann_bc}
    fes_order = 2

    params = {'rhs': rhs,
              'boundary_c': bnd_cond}
    
    dh = 0.1
    geo = unit_cube
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    solver = VectorPoissonSolver(mesh, fes_order=fes_order, params = params)

    sol_generator = solver()
    next(sol_generator)
    
    params = {'exact_solutions': [u_ex],
              'vol_or_bnd': [VOL],
              'norms': ['L2']}
    errors = ErrorTools(solver, params=params)

    ERR_u = errors.compute_errors()

    assert ERR_u[0]<3.4e-2

def test_vector_poisson_3D_mixed():

    u_ex = CF((cos(4*pi*x)*cos(6*pi*y),sin(2*pi*x)*sin(2*pi*y), sin(2*pi*x)*cos(4*pi*y))).Compile()
    grad_u = CF((u_ex[0].Diff(x), u_ex[0].Diff(y), u_ex[0].Diff(z),\
                  u_ex[1].Diff(x), u_ex[1].Diff(y), u_ex[1].Diff(z),\
                    u_ex[2].Diff(x), u_ex[2].Diff(y), u_ex[2].Diff(z)), dims = (3,3)).Compile()

    rhs = CF((-u_ex[0].Diff(x).Diff(x) - u_ex[0].Diff(y).Diff(y) - u_ex[0].Diff(z).Diff(z),\
               -u_ex[1].Diff(x).Diff(x) - u_ex[1].Diff(y).Diff(y) - u_ex[1].Diff(z).Diff(z),\
                 -u_ex[2].Diff(x).Diff(x) - u_ex[2].Diff(y).Diff(y) - u_ex[2].Diff(z).Diff(z) )).Compile()

    ## Case with mixed boundary conditions
    neumann_bc = [['left', grad_u],
                    ['right', grad_u]]
    dirichlet_bc = [['top', u_ex],
                        ['bottom', u_ex],
                        ['front', u_ex],
                        ['back', u_ex]]
    bnd_cond={'neumann': neumann_bc,
            'dirichlet': dirichlet_bc}
    fes_order = 2

    params = {'rhs': rhs,
              'boundary_c': bnd_cond}
    
    dh = 0.1
    geo = unit_cube
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    solver = VectorPoissonSolver(mesh, fes_order=fes_order, params = params)

    sol_generator = solver()
    next(sol_generator)
    
    params = {'exact_solutions': [u_ex],
              'vol_or_bnd': [VOL],
              'norms': ['L2']}
    errors = ErrorTools(solver, params=params)

    ERR_u = errors.compute_errors()

    assert ERR_u[0]<3.4e-2

def test_poisson_convergence():

    u_ex = CF((cos(4*pi*x)*cos(6*pi*y),sin(2*pi*x)*sin(2*pi*y))).Compile()
    grad_u = CF((u_ex[0].Diff(x), u_ex[0].Diff(y), u_ex[1].Diff(x), u_ex[1].Diff(y)), dims = (2,2)).Compile()

    rhs = CF((-u_ex[0].Diff(x).Diff(x) - u_ex[0].Diff(y).Diff(y), -u_ex[1].Diff(x).Diff(x) - u_ex[1].Diff(y).Diff(y))).Compile()

    ## Case with mixed boundary conditions
    neumann_bc = [['left', grad_u],
                    ['right', grad_u]]
    dirichlet_bc = [['top', u_ex],
                        ['bottom', u_ex]]
    bnd_cond={'neumann': neumann_bc,
            'dirichlet': dirichlet_bc}
    fes_order = 2

    solver_params = {'rhs': rhs,
              'boundary_c': bnd_cond}
    
    dh0 = 0.1
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh0))
    
    solver = VectorPoissonSolver(mesh, fes_order=fes_order, params = solver_params)

    sol_generator = solver()
    next(sol_generator)

    error_params = {'exact_solutions': [u_ex],
              'vol_or_bnd': [VOL],
              'norms': ['L2']}
    
    errors = ErrorTools(solver, params=error_params)

    h_power = 1.5
    n_refinements = 3

    conv_prams = {
        'geometry': geo,
        'space_params': [dh0, h_power, n_refinements]
    } 

    ERRS = errors.compute_eoc(conv_prams)

    eoc = []
    for i in range(1):
        eoc.append(np.mean(np.log(ERRS[i][:-1]/ERRS[i][1:])/np.log(h_power)))

    assert math.isclose(eoc[0], 3.1, rel_tol = 5e-2) 