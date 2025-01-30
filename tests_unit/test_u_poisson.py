from glapypack.solvers.poisson import PoissonSolver
from glapypack.solvers.base_problem import SteadyProblem
from ngsolve import *

def test_u_poisson_solver_attributes():

    p_sol = PoissonSolver()

    attributes = ['rhs', 'dir_bc', 'neu_bc']

    test = True
    for el in attributes:
        if not el in p_sol.params.keys():
            test = False

    assert test

def test_u_poisson_solver_attributes_val():

    p_sol = PoissonSolver()

    attributes = ['dir_bc', 'neu_bc']
    defaults = [{}, {}]

    test = True
    for el, df in zip(attributes, defaults):
        if df != p_sol.params[el]:
            test = False

    assert test

def test_u_poisson_solver_initialize():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=0)
    p_sol = PoissonSolver()
    simulation.attach_solver(p_sol)

    test = True
    try:
        p_sol.initialize()
    except:
        test = False

    assert test

def test_u_poisson_solver_solve_step():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=0)
    p_sol = PoissonSolver()
    simulation.attach_solver(p_sol)

    test = True
    try:
        p_sol.initialize()
        p_sol.solve_step()
    except:
        test = False

    assert test

def test_u_poisson_solver_update():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=0)
    p_sol = PoissonSolver()
    simulation.attach_solver(p_sol)

    test = True
    try:
        p_sol.initialize()
        p_sol.solve_step()
        p_sol.update()
    except:
        test = False

    assert test

def test_u_poisson_solver_set_solution():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=0)
    p_sol = PoissonSolver()
    simulation.attach_solver(p_sol)

    test = False
    try:
        p_sol.initialize()
        p_sol.solve_step()
        p_sol.update()
        p_sol.set_solution()
    except:
        test = True

    assert test

def test_u_poisson_solver_get_solution():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=0)
    p_sol = PoissonSolver()
    simulation.attach_solver(p_sol)

    test = True
    try:
        p_sol.initialize()
        p_sol.solve_step()
        p_sol.update()
        p_sol.get_solution()
    except:
        test = False

    assert test

def test_u_poisson_solver_draw_solution():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=0)
    p_sol = PoissonSolver()
    simulation.attach_solver(p_sol)

    test = True
    try:
        p_sol.initialize()
        p_sol.solve_step()
        p_sol.update()
        p_sol.draw_solution()
    except:
        test = False

    assert test

def test_u_poisson_solver_save_solution():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=0) 
    p_sol = PoissonSolver()
    simulation.attach_solver(p_sol)

    test = True
    try:
        p_sol.initialize()
        p_sol.solve_step()
        p_sol.update()
        p_sol.save_solution(filename = './dummy/dummy')
    except:
        test = False

    assert test

def test_u_poisson_solver_error():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=0)
    p_sol = PoissonSolver()
    simulation.attach_solver(p_sol)

    test = True
    try:
        p_sol.initialize()
        p_sol.solve_step()
        p_sol.update()
        err = p_sol.compute_error(CF(0.0), 'L2')
        err = p_sol.compute_error(CF(0.0), 'H1')
    except:
        test = False

    assert test

def test_u_poisson_solver_print_info():
    
    p_sol = PoissonSolver()

    test = True
    try:
        p_sol.print_info()
    except:
        test = False

    assert test