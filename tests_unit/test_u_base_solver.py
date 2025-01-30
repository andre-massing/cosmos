from cosmos.solvers.base_solver import BaseSolver
from cosmos.solvers.base_problem import SteadyProblem
from ngsolve import *

def test_u_base_solver_fes_order():

    solver = BaseSolver()

    attributes = ['fes_order', 'params', 'sol', 'problem']

    test = True
    for el in attributes:
        if not hasattr(solver,el):
            test = False

    assert test

def test_u_base_solver_sol_1():

    solver = BaseSolver()

    assert solver.sol == []

def test_u_base_solver_sol_2():

    solver = BaseSolver()

    assert solver.fes_order == 1

def test_u_base_solver_sol_3():

    solver = BaseSolver()

    assert solver.problem == None

def test_u_base_solver_initialize():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)
    solver = BaseSolver()
    simulation.attach_solver(solver)

    err = False
    try:
        solver.initialize()
    except NotImplementedError:
        err = True

    assert err

def test_u_base_solver_solve_step():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)
    solver = BaseSolver()
    simulation.attach_solver(solver)

    err = False
    try:
        solver.solve_step()
    except NotImplementedError:
        err = True

    assert err

def test_u_base_solver_update():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)
    solver = BaseSolver()
    simulation.attach_solver(solver)

    err = False
    try:
        solver.update()
    except NotImplementedError:
        err = True

    assert err

def test_u_base_solver_compute_error():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))

    u_ex = CF(0)
    norm = 'L2'
    
    simulation = SteadyProblem(mesh=mesh)
    solver = BaseSolver()
    simulation.attach_solver(solver)

    err = False
    try:
        solver.compute_error(u_ex, norm)
    except NotImplementedError:
        err = True

    assert err

def test_u_base_solver_get_solution():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)
    solver = BaseSolver()
    simulation.attach_solver(solver)

    err = False
    try:
        solver.get_solution()
    except NotImplementedError:
        err = True

    assert err

def test_u_base_solver_draw_solution():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)
    solver = BaseSolver()
    simulation.attach_solver(solver)

    err = False
    try:
        solver.draw_solution()
    except NotImplementedError:
        err = True

    assert err

def test_u_base_solver_save_solution():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)
    solver = BaseSolver()
    simulation.attach_solver(solver)

    err = False
    try:
        solver.save_solution()
    except NotImplementedError:
        err = True

    assert err

def test_u_base_solver_print_info():

    solver = BaseSolver()

    err = False
    try:
        solver.print_info()
    except NotImplementedError:
        err = True

    assert err