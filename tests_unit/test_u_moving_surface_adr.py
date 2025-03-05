from cosmos.solvers.moving_surface_adr import MovingSurfaceADRSolver
from cosmos.solvers.simulation import MovingSimulation
from ngsolve import *

def test_u_surface_adr_solver_fes_order():

    adr_sol = MovingSurfaceADRSolver()

    assert hasattr(adr_sol, 'fes_order')

def test_u_surface_adr_solver_params():

    adr_sol = MovingSurfaceADRSolver()

    assert hasattr(adr_sol, 'params')

def test_u_surface_adr_solver_sol():

    adr_sol = MovingSurfaceADRSolver()

    assert hasattr(adr_sol, 'sol')

def test_u_surface_adr_solver_problem():

    adr_sol = MovingSurfaceADRSolver()

    assert hasattr(adr_sol, 'problem')

def test_u_surface_adr_solver_attributes():

    adr_sol = MovingSurfaceADRSolver()

    attributes = ['rhs', 'advection', 'diffusion', 'reaction',
                         'u0', 'dir_bc', 'flux_c_bc', 'flux_d_bc']

    test = True
    for el in attributes:
        if not el in adr_sol.params.keys():
            test = False

    assert test

def test_u_surface_adr_solver_attributes_1():

    adr_sol = MovingSurfaceADRSolver()

    attributes = ['dir_bc', 'flux_c_bc', 'flux_d_bc']
    defaults = [{}, {}, {}]

    test = True
    for el, df in zip(attributes, defaults):
        if df != adr_sol.params[el]:
            test = False

    assert test

def test_u_surface_adr_solver_initialize():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)
    adr_sol = MovingSurfaceADRSolver()
    simulation.AddSolver(adr_sol)

    test = True
    try:
        adr_sol.initialize()
    except:
        test = False

    assert test

def test_u_surface_adr_solver_solve_step():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)
    
    adr_sol = MovingSurfaceADRSolver()
    simulation.AddSolver(adr_sol)

    test = True
    try:
        adr_sol.initialize()
        adr_sol.solve_step()
    except Exception as e:
        print(e)
        test = False

    assert test

def test_u_surface_adr_solver_update():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)
    
    adr_sol = MovingSurfaceADRSolver()
    simulation.AddSolver(adr_sol)

    test = True
    try:
        adr_sol.initialize()
        adr_sol.solve_step()
        adr_sol.update()
    except:
        test = False

    assert test

def test_u_surface_adr_solver_set_solution():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)
    
    adr_sol = MovingSurfaceADRSolver()
    simulation.AddSolver(adr_sol)

    test = True
    try:
        adr_sol.initialize()
        adr_sol.solve_step()
        adr_sol.update()
        adr_sol.set_solution(value = CF(1))
    except:
        test = False

    assert test

def test_u_surface_adr_solver_get_solution():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)
    
    adr_sol = MovingSurfaceADRSolver()
    simulation.AddSolver(adr_sol)

    test = True
    try:
        adr_sol.initialize()
        adr_sol.solve_step()
        adr_sol.update()
        adr_sol.get_solution()
    except Exception as e:
        print(e)
        test = False

    assert test

def test_u_surface_adr_solver_draw_solution():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)
    
    adr_sol = MovingSurfaceADRSolver()
    simulation.AddSolver(adr_sol)

    test = True
    try:
        adr_sol.initialize()
        adr_sol.solve_step()
        adr_sol.update()
        adr_sol.draw_solution()
    except Exception as e:
        print(e)
        test = False

    assert test

def test_u_surface_adr_solver_save_solution():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)
    
    adr_sol = MovingSurfaceADRSolver()
    simulation.AddSolver(adr_sol)

    test = True
    try:
        simulation.run()
        adr_sol.save_solution(filename = './dummy/dummy')
    except Exception as e:
        print(e)
        test = False

    assert test

def test_u_surface_adr_solver_error():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)
    
    adr_sol = MovingSurfaceADRSolver()
    simulation.AddSolver(adr_sol)

    test = True
    try:
        simulation.run()
        err = adr_sol.compute_error(CF(0.0), 'L2')
        err = adr_sol.compute_error(CF(0.0), 'H1')
    except Exception as e:
        print(e)
        test = False

    assert test

def test_u_surface_adr_solver_print_info():
    
    adr_sol = MovingSurfaceADRSolver()

    test = True
    try:
        adr_sol.print_info()
    except Exception as e:
        print(e)
        test = False

    assert test