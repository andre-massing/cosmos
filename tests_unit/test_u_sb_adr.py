from glapypack.solvers.sb_adr import SurfaceBulkADRSolver
from glapypack.solvers.base_problem import UnsteadyProblem
from ngsolve import *

def test_u_sb_adr_solver_fes_order():

    sb_adr_sol = SurfaceBulkADRSolver()

    assert hasattr(sb_adr_sol, 'fes_order')

def test_u_sb_adr_solver_params():

    sb_adr_sol = SurfaceBulkADRSolver()

    assert hasattr(sb_adr_sol, 'params')

def test_u_sb_adr_solver_sol():

    sb_adr_sol = SurfaceBulkADRSolver()

    assert hasattr(sb_adr_sol, 'sol')

def test_u_sb_adr_solver_problem():

    sb_adr_sol = SurfaceBulkADRSolver()

    assert hasattr(sb_adr_sol, 'problem')

def test_u_sb_adr_solver_initialize():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)
    sb_adr_sol = SurfaceBulkADRSolver()
    simulation.attach_solver(sb_adr_sol)

    sb_adr_sol.attach_bulk_adr()
    sb_adr_sol.attach_surface_adr()

    test = True
    try:
        sb_adr_sol.initialize()
    except:
        test = False

    assert test

def test_u_sb_adr_solver_solve_step():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)
    sb_adr_sol = SurfaceBulkADRSolver()
    simulation.attach_solver(sb_adr_sol)

    sb_adr_sol.attach_bulk_adr()
    sb_adr_sol.attach_surface_adr()

    test = True
    try:
        sb_adr_sol.initialize()
        sb_adr_sol.solve_step()
    except Exception as e:
        print(e)
        test = False

    assert test

def test_u_sb_adr_solver_update():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)
    sb_adr_sol = SurfaceBulkADRSolver()
    simulation.attach_solver(sb_adr_sol)

    sb_adr_sol.attach_bulk_adr()
    sb_adr_sol.attach_surface_adr()

    test = True
    try:
        sb_adr_sol.initialize()
        sb_adr_sol.solve_step()
        sb_adr_sol.update()
    except:
        test = False

    assert test

def test_u_sb_adr_solver_set_solution():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)
    sb_adr_sol = SurfaceBulkADRSolver()
    simulation.attach_solver(sb_adr_sol)

    sb_adr_sol.attach_bulk_adr()
    sb_adr_sol.attach_surface_adr()

    test = True
    try:
        sb_adr_sol.initialize()
        sb_adr_sol.solve_step()
        sb_adr_sol.update()
        sb_adr_sol.set_solution(values = [CF(1), CF(0)])
    except:
        test = False

    assert test

def test_u_sb_adr_solver_get_solution():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)
    sb_adr_sol = SurfaceBulkADRSolver()
    simulation.attach_solver(sb_adr_sol)

    sb_adr_sol.attach_bulk_adr()
    sb_adr_sol.attach_surface_adr()

    test = True
    try:
        sb_adr_sol.initialize()
        sb_adr_sol.solve_step()
        sb_adr_sol.update()
        sb_adr_sol.get_solution()
    except Exception as e:
        print(e)
        test = False

    assert test

def test_u_sb_adr_solver_draw_solution():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)
    sb_adr_sol = SurfaceBulkADRSolver()
    simulation.attach_solver(sb_adr_sol)

    sb_adr_sol.attach_bulk_adr()
    sb_adr_sol.attach_surface_adr()

    test = True
    try:
        sb_adr_sol.initialize()
        sb_adr_sol.solve_step()
        sb_adr_sol.update()
        sb_adr_sol.draw_solution()
    except Exception as e:
        print(e)
        test = False

    assert test

def test_u_sb_adr_solver_save_solution():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)
    sb_adr_sol = SurfaceBulkADRSolver()
    simulation.attach_solver(sb_adr_sol)

    sb_adr_sol.attach_bulk_adr()
    sb_adr_sol.attach_surface_adr()

    test = True
    try:
        simulation.run()
        sb_adr_sol.save_solution(filename = './dummy/dummy')
    except Exception as e:
        print(e)
        test = False

    assert test

def test_u_sb_adr_solver_error():

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)
    sb_adr_sol = SurfaceBulkADRSolver()
    simulation.attach_solver(sb_adr_sol)

    sb_adr_sol.attach_bulk_adr()
    sb_adr_sol.attach_surface_adr()

    test = True
    try:
        simulation.run()
        err = sb_adr_sol.compute_error([CF(1.0), CF(0.0)], 'L2')
        err = sb_adr_sol.compute_error([CF(0.0), CF(1.0)], 'H1')
    except Exception as e:
        print(e)
        test = False

    assert test

def test_u_sb_adr_solver_print_info():
    
    sb_adr_sol = SurfaceBulkADRSolver()

    test = True
    try:
        sb_adr_sol.print_info()
    except Exception as e:
        print(e)
        test = False

    assert test