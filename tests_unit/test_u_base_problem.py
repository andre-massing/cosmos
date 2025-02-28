from cosmos.solvers.simulation import SteadyProblem, UnsteadyProblem, MovingProblem
from cosmos.solvers.base_solver import BaseSolver
from cosmos.solvers.base import Base
from ngsolve import *
from netgen.csg import *

def test_u_steady_problem_attributes():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)

    attributes = ['mesh', 'params', 'solvers', 'data',
                  'domain_markers', 'boundary_markers']

    test = True
    for el in attributes:
        if not hasattr(simulation, el):
            test = False

    assert test

def test_u_steady_problem_attributes_defaults():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)

    assert simulation.params['verbose'] == 0 and simulation.solvers == []

def test_u_steady_problem_data():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=1)

    attributes = ['mesh', 'verbosity', 'solvers',
                  'domain_mrk', 'boundary_mrk']

    test = True
    for el in attributes:
        if not el in simulation.data.keys():
            test = False

    assert test

def test_u_steady_problem_verbose():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)

    assert simulation.data['verbosity'] == 0

def test_u_steady_problem_domain_markers():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)

    assert simulation.domain_markers == ['default'] and \
            simulation.boundary_markers == ['bottom', 'right', 'top', 'left']
    
def test_u_steady_problem_domain_markers_1():

    geo          = CSGeometry()
    sphere       = Sphere(Pnt(0,0,0), 1)
    bot          = Plane(Pnt(0,0,0), Vec(0,0,-1))
    finitesphere = sphere * bot

    geo.AddSurface(sphere, finitesphere.bc("surface"))
    geo.NameEdge(sphere,bot, "bottom")

    mesh = Mesh(geo.GenerateMesh(maxh=0.5))
        
    simulation = SteadyProblem(mesh=mesh)

    assert simulation.domain_markers == ['surface'] and \
            simulation.boundary_markers == ['bottom']

def test_u_steady_problem_set_verbose():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))

    simulation = SteadyProblem(mesh=mesh)

    test = False
    try:
        simulation.set_verbosity('test')
    except ValueError:
        simulation.set_verbosity(1)
        test = True

    assert simulation.data['verbosity'] == 1 and test

def test_u_steady_problem_print_mesh_info():

    geo          = CSGeometry()
    sphere       = Sphere(Pnt(0,0,0), 1)
    bot          = Plane(Pnt(0,0,0), Vec(0,0,-1))
    finitesphere = sphere * bot

    geo.AddSurface(sphere, finitesphere.bc("surface"))
    geo.NameEdge(sphere,bot, "bottom")

    mesh = Mesh(geo.GenerateMesh(maxh=0.3))
    
    simulation = SteadyProblem(mesh=mesh)

    test = True
    try:
        simulation.print_mesh_info()
    except:
        test = False

    assert test

def test_u_steady_problem_attach_solver():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=1)

    base = Base()
    sol = BaseSolver(fes_order=1)

    test = False
    try:
        simulation.attach_solver(base)
    except TypeError:
        try:
            simulation.attach_solver(sol)
        except NotImplementedError:
            test = True

    assert len(simulation.solvers) == 1 and test

def test_u_steady_problem_initialize():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=0)

    test = True
    try:
        simulation.initialize()
    except:
        test = False

    assert test

def test_u_steady_problem_solve_step():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)

    test = True
    try:
        simulation.solve_step()
    except:
        test = False

    assert test

def test_u_steady_problem_run():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=0)

    sol = BaseSolver(fes_order=1)

    simulation.attach_solver(sol)

    test = False
    try:
        simulation.run()
    except NotImplementedError:
        test = True

    assert test

def test_u_steady_problem_post_process():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=0)

    test = True
    try:
        simulation.post_process()
    except:
        test = False

    assert test

def test_u_unsteady_problem_attributes():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))

    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = UnsteadyProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    attributes = ['dt', 'T', 't']

    test = True
    for el in attributes:
        if not hasattr(simulation, el):
            test = False

    assert test

def test_u_unsteady_problem_data_val():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = UnsteadyProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    attributes = ['t', 'dt', 't_array']

    test = True
    for el in attributes:
        if not el in simulation.data.keys():
            test = False

    assert test

def test_u_unsteady_problem_run():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))

    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = UnsteadyProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=0)

    test = True
    try:
        simulation.run()
    except:
        test = False

    assert test

def test_u_unsteady_problem_post_process():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))

    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = UnsteadyProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=0)

    simulation.post_process()

    assert len(simulation.data["t_array"]) == 2


def test_u_unsteady_problem_initialize():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = UnsteadyProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    test = True
    try:
        simulation.initialize()
    except:
        test = False

    assert test

def test_u_unsteady_problem_run():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = UnsteadyProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    test = True
    try:
        simulation.run()
    except:
        test = False

    assert test

def test_u_unsteady_problem_post_process():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = UnsteadyProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    simulation.post_process()
    
    test = len(simulation.data['t_array']) == 2

    assert test

def test_u_moving_problem_attributes():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))

    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    attributes = ['displ', 'displ_old', 'displ_solvers', 'mm_solvers']

    test = True
    for el in attributes:
        if not hasattr(simulation, el):
            test = False

    assert test

def test_u_moving_problem_data_val():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    
    attributes = ['dX', 'dX_old', 'dX_array', 'f_dX']

    test = True
    for el in attributes:
        if not el in simulation.data.keys():
            test = False

    assert test

def test_u_moving_problem():

    geo          = CSGeometry()
    sphere       = Sphere(Pnt(0,0,0), 1)
    bot          = Plane(Pnt(0,0,0), Vec(0,0,-1))
    finitesphere = sphere * bot

    geo.AddSurface(sphere, finitesphere.bc("surface"))
    geo.NameEdge(sphere,bot, "bottom")

    mesh = Mesh(geo.GenerateMesh(maxh=0.3))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)

    test = True
    try:
        simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T)
    except:
        test = False

    assert test

def test_u_moving_problem_initialize():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    test = True
    try:
        simulation.initialize()
    except:
        test = False

    assert test

def test_u_moving_problem_run():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    test = True
    try:
        simulation.run()
    except:
        test = False

    assert test

def test_u_moving_problem_post_process():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    simulation.post_process()

    test = len(simulation.data['t_array']) == 2 and \
        len(simulation.data['dX_array']) == 2

    assert test

def test_u_moving_problem_set_displacement():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    def f():
        return True
    
    test = False
    try:
        simulation.set_displacement('test')
    except ValueError:
        simulation.set_displacement(f)
        test = True

    assert simulation.data['f_dX']() and test

def test_u_moving_problem_set_displacement_bnd():

    geo          = CSGeometry()
    sphere       = Sphere(Pnt(0,0,0), 1)
    bot          = Plane(Pnt(0,0,0), Vec(0,0,-1))
    finitesphere = sphere * bot

    geo.AddSurface(sphere, finitesphere.bc("surface"))
    geo.NameEdge(sphere,bot, "bottom")

    mesh = Mesh(geo.GenerateMesh(maxh=0.3))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    def f():
        return True
    
    test = False
    try:
        simulation.set_displacement('test')
    except ValueError:
        simulation.set_displacement(f)
        test = True

    assert simulation.data['f_dX']() and test

def test_u_moving_problem_displ_step():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    def f():
        return CF((x, y))

    simulation.set_displacement(f)

    test = True
    try:
        simulation.displ_step()
    except:
        test = False

    assert test

def test_u_moving_problem_attach_displ_solver():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T, verbose = 1)

    sol = BaseSolver()
    base = Base()

    test = False
    try:
        simulation.attach_displ_solver(base)
    except TypeError:
        try:
            simulation.attach_displ_solver(sol)
        except NotImplementedError:
            test = True

    assert test

def test_u_moving_problem_mm_step():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=0)

    test = True
    try:
        simulation.mm_step()
    except:
        test = False

    assert test

def test_u_moving_problem_attach_mm_solver():

    dh = 0.5
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    dt = Parameter(0.1)
    T = 1.0
    t = Parameter(0)
    
    simulation = MovingProblem(mesh=mesh, dt = dt, t=t, T = T, verbose=1)

    sol = BaseSolver()
    base = Base()

    test = False
    try:
        simulation.attach_mm_solver(base)
    except TypeError:
        try:
            simulation.attach_mm_solver(sol)
        except NotImplementedError:
            test = True

    assert test