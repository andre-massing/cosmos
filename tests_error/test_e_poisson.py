from glapypack.solvers.poisson import PoissonSolver
from glapypack.solvers.base_problem import SteadyProblem
from glapypack.utils.test_tools import gradient

from ngsolve import *
import numpy as np
import math

def test_e_poisson_2D_dirichlet():
    
    dh = 0.05
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh)

    u_ex = cos(4*pi*x)*cos(6*pi*y)
    rhs = -u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y)
    fes_order = 2
    dirichlet_bc = {'top': u_ex,
                    'bottom': u_ex,
                    'right': u_ex,
                    'left': u_ex}

    p_sol = PoissonSolver(fes_order=fes_order, rhs = rhs, dir_bc = dirichlet_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(p_sol)

    simulation.run()

    err = p_sol.compute_error(u_ex, 'L2')

    assert err<2.3e-3

def test_e_poisson_2D_neumann():

    dh = 0.05
    geo = unit_square
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=1)

    u_ex = cos(4*pi*x)*cos(6*pi*y)
    grad_u = CF((u_ex.Diff(x), u_ex.Diff(y)))
    rhs = -u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y)
    fes_order = 2
    neumann_bc = {'top': grad_u,
                    'bottom': grad_u,
                    'right': grad_u,
                    'left': grad_u}

    p_sol = PoissonSolver(fes_order=fes_order, rhs = rhs, neu_bc=neumann_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(p_sol)

    simulation.run()

    err = p_sol.compute_error(u_ex, 'L2')

    assert err<2.3e-3

def test_e_poisson_2D_mixed():

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

    err = p_sol.compute_error(u_ex, 'L2')

    assert err<2.3e-3

def test_e_poisson_3D_dirichlet():
    
    dh = 0.1
    geo = unit_cube
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=1)

    u_ex = cos(4*pi*x)*cos(6*pi*y)*cos(2*pi*z)
    rhs = -u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y) - u_ex.Diff(z).Diff(z)
    fes_order = 2
    dirichlet_bc = {'top': u_ex,
                    'bottom': u_ex,
                    'right': u_ex,
                    'left': u_ex,
                    'front': u_ex,
                    'back': u_ex}

    p_sol = PoissonSolver(fes_order=fes_order, rhs = rhs, dir_bc=dirichlet_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(p_sol)

    simulation.run()

    err = p_sol.compute_error(u_ex, 'L2')

    assert err<0.025

def test_e_poisson_3D_neumann():
    
    dh = 0.1
    geo = unit_cube
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=1)

    u_ex = cos(4*pi*x)*cos(6*pi*y)*cos(2*pi*z)
    grad_u = gradient(u_ex, Id(3))
    rhs = -u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y) - u_ex.Diff(z).Diff(z)
    fes_order = 2
    neumann_bc = {'top': grad_u,
                    'bottom': grad_u,
                    'right': grad_u,
                    'left': grad_u,
                    'front': grad_u,
                    'back': grad_u}

    p_sol = PoissonSolver(fes_order=fes_order, rhs = rhs, neu_bc=neumann_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(p_sol)

    simulation.run()

    err = p_sol.compute_error(u_ex, 'L2')

    assert err<0.025

def test_e_poisson_3D_mixed():
    
    dh = 0.1
    geo = unit_cube
    mesh = Mesh(geo.GenerateMesh(maxh = dh))
    
    simulation = SteadyProblem(mesh=mesh, verbose=1)

    u_ex = cos(4*pi*x)*cos(6*pi*y)*cos(2*pi*z)
    grad_u = gradient(u_ex, Id(3))
    rhs = -u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y) - u_ex.Diff(z).Diff(z)
    fes_order = 2
    neumann_bc = {'top': grad_u,
                    'bottom': grad_u,
                    'right': grad_u}
    
    dirichlet_bc = {'left': u_ex,
                'front': u_ex,
                'back': u_ex}

    p_sol = PoissonSolver(fes_order=fes_order, rhs = rhs, neu_bc=neumann_bc, dir_bc = dirichlet_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(p_sol)

    simulation.run()

    err = p_sol.compute_error(u_ex, 'L2')

    assert err<0.025