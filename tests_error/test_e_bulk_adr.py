from cosmos.solvers.bulk_adr import BulkADRSolver
from cosmos.solvers.simulation import UnsteadyProblem
from cosmos.utils.manufactured_solution_tools import gradient
from ngsolve import *
import numpy as np

def test_e_bulk_adr_2D_dirichlet():
    
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
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, Id(2))) + c*u_ex).Compile()
    fes_order = 1
    dirichlet_bc = {'top': u_ex,
                    'bottom': u_ex,
                    'right': u_ex,
                    'left': u_ex}

    bulk_adr_sol = BulkADRSolver(fes_order=fes_order,
                                 rhs = rhs,
                                 advection = b,
                                 diffusion = d,
                                 reaction = c,
                                 u0 = u_ex,
                                 dir_bc=dirichlet_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(bulk_adr_sol)

    simulation.run()

    err = bulk_adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    assert err<0.014

def test_e_bulk_adr_2D_neumann():
    
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
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, Id(2))) + c*u_ex).Compile()
    fes_order = 1
    flux_c_bc = {'top': flux1,
                    'bottom': flux1,
                    'right': flux1,
                    'left': flux1}
    flux_d_bc = {'top': flux2,
                    'bottom': flux2,
                    'right': flux2,
                    'left': flux2}

    bulk_adr_sol = BulkADRSolver(fes_order=fes_order,
                                 rhs = rhs,
                                 advection = b,
                                 diffusion = d,
                                 reaction = c,
                                 u0 = u_ex,
                                 flux_c_bc=flux_c_bc,
                                 flux_d_bc=flux_d_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(bulk_adr_sol)

    simulation.run()

    err = bulk_adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    assert err<0.0021

def test_e_bulk_adr_2D_mixed():
    
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

    err = bulk_adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    assert err<0.00221