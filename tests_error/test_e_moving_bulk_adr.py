from glapypack.solvers.moving_bulk_adr import MovingBulkADRSolver
from glapypack.solvers.base_problem import MovingProblem
from glapypack.utils.test_tools import gradient, get_lin_trans_params
from glapypack.utils.meshes import generate_ball, generate_circle
from ngsolve import *
import numpy as np

def test_e_moving_bulk_adr_2D_dirichlet():
    
    mesh, _ = generate_circle(maxh=0.1, R = 1, order_g =1)
    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

    # General affine transformation (sphere to ellipse)
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t),\
                sin(t), (1-0.25*sin(t))*cos(t)), dims = (2,2))
    B = CF((0.2*t, 0.1*t))

    phi = A*CF((x,y)) + B
    detJ = Det(A)
    invA = Cof(A).trans/detJ
    inv_phi = invA*(CF((x,y)) - B)
    w_phi = A.Diff(t)*inv_phi + B.Diff(t)
    displ_ex = phi - CF((x,y))

    def displacement():
        return displ_ex

    simulation.set_displacement(displacement)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + x**2
    c = cos(x)
    b = CF((2,1))
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, Id(2))).Compile()
    rel_flux = (w_phi*u_ex).Compile()
    flux = flux1 + flux2 + rel_flux
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, Id(2))) + c*u_ex).Compile()
    fes_order = 2
    dirichlet_bc = {'.*': u_ex}

    adr_sol = MovingBulkADRSolver(fes_order=fes_order,
                                  rhs = rhs,
                                  advection = b,
                                  diffusion = d, 
                                  reaction = c,
                                  u0 = u_ex, 
                                  dir_bc = dirichlet_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(adr_sol)

    simulation.run()

    err = adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    assert err<7e-3

def test_e_moving_bulk_adr_2D_neumann():
    
    mesh, _ = generate_circle(maxh=0.1, R = 1, order_g =1)
    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

    # General affine transformation (sphere to ellipse)
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t),\
                sin(t), (1-0.25*sin(t))*cos(t)), dims = (2,2))
    B = CF((0.2*t, 0.1*t))

    phi = A*CF((x,y)) + B
    detJ = Det(A)
    invA = Cof(A).trans/detJ
    inv_phi = invA*(CF((x,y)) - B)
    w_phi = A.Diff(t)*inv_phi + B.Diff(t)
    displ_ex = phi - CF((x,y))

    def displacement():
        return displ_ex

    simulation.set_displacement(displacement)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + x**2
    c = cos(x)
    b = CF((2,1))
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, Id(2))).Compile()
    rel_flux = (w_phi*u_ex).Compile()
    flux = flux1 + flux2 + rel_flux
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, Id(2))) + c*u_ex).Compile()
    fes_order = 1
    flux_c_bc = {'.*': flux1}
    flux_d_bc = {'.*': flux2}

    adr_sol = MovingBulkADRSolver(fes_order=fes_order,
                                  rhs = rhs,
                                  advection = b,
                                  diffusion = d, 
                                  reaction = c,
                                  u0 = u_ex,
                                  flux_c_bc=flux_c_bc,
                                  flux_d_bc=flux_d_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(adr_sol)

    simulation.run()

    err = adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    assert err<0.0