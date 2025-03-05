from cosmos.solvers.moving_surface_adr import MovingSurfaceADRSolver
from cosmos.solvers.simulation import MovingSimulation
from cosmos.utils.manufactured_solution_tools import gradient, get_lin_trans_params
from cosmos.utils.generate_surface_meshes import generate_sphere, generate_half_sphere
from ngsolve import *
import numpy as np

def test_e_moving_surface_adr_3D_closed():
    
    mesh, _ = generate_sphere(maxh=0.1, R = 1, order_g =1)
    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)

    # General affine transformation (sphere to ellipse)
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t), 0,\
                sin(t), (1-0.25*sin(t))*cos(t), 0,\
                    0, 0, 1), dims = (3,3))
    B = CF((0.2*t, 0.1*t, 0))

    phi, inv_phi, w_phi, detJ, n_ex = get_lin_trans_params(A, B, t)
    P_ex = Id(3) - OuterProduct(n_ex, n_ex)
    displ_ex = phi - CF((x,y,z))

    def displacement():
        return displ_ex

    simulation.set_displacement(displacement)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + x**2
    c = cos(x)
    b = P_ex*CF((2,1,0))
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, P_ex)).Compile()
    rel_flux = u_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_ex, Id(3))
    flux = flux1 + flux2
    rhs = (u_ex.Diff(t) + rel_flux + Trace(gradient(flux, P_ex)) + c*u_ex).Compile()
    fes_order = 1
    params = {'rhs': rhs,
                'advection': b,
                'diffusion': d,
                'reaction': c,
                'u0': u_ex}

    adr_sol = MovingSurfaceADRSolver(fes_order=fes_order,
                                     rhs = rhs,
                                     advection = b,
                                     diffusion = d,
                                     reaction = c,
                                     u0 = u_ex)
    # Adding the solver to the simulation
    simulation.AddSolver(adr_sol)

    simulation.run()

    err = adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    assert err<0.053

def test_e_moving_surface_adr_3D_dirichlet():
    
    mesh, _ = generate_half_sphere(maxh=0.1, R = 1, order_g =1)
    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)

    # General affine transformation (sphere to ellipse)
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t), 0,\
                sin(t), (1-0.25*sin(t))*cos(t), 0,\
                    0, 0, 1), dims = (3,3))
    B = CF((0.2*t, 0.1*t, 0))

    phi, inv_phi, w_phi, detJ, n_ex = get_lin_trans_params(A, B, t)
    P_ex = Id(3) - OuterProduct(n_ex, n_ex)
    displ_ex = phi - CF((x,y,z))

    def displacement():
        return displ_ex

    simulation.set_displacement(displacement)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + x**2
    c = cos(x)
    b = P_ex*CF((2,1,0))
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, P_ex)).Compile()
    rel_flux = u_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_ex, Id(3))
    flux = flux1 + flux2 
    rhs = (u_ex.Diff(t) + rel_flux + Trace(gradient(flux, P_ex)) + c*u_ex).Compile()
    fes_order = 1
    params = {'rhs': rhs,
                'advection': b,
                'diffusion': d,
                'reaction': c,
                'u0': u_ex}
    dirichlet_bc = {'.*': u_ex}

    adr_sol = MovingSurfaceADRSolver(fes_order=fes_order,
                                     rhs = rhs,
                                     advection = b,
                                     diffusion = d,
                                     reaction = c,
                                     u0 = u_ex,
                                     dir_bc = dirichlet_bc)
    # Adding the solver to the simulation
    simulation.AddSolver(adr_sol)

    simulation.run()

    err = adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    assert err<0.026

def test_e_moving_surface_adr_3D_neumann():
    
    mesh, _ = generate_half_sphere(maxh=0.1, R = 1, order_g =1)
    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)

    # General affine transformation (sphere to ellipse)
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t), 0,\
                sin(t), (1-0.25*sin(t))*cos(t), 0,\
                    0, 0, 1), dims = (3,3))
    B = CF((0.2*t, 0.1*t, 0))

    phi, inv_phi, w_phi, detJ, n_ex = get_lin_trans_params(A, B, t)
    P_ex = Id(3) - OuterProduct(n_ex, n_ex)
    displ_ex = phi - CF((x,y,z))

    def displacement():
        return displ_ex

    simulation.set_displacement(displacement)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + x**2
    c = cos(x)
    b = P_ex*CF((2,1,0))
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, P_ex)).Compile()
    rel_flux = u_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_ex, Id(3))
    flux = flux1 + flux2 
    rhs = (u_ex.Diff(t) + rel_flux + Trace(gradient(flux, P_ex)) + c*u_ex).Compile()
    fes_order = 1
    flux_c_bc = {'.*': flux1}
    flux_d_bc = {'.*': flux2}

    adr_sol = MovingSurfaceADRSolver(fes_order=fes_order,
                                     rhs = rhs,
                                     advection = b,
                                     diffusion = d,
                                     reaction = c,
                                     u0 = u_ex,
                                     flux_c_bc=flux_c_bc,
                                     flux_d_bc=flux_d_bc)
    # Adding the solver to the simulation
    simulation.AddSolver(adr_sol)

    simulation.run()

    err = adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    assert err<0.036