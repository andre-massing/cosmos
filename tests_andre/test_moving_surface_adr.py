# %%

from cosmos.solvers.moving_surface_adr import MovingSurfaceADRSolver
from cosmos.solvers.simulation import MovingProblem
from cosmos.utils.manufactured_solution_tools import gradient, get_lin_trans_params
from cosmos.utils.generate_surface_meshes import *
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw

'''
Deforming domain surface ADR:
- All simulations seem to show 1st order convergence which is the expected
'''

'''
Deforming domain surface ADR simulation with no boundary
'''

def solve_m_surface_adr_closed(dh, dt):

    '''
    Generation of the mesh
    '''
    mesh, _ = generate_sphere(maxh=dh, R = 1)

    '''
    Definition of time variables
    '''
    T = 1.0
    dt = Parameter(dt)
    t = Parameter(0)

    '''
    Initialization of the time-dependent deformable simulation (that can contain multiple solvers)
    '''
    simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

    '''
    Generation of the deformation properties
    I consider a linear transformation of a sphere/ball
    '''
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t), 0,\
                sin(t), (1-0.25*sin(t))*cos(t), 0,\
                    0, 0, 1), dims = (3,3))
    B = CF((0.2*t, 0.1*t, 0))
    phi, inv_phi, w_phi, detJ, n_ex = get_lin_trans_params(A, B, t)
    # phi is the deformation map
    # inv_phi is its inverse
    # w_phi is the velocity of the domain
    # n_ex is the normal to the surface
    P_ex = Id(3) - OuterProduct(n_ex, n_ex)
    displ_ex = phi - CF((x,y,z))

    '''
    Prescribing the displacement I want. In this case it's given explicitly
    '''
    def displacement():
        return displ_ex
    simulation.set_displacement(displacement)

    '''
    Generation of the manufactured solution and relative coefficients
    '''
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

    '''
    Creation of the solver with relative parameters
    '''
    adr_sol = MovingSurfaceADRSolver(fes_order=fes_order,
                                        rhs = rhs,
                                        advection = b,
                                        diffusion = d,
                                        reaction = c,
                                        u0 = u_ex)

    '''
    Adding the solver to the simulation
    '''
    simulation.attach_solver(adr_sol)

    '''
    Run the simulation
    '''
    simulation.run()

    '''
    Computation and printing of the error
    '''
    err = adr_sol.compute_error(u_ex, 'L2')
    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))
    print(err)

    # adr_sol.draw_solution()
    # adr_sol.save_solution(filename = 'results/moving_surface_adr_closed')

'''
Repeated simulation to see if there is convergence
'''
solve_m_surface_adr_closed(0.2, 0.2)
solve_m_surface_adr_closed(0.1, 0.1)
solve_m_surface_adr_closed(0.05, 0.05)

#%%

from cosmos.solvers.moving_surface_adr import MovingSurfaceADRSolver
from cosmos.solvers.simulation import MovingProblem
from cosmos.utils.manufactured_solution_tools import gradient, get_lin_trans_params
from cosmos.utils.generate_surface_meshes import *
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw

'''
Deforming domain surface ADR simulation with no boundary
'''

def solve_m_surface_adr_open(dh, dt, neu = False):

    '''
    Generation of the mesh
    '''
    mesh, _ = generate_half_sphere(maxh=dh, R = 1)

    '''
    Definition of time variables
    '''
    T = 1.0
    dt = Parameter(dt)
    t = Parameter(0)

    '''
    Initialization of the time-dependent deformable simulation (that can contain multiple solvers)
    '''
    simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

    '''
    Generation of the manufactured solution and relative coefficients
    I consider a linear transformation of a sphere/ball
    '''
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t), 0,\
                sin(t), (1-0.25*sin(t))*cos(t), 0,\
                    0, 0, 1), dims = (3,3))
    B = CF((0.2*t, 0.1*t, 0))

    phi, inv_phi, w_phi, detJ, n_ex = get_lin_trans_params(A, B, t)
    # parameters explained above
    P_ex = Id(3) - OuterProduct(n_ex, n_ex)
    displ_ex = phi - CF((x,y,z))

    '''
    Prescribing the displacement I want. In this case it's given explicitly
    '''
    def displacement():
        return displ_ex
    simulation.set_displacement(displacement)

    '''
    Generation of the manufactured solution and relative coefficients
    '''
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
    flux_c_bc = {'bottom': flux1}
    flux_d_bc = {'bottom': flux2}
    dirichlet_bc = {'bottom': u_ex}

    '''
    Creation of the solver with relative parameters based on boundary conditions imposed
    '''
    if neu:

        adr_sol = MovingSurfaceADRSolver(fes_order=fes_order,
                                        rhs = rhs,
                                        advection = b,
                                        diffusion = d,
                                        reaction = c,
                                        u0 = u_ex,
                                        flux_c_bc=flux_c_bc,
                                        flux_d_bc=flux_d_bc)
        
    else:

        adr_sol = MovingSurfaceADRSolver(fes_order=fes_order,
                                        rhs = rhs,
                                        advection = b,
                                        diffusion = d,
                                        reaction = c,
                                        u0 = u_ex,
                                        dir_bc = dirichlet_bc)

    '''
    Adding the solver to the simulation
    '''
    simulation.attach_solver(adr_sol)

    '''
    Run the simulation
    '''
    simulation.run()

    '''
    Computation and printing of the error
    '''
    err = adr_sol.compute_error(u_ex, 'L2')
    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))
    print(err)

    # adr_sol.draw_solution()
    # adr_sol.save_solution(filename = 'results/moving_surface_adr_open')

neu = True
solve_m_surface_adr_open(0.2, 0.2, neu)
solve_m_surface_adr_open(0.1, 0.1, neu)
solve_m_surface_adr_open(0.05, 0.05, neu)
# %%
