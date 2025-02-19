# %%
 
from cosmos.solvers.moving_bulk_adr import MovingBulkADRSolver
from cosmos.solvers.base_problem import MovingProblem
from cosmos.utils.manufactured_solution_tools import gradient
from cosmos.utils.generate_surface_meshes import *
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw

'''
Deforming domain bulk ADR:
- dirichlet boundary conditions seem ok
- neumann boundary conditions are not
'''

def solve_m_bulk_adr(dh, dt, neu = False):


    '''
    Generation of the mesh
    '''
    mesh, _ = generate_circle(maxh=dh, R = 1)

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
    I consider a linear transformation of a circle
    '''
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t),\
                sin(t), (1-0.25*sin(t))*cos(t)), dims = (2,2))
    B = CF((0.2*t, 0.1*t))
    phi = A*CF((x,y)) + B # Deformation map
    detJ = Det(A)
    invA = Cof(A).trans/detJ
    inv_phi = invA*(CF((x,y)) - B) # Inverse of deformation map
    w_phi = A.Diff(t)*inv_phi + B.Diff(t) # velocity of the moving domain
    displ_ex = phi - CF((x,y)) # exact displacement

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
    c = 1 + x**2
    b = CF((sin(x), 1))
    flux = b*u_ex - d*gradient(u_ex, Id(2)) + w_phi*u_ex
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, Id(2))) + c*u_ex).Compile()
    fes_order = 1
    flux_c = b*u_ex
    flux_d = - d*gradient(u_ex, Id(2))
    flux_c_bc = {'.*': flux_c}
    flux_d_bc = {'.*': flux_d}
    dirichlet_bc = {'.*': u_ex}

    '''
    Creation of the solver with relative parameters based on boundary conditions imposed
    '''
    if neu:

        adr_sol = MovingBulkADRSolver(fes_order=fes_order,
                              rhs = rhs,
                              advection = b,
                              diffusion = d,
                              reaction = c,
                              u0 = u_ex,
                              flux_c_bc=flux_c_bc,
                              flux_d_bc=flux_d_bc)
    else:
        adr_sol = MovingBulkADRSolver(fes_order=fes_order,
                              rhs = rhs,
                              advection = b,
                              diffusion = d,
                              reaction = c,
                              u0 = u_ex,
                              dir_bc=dirichlet_bc)
        
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
    # adr_sol.save_solution(filename = './moving_bulk_adr/moving_bulk_adr')

'''
Repeated simulation to see if there is convergence
'''
neu = True
solve_m_bulk_adr(0.2, 0.2, neu)
solve_m_bulk_adr(0.1, 0.1, neu)
solve_m_bulk_adr(0.05, 0.05, neu)
# solve_m_bulk_adr(0.025, 0.025, neu)
# %%
