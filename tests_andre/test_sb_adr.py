# %%

from cosmos.solvers.sb_adr import SurfaceBulkADRSolver
from cosmos.solvers.simulation import UnsteadyProblem
from cosmos.utils.manufactured_solution_tools import gradient
from cosmos.utils.generate_surface_meshes import generate_circle
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw

'''
Non-deforming domain surface-bulk ADR seems ok with an expected convergence of 1.

NOTE: Here I am not dividing the flux in advective and diffusive for simplicity
since at the beginning I just wanted to test and it seems to converge even
without the splitting. Thus for the bulk only Neumann b.c. are allowed. 
It is given for granted that the surface is the entire boundary of the bulk. 
Later we will need to modify
this if we want to simulate Mayte's article.
'''

def solve_sb_adr(dh, dt):
    '''
    Generation of the mesh
    '''
    mesh, _ = generate_circle(maxh = dh)

    '''
    Definition of time variables
    '''
    T = 1
    dt = Parameter(dt)
    t = Parameter(0)

    '''
    Initialization of the time-dependent deformable simulation (that can contain multiple solvers)
    '''
    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)
    '''
    Initialization of the solver and adding it to the simulation
    '''
    fes_order = 1
    sb_sol = SurfaceBulkADRSolver(fes_order=fes_order)
    simulation.attach_solver(sb_sol)

    '''
    Generation of the manufactured solution and relative coefficients for the volume part
    '''
    u_v_ex = cos(pi*x)*sin(3*y)*cos(t)
    d_v = 1+t**2
    c_v = cos(x)
    b_v = CF((2,1))
    flux1_v = (b_v*u_v_ex).Compile()
    flux2_v = (- d_v*gradient(u_v_ex, Id(2))).Compile()
    flux_v = flux1_v + flux2_v
    rhs_v = (u_v_ex.Diff(t) + Trace(gradient(flux_v, Id(2))) 
             + c_v*u_v_ex).Compile()
    flux_c_bc = {'.*': flux1_v}
    flux_d_bc = {'.*': flux2_v}
    # Attaching this PDE to the Surface-Bulk Solver
    sb_sol.attach_bulk_adr(rhs = rhs_v,
                           advection = b_v,
                           diffusion = d_v,
                           reaction = c_v,
                           u0 = u_v_ex,
                           flux_c_bc = flux_c_bc,
                           flux_d_bc = flux_d_bc)

    ## Coupling parameter
    alpha = CF(3.0)


    '''
    Generation of the manufactured solution and relative coefficients for the surface part
    '''
    normal = CF((x,y))/Norm(CF((x,y)))
    P = Id(2) - OuterProduct(normal, normal)

    u_s_ex = sin(pi*x)*cos(3*y)*cos(2*t)
    d_s = 1+y**2
    c_s = cos(t) +1
    b_s = CF((-y,x))
    flux1_s = (b_s*u_s_ex).Compile()
    flux2_s = (- d_s*gradient(u_s_ex, P)).Compile()
    flux_s = flux1_s + flux2_s
    rhs_s = (u_s_ex.Diff(t) + Trace(gradient(flux_s, P)) + 
             c_s*u_s_ex - alpha*(u_v_ex**2 - u_s_ex**3)).Compile()
    # Attaching this PDE to the Surface-Bulk Solver
    sb_sol.attach_surface_adr(rhs = rhs_s,
                              advection = b_s, 
                              diffusion = d_s,
                              reaction = c_s,
                              u0 = u_s_ex)
    
    '''
    Definition of the non-linear coupling type
    '''
    def f2(u1, u2):
        return -alpha*(u1**2 - u2**3)
    # Attaching the non-linear coupling to the Surface-Bulk Solver
    sb_sol.add_coupling(mrk1 = 0, mrk2 = 1, f2 = f2)

    '''
    Run the simulation
    '''
    simulation.run()

    '''
    Computation and printing of the error
    '''
    err = sb_sol.compute_error([u_v_ex, u_s_ex], 'L2')
    for i in range(2):
        print(np.sqrt(np.sum(simulation.dt.Get()*np.array(err[i])**2)))

    # sb_sol.draw_solution()
    # sb_sol.save_solution(filename = 'results/sb_adr')

'''
Repeated simulation to see if there is convergence
'''
solve_sb_adr(0.2, 0.2)
solve_sb_adr(0.1, 0.1)
solve_sb_adr(0.05, 0.05)

# %%
