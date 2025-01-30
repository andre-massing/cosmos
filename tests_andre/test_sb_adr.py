# %%

from cosmos.solvers.sb_adr import SurfaceBulkADRSolver
from cosmos.solvers.base_problem import UnsteadyProblem
from cosmos.utils.manufactured_solution_tools import gradient
from cosmos.utils.generate_surface_meshes import generate_circle
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw

'''
NOTE: Here I am not dividing the flux in advective and diffusive for simplicity
since at the beginning I just wanted to test and it seems to converge.
Also, to keep things simple in the coupled case, only Neumann b.c. are allowed 
and it is given for granted that the surface is clsoed. Later we will need to modify
this if we want to simulate Mayte's article.
'''

def solve_sb_adr(dh, dt):

    T = 1
    dt = Parameter(dt)
    t = Parameter(0)

    mesh, _ = generate_circle(maxh = dh)

    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)
    fes_order = 1
    sb_sol = SurfaceBulkADRSolver(fes_order=fes_order)
    simulation.attach_solver(sb_sol)

    ## Volume part

    u_v_ex = cos(pi*x)*sin(3*y)*cos(t)
    d_v = 1+t**2
    c_v = cos(x)
    b_v = CF((2,1))
    flux1_v = (b_v*u_v_ex).Compile()
    flux2_v = (- d_v*gradient(u_v_ex, Id(2))).Compile()
    flux_v = flux1_v + flux2_v
    grad_u_v = -flux_v
    rhs_v = (u_v_ex.Diff(t) + Trace(gradient(flux_v, Id(2))) 
             + c_v*u_v_ex).Compile()
    sb_sol.attach_bulk_adr(rhs = rhs_v,
                           advection = b_v,
                           diffusion = d_v,
                           reaction = c_v,
                           u0 = u_v_ex,
                           neu_bc = grad_u_v)

    ## Coupling parameters

    alpha = CF(3.0)

    ## Surface part
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
    sb_sol.attach_surface_adr(rhs = rhs_s,
                              advection = b_s, 
                              diffusion = d_s,
                              reaction = c_s,
                              u0 = u_s_ex)

    ## Define non-linear coupling
    def f2(u1, u2):
        return -alpha*(u1**2 - u2**3)
    sb_sol.add_coupling(mrk1 = 0, mrk2 = 1, f2 = f2)

    simulation.run()

    err = sb_sol.compute_error([u_v_ex, u_s_ex], 'L2')

    for i in range(2):
        print(np.sqrt(np.sum(simulation.dt.Get()*np.array(err[i])**2)))

solve_sb_adr(0.2, 0.2)
solve_sb_adr(0.1, 0.1)
solve_sb_adr(0.05, 0.05)
