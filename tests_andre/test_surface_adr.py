# %%

from cosmos.solvers.surface_adr import SurfaceADRSolver
from cosmos.solvers.base_problem import UnsteadyProblem
from cosmos.utils.manufactured_solution_tools import gradient
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw
import netgen.occ as occ
import netgen.csg as csg

'''
Case of closed boundary
'''

def solve_surface_adr_closed(dh, dt):
    sphere = occ.Sphere((0,0,0),1).faces[0]
    geo = occ.OCCGeometry(sphere)
    mesh = Mesh(geo.GenerateMesh(maxh=dh))

    T = 1.0
    dt = Parameter(dt)
    t = Parameter(0)

    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1+t
    c = cos(x)
    b = CF((-z,0,x))
    normal = CF((x,y,z))/Norm(CF((x,y,z)))
    P = Id(3) - OuterProduct(normal, normal)
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, P)).Compile()
    flux = flux1 + flux2
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, P)) + c*u_ex).Compile()
    fes_order = 1

    adr_sol = SurfaceADRSolver(fes_order=fes_order,
                            rhs = rhs,
                            advection = b,
                                diffusion = d,
                                reaction = c,
                                u0 = u_ex)
    # Adding the solver to the simulation
    simulation.attach_solver(adr_sol)

    simulation.run()

    err = adr_sol.compute_error(u_ex, 'L2')

    print(np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2)))

    # adr_sol.draw_solution()

solve_surface_adr_closed(0.2, 0.2)
solve_surface_adr_closed(0.1, 0.1)
solve_surface_adr_closed(0.05, 0.05)

# %%

'''
Case of open boundary
'''

def solve_surface_adr_open(dh, dt, neu = False):

    geo          = csg.CSGeometry()
    sphere       = csg.Sphere(csg.Pnt(0,0,0), 1)
    bot          = csg.Plane(csg.Pnt(0,0,0), csg.Vec(0,0,-1))
    finitesphere = sphere*bot

    geo.AddSurface(sphere, finitesphere.bc("surface"))
    geo.NameEdge(sphere,bot, "bottom")

    mesh = Mesh(geo.GenerateMesh(maxh=dh))
    T = 1.0
    dt = Parameter(dt)
    t = Parameter(0)

    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    normal = CF((x,y,z))/Norm(CF((x,y,z)))
    P = Id(3) - OuterProduct(normal, normal)
    d = 1 + x**2
    c = cos(x)
    b = P*CF((-z,0,x))
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, P)).Compile()
    flux = flux1 + flux2
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, P)) + c*u_ex).Compile()
    fes_order = 1
    flux_c_bc = {'bottom': flux1}
    flux_d_bc = {'bottom': flux2}
    dirichlet_bc = {'bottom': u_ex}

    if neu:

        adr_sol = SurfaceADRSolver(fes_order=fes_order,
                                   rhs = rhs,
                                   advection = b,
                                   diffusion = d,
                                   reaction = c,
                                   u0 = u_ex,
                                   flux_c_bc=flux_c_bc,
                                   flux_d_bc=flux_d_bc)
        
    else:

        adr_sol = SurfaceADRSolver(fes_order=fes_order,
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

    print(np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2)))

neu = True
solve_surface_adr_open(0.2, 0.2, neu)
solve_surface_adr_open(0.1, 0.1, neu)
solve_surface_adr_open(0.05, 0.05, neu)
# %%
