from cosmos.solvers.surface_adr import SurfaceADRSolver
from cosmos.solvers.base_problem import UnsteadyProblem
from cosmos.utils.manufactured_solution_tools import gradient
from cosmos.utils.generate_surface_meshes import generate_circle
from ngsolve import *
import netgen.occ as occ
import netgen.csg as csg
import numpy as np

def test_g_surface_adr_vol():
    
    mesh, _ = generate_circle()

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)

    normal = CF((x,y))/Norm(CF((x,y)))
    P = Id(2) - OuterProduct(normal, normal)
    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + t
    c = cos(y)
    b = P*CF((-y, x))
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, P)).Compile()
    flux = flux1 + flux2
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, P)) + c*u_ex).Compile()
    fes_order = 1

    surf_adr_sol = SurfaceADRSolver(fes_order=fes_order,
                                    rhs = rhs,
                                    advection = b,
                                    diffusion = d,
                                    reaction = c,
                                    u0 = u_ex)
    # Adding the solver to the simulation
    simulation.attach_solver(surf_adr_sol)

    simulation.run()

    surf_adr_sol.save_solution(filename='./results/surf_adr/surf_adr_vol')

    assert True

def test_f_surface_adr_bnd():
    
    dh = 0.1
    geo          = csg.CSGeometry()
    sphere       = csg.Sphere(csg.Pnt(0,0,0), 1)
    bot          = csg.Plane(csg.Pnt(0,0,0), csg.Vec(0,0,-1))
    finitesphere = sphere * bot
    geo.AddSurface(sphere, finitesphere.bc("surface"))
    geo.NameEdge(sphere,bot, "bottom")
    mesh = Mesh(geo.GenerateMesh(maxh=dh))

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = UnsteadyProblem(mesh=mesh, dt=dt, t=t, T=T)

    normal = CF((x,y,z))/Norm(CF((x,y,z)))
    P = Id(3) - OuterProduct(normal, normal)
    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + t
    c = cos(y)
    b = P*CF((-z,0,x))
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, P)).Compile()
    flux = flux1 + flux2
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, P)) + c*u_ex).Compile()
    fes_order = 1
    flux_c_bc = {'bottom': flux1}
    flux_d_bc = {'bottom': flux2}

    surf_adr_sol = SurfaceADRSolver(fes_order=fes_order,
                                    rhs = rhs,
                                    advection = b,
                                    diffusion = d,
                                    reaction = c,
                                    u0 = u_ex,
                                    flux_c_bc=flux_c_bc, flux_d_bc=flux_d_bc)
    # Adding the solver to the simulation
    simulation.attach_solver(surf_adr_sol)

    simulation.run()

    surf_adr_sol.save_solution(filename='./results/surf_adr/surf_adr_bnd')

    assert True