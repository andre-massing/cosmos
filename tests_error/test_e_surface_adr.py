from cosmos.solvers.surface_adr import SurfaceADRSolver
from cosmos.solvers.simulation import Simulation
from cosmos.utils.manufactured_solution_tools import gradient
from ngsolve import *
import netgen.occ as occ
import netgen.csg as csg
import numpy as np

def test_e_surface_adr_closed():
    
    dh = 0.1
    sphere = occ.Sphere((0,0,0),1).faces[0]
    sphere.name = "sphere"
    geo = occ.OCCGeometry(sphere)
    mesh = Mesh(geo.GenerateMesh(maxh=dh))

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = Simulation(mesh=mesh, dt=dt, t=t, T=T)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + t
    c = cos(x)
    b = CF((-z,0,x))
    normal = CF((x,y,z))/Norm(CF((x,y,z)))
    P = Id(3) - OuterProduct(normal, normal)
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
    simulation.AddSolver(surf_adr_sol)

    simulation.run()

    err = surf_adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    assert err<0.02

def test_e_surface_adr_open_dirichlet():
    
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

    simulation = Simulation(mesh=mesh, dt=dt, t=t, T=T)

    u_ex = cos(pi*x)*sin(pi*y)*cos(t)
    d = 1 + t
    c = cos(y)
    b = CF((-z,0,x))
    normal = CF((x,y,z))/Norm(CF((x,y,z)))
    P = Id(3) - OuterProduct(normal, normal)
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*gradient(u_ex, P)).Compile()
    flux = flux1 + flux2
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, P)) + c*u_ex).Compile()
    fes_order = 1
    dirichlet_bc = {'bottom': u_ex}

    surf_adr_sol = SurfaceADRSolver(fes_order=fes_order,
                                    rhs = rhs,
                                    advection = b,
                                    diffusion = d,
                                    reaction = c,
                                    u0 = u_ex,
                                    dir_bc=dirichlet_bc)
    # Adding the solver to the simulation
    simulation.AddSolver(surf_adr_sol)

    simulation.run()

    err = surf_adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    assert err<0.011

def test_e_surface_adr_open_neumann():
    
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

    simulation = Simulation(mesh=mesh, dt=dt, t=t, T=T)

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
    simulation.AddSolver(surf_adr_sol)

    simulation.run()

    err = surf_adr_sol.compute_error(u_ex, 'L2')

    err = np.sqrt(np.sum(simulation.dt.Get()*np.array(err)**2))

    assert err<0.013