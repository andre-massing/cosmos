from cosmos.solvers.willmore import StabWillmoreSolver
from cosmos.utils.generate_surface_meshes import *
from cosmos.solvers.simulation import MovingProblem

from ngsolve import *
import numpy as np

def test_e_stab_willmore_solver_sphere():

    mesh, _ = generate_sphere(maxh=0.2)

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

    fes_order = 1
    willmore_sol = StabWillmoreSolver(fes_order=fes_order, sp_curv = -2)

    simulation.attach_displ_solver(willmore_sol)
    def displacement():
        cf = willmore_sol.get_solution()[0] + simulation.data['dX_old']
        return cf

    simulation.set_displacement(displacement)

    simulation.run()

    dX_ex = CF((0, 0, 0))
    kappa_ex = -2*CF((x, y, z))/Norm(CF((x, y, z)))
    err = willmore_sol.compute_error([dX_ex, kappa_ex], 'L2')

    ERR = [np.sqrt(np.sum(simulation.dt.Get()*np.array(err[0])**2)),
           np.sqrt(np.sum(simulation.dt.Get()*np.array(err[1])**2))]

    assert ERR[0]<0.005 and ERR[1]<0

def test_e_stab_willmore_solver_half_sphere():

    mesh, _ = generate_half_sphere(maxh=0.2)

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

    fes_order = 1
    willmore_sol = StabWillmoreSolver(fes_order=fes_order, clamped_bc='bottom')

    simulation.attach_displ_solver(willmore_sol)
    def displacement():
        cf = willmore_sol.get_solution()[0] + simulation.data['dX_old']
        return cf

    simulation.set_displacement(displacement)

    simulation.run()

    dX_ex = CF((0, 0, 0))
    kappa_ex = -2*CF((x, y, z))/Norm(CF((x, y, z)))
    err = willmore_sol.compute_error([dX_ex, kappa_ex], 'L2')

    ERR = [np.sqrt(np.sum(simulation.dt.Get()*np.array(err[0])**2)),
           np.sqrt(np.sum(simulation.dt.Get()*np.array(err[1])**2))]

    assert ERR[0]<0.012 and ERR[1]<0

def test_e_stab_willmore_solver_torus():

    R = sqrt(2)
    r = 1

    mesh, _ = generate_torus(R=R, r = r, maxh=0.5, vol_or_bnd='BND')

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

    fes_order = 1
    willmore_sol = StabWillmoreSolver(fes_order=fes_order)

    simulation.attach_displ_solver(willmore_sol)
    def displacement():
        cf = willmore_sol.get_solution()[0] + simulation.data['dX_old']
        return cf

    simulation.set_displacement(displacement)

    simulation.run()

    dX_ex = CF((0,0,0))
    a = (sqrt(x**2+y**2)-R)
    b = z
    c = sqrt(a**2 + b**2)
    v = IfPos(b, acos(a/c), 2*pi - acos(a/c))
    u = IfPos(y, acos(x/sqrt(x**2+y**2)), 2*pi - acos(x/sqrt(x**2+y**2)))
    n = CF((cos(u)*cos(v), sin(u)*cos(v), sin(v)))
    H = CF(2*(R+2*r*cos(v))/(2*r*(R+r*cos(v))))
    kappa_ex = -n*H
    err = willmore_sol.compute_error([dX_ex, kappa_ex], 'L2')

    ERR = [np.sqrt(np.sum(simulation.dt.Get()*np.array(err[0])**2)),
           np.sqrt(np.sum(simulation.dt.Get()*np.array(err[1])**2))]

    assert ERR[0]<0.065 and ERR[1]<0

def test_e_stab_willmore_solver_half_torus():

    R = sqrt(2)
    r = 1

    mesh, _ = generate_half_torus(R=R, r = r, maxh=0.5, vol_or_bnd='BND')

    T = 1.0
    dt = Parameter(0.1)
    t = Parameter(0)

    simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

    fes_order = 1
    willmore_sol = StabWillmoreSolver(fes_order=fes_order, clamped_bc='bottom')

    simulation.attach_displ_solver(willmore_sol)
    def displacement():
        cf = willmore_sol.get_solution()[0] + simulation.data['dX_old']
        return cf

    simulation.set_displacement(displacement)

    simulation.run()

    dX_ex = CF((0,0,0))
    dX_ex = CF((0,0,0))
    a = (sqrt(x**2+y**2)-R)
    b = z
    c = sqrt(a**2 + b**2)
    v = IfPos(b, acos(a/c), 2*pi - acos(a/c))
    u = IfPos(y, acos(x/sqrt(x**2+y**2)), 2*pi - acos(x/sqrt(x**2+y**2)))
    n = CF((cos(u)*cos(v), sin(u)*cos(v), sin(v)))
    H = CF(2*(R+2*r*cos(v))/(2*r*(R+r*cos(v))))
    kappa_ex = -n*H
    err = willmore_sol.compute_error([dX_ex, kappa_ex], 'L2')

    ERR = [np.sqrt(np.sum(simulation.dt.Get()*np.array(err[0])**2)),
           np.sqrt(np.sum(simulation.dt.Get()*np.array(err[1])**2))]

    assert ERR[0]<0.04 and ERR[1]<0

# def test_e_willmore_solver_sphere():

#     mesh, _ = generate_sphere(maxh=0.2)

#     T = 1.0
#     dt = Parameter(0.1)
#     t = Parameter(0)

#     simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

#     fes_order = 1
#     params = {'sp_curv': -2}
#     willmore_sol = WillmoreSolver(fes_order=fes_order, params=params)

#     simulation.attach_displ_solver(willmore_sol)
#     def displacement():
#         cf = willmore_sol.get_solution()[0] + simulation.data['dX_old']
#         return cf

#     simulation.set_displacement(displacement)

#     simulation.run()

#     dX_ex = CF((0, 0, 0))
#     kappa_ex = -2*CF((x, y, z))/Norm(CF((x, y, z)))
#     err = willmore_sol.compute_error([dX_ex, kappa_ex], 'L2')

#     ERR = [np.sqrt(np.sum(simulation.dt.Get()*np.array(err[0])**2)),
#            np.sqrt(np.sum(simulation.dt.Get()*np.array(err[1])**2))]

#     assert ERR[0]<0.005 and ERR[1]<0

# def test_e_willmore_solver_half_sphere():

#     mesh, _ = generate_half_sphere(maxh=0.2)

#     T = 1.0
#     dt = Parameter(0.1)
#     t = Parameter(0)

#     simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

#     fes_order = 1
#     params = {}
#     willmore_sol = WillmoreSolver(fes_order=fes_order, params=params, clamped_bc='bottom')

#     simulation.attach_displ_solver(willmore_sol)
#     def displacement():
#         cf = willmore_sol.get_solution()[0] + simulation.data['dX_old']
#         return cf

#     simulation.set_displacement(displacement)

#     simulation.run()

#     dX_ex = CF((0, 0, 0))
#     kappa_ex = -2*CF((x, y, z))/Norm(CF((x, y, z)))
#     err = willmore_sol.compute_error([dX_ex, kappa_ex], 'L2')

#     ERR = [np.sqrt(np.sum(simulation.dt.Get()*np.array(err[0])**2)),
#            np.sqrt(np.sum(simulation.dt.Get()*np.array(err[1])**2))]

#     assert ERR[0]<0.012 and ERR[1]<0

# def test_e_willmore_solver_torus():

#     R = sqrt(2)
#     r = 1

#     mesh, _ = generate_torus(R=R, r = r, maxh=0.5, vol_or_bnd='BND')

#     T = 1.0
#     dt = Parameter(0.1)
#     t = Parameter(0)

#     simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

#     fes_order = 1
#     params = {}
#     willmore_sol = WillmoreSolver(fes_order=fes_order, params=params)

#     simulation.attach_displ_solver(willmore_sol)
#     def displacement():
#         cf = willmore_sol.get_solution()[0] + simulation.data['dX_old']
#         return cf

#     simulation.set_displacement(displacement)

#     simulation.run()

#     dX_ex = CF((0,0,0))
#     a = (sqrt(x**2+y**2)-R)
#     b = z
#     c = sqrt(a**2 + b**2)
#     v = IfPos(b, acos(a/c), 2*pi - acos(a/c))
#     u = IfPos(y, acos(x/sqrt(x**2+y**2)), 2*pi - acos(x/sqrt(x**2+y**2)))
#     n = CF((cos(u)*cos(v), sin(u)*cos(v), sin(v)))
#     H = CF(2*(R+2*r*cos(v))/(2*r*(R+r*cos(v))))
#     kappa_ex = -n*H
#     err = willmore_sol.compute_error([dX_ex, kappa_ex], 'L2')

#     ERR = [np.sqrt(np.sum(simulation.dt.Get()*np.array(err[0])**2)),
#            np.sqrt(np.sum(simulation.dt.Get()*np.array(err[1])**2))]

#     assert ERR[0]<0.065 and ERR[1]<0

# def test_e_willmore_solver_half_torus():

#     R = sqrt(2)
#     r = 1

#     mesh, _ = generate_half_torus(R=R, r = r, maxh=0.5, vol_or_bnd='BND')

#     T = 1.0
#     dt = Parameter(0.1)
#     t = Parameter(0)

#     simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

#     fes_order = 1
#     willmore_sol = WillmoreSolver(fes_order=fes_order, clamped_bc='bottom')

#     simulation.attach_displ_solver(willmore_sol)
#     def displacement():
#         cf = willmore_sol.get_solution()[0] + simulation.data['dX_old']
#         return cf

#     simulation.set_displacement(displacement)

#     simulation.run()

#     dX_ex = CF((0,0,0))
#     dX_ex = CF((0,0,0))
#     a = (sqrt(x**2+y**2)-R)
#     b = z
#     c = sqrt(a**2 + b**2)
#     v = IfPos(b, acos(a/c), 2*pi - acos(a/c))
#     u = IfPos(y, acos(x/sqrt(x**2+y**2)), 2*pi - acos(x/sqrt(x**2+y**2)))
#     n = CF((cos(u)*cos(v), sin(u)*cos(v), sin(v)))
#     H = CF(2*(R+2*r*cos(v))/(2*r*(R+r*cos(v))))
#     kappa_ex = -n*H
#     err = willmore_sol.compute_error([dX_ex, kappa_ex], 'L2')

#     ERR = [np.sqrt(np.sum(simulation.dt.Get()*np.array(err[0])**2)),
#            np.sqrt(np.sum(simulation.dt.Get()*np.array(err[1])**2))]

#     assert ERR[0]<0.04 and ERR[1]<0
