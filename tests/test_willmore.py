from cosmos.solvers.willmore_solver import WillmoreSolver
from cosmos.utils.generate_surface_meshes import *
from cosmos.utils.manufactured_solution_tools import ErrorTools

from ngsolve import *
import numpy as np

def test_willmore_circle():

    displ_ex = CF((0,0))
    mc_exact = -CF((x,y))

    fes_order = 1
    T = 1.0
    dt0 = 0.05
    dt = Parameter(dt0)
    dh0 = 0.2
    mesh, geo = generate_circle(maxh = dh0, order_g=1, R=1)
    rhs = CF(0.0)

    V = VectorH1(mesh, order = 1)

    t = Parameter(0)
    sol_params = {'rhs': rhs,
                'domain_name': '.*'}
    solver = WillmoreSolver(mesh, fes_order=fes_order, dt=dt, t=t, T=T, params = sol_params, verbose = 0)

    err_params = {'exact_solutions': [displ_ex, mc_exact],
            'vol_or_bnd': [BND, BND],
            'norms': ['L2L2', 'L2L2']}
    errors = ErrorTools(solver, params=err_params)

    h_power = 1.5
    t_power = 1.5
    n_h_refs = 4
    n_t_refs = 4

    conv_params = {
            'geometry': geo,
            'space_params': [dh0, h_power, n_h_refs],
            'time_params': [dt0, t_power, n_t_refs]
        }

    ERRS = errors.compute_eoc(conv_params)

    for i in range(2):

        ERRS[i] = np.diag(ERRS[i][:,:])

    eoc = []
    for i in range(2):
        eoc.append(np.mean(np.log(ERRS[i][:-1]/ERRS[i][1:])/np.log(np.max([h_power, t_power]))))

    assert eoc[0]<4.2 and eoc[1]<4.1

def test_willmore_half_torus():

    t = Parameter(0.0)

    r = 1
    R = sqrt(2)

    u_ex = CF((0,0,0))
    n = CF((x,y,z))/r-R*CF((x,0,z))/r/sqrt(x**2+z**2)
    H = (R-3*sqrt(x**2+z**2))/(2*r*sqrt(x**2+z**2))
    mc_exact = -n*H
    rhs = CF((0,0,0))

    fes_order = 1

    T = 1.0
    dt = Parameter(0.05)
    
    mesh, _ = generate_half_torus(maxh = 0.2, order_g=1, R=R, r=r, vol_or_bnd='BND')

    params = {'rhs': rhs,
              'domain_name': 'membrane',
              'clamped_bnd': 'bottom'}
    
    solver = WillmoreSolver(mesh, fes_order=fes_order, dt=dt, t=t, T=T, params = params)

    params = {'exact_solutions': [u_ex, mc_exact],
              'vol_or_bnd': [BND, BND],
              'norms': ['LinfL2', 'LinfL2']}
    errors = ErrorTools(solver, params=params)

    ERR = errors.compute_errors()

    assert ERR[0]<0.19

def test_willmore_half_sphere():

    t = Parameter(0.0)

    u_ex = CF((0,0,0))
    rhs = CF((0,0,0))

    fes_order = 1

    T = 1.0
    dt = Parameter(0.05)
    
    mesh, _ = generate_half_sphere(maxh = 0.2, order_g=1, R=sqrt(2))

    params = {'rhs': rhs,
              'domain_name': '.*',
              'clamped_bnd': 'bottom'}
    
    solver = WillmoreSolver(mesh, fes_order=fes_order, dt=dt, t=t, T=T, params = params)

    params = {'exact_solutions': [u_ex, CF((0,0,0))],
              'vol_or_bnd': [BND, BND],
              'norms': ['LinfL2', 'LinfL2']}
    errors = ErrorTools(solver, params=params)

    ERR = errors.compute_errors()

    assert ERR[0]<0.22

def test_willmore_torus():

    t = Parameter(0.0)

    u_ex = CF((0,0,0))
    rhs = CF((0,0,0))

    fes_order = 1

    T = 1.0
    dt = Parameter(0.05)
    
    mesh, _ = generate_torus(maxh = 0.2, order_g=1, R=sqrt(2), r=1, vol_or_bnd='BND')

    params = {'rhs': rhs,
              'domain_name': '.*'}
    
    solver = WillmoreSolver(mesh, fes_order=fes_order, dt=dt, t=t, T=T, params = params)

    params = {'exact_solutions': [u_ex, CF((0, 0, 0))],
              'vol_or_bnd': [BND, BND],
              'norms': ['LinfL2', 'LinfL2']}
    errors = ErrorTools(solver, params=params)

    ERR = errors.compute_errors()

    assert ERR[0]<0.262

