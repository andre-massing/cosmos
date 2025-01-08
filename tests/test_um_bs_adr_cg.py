from cosmos.solvers.um_bs_adr_cg_solver import um_sb_ADR_CGSolver
from cosmos.utils.generate_surface_meshes import *
from cosmos.utils.manufactured_solution_tools import ErrorTools, gradient, get_lin_trans_params

from ngsolve import *

def test_um_bs_adr_cg_3D():

    t = Parameter(0.0)

    # General affine transformation (sphere to ellipse)
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t), 0,\
                sin(t), (1-0.25*sin(t))*cos(t), 0,\
                    0, 0, 1), dims = (3,3))
    b = CF((0.2*t, 0.1*t, 0))

    phi, inv_phi, w_phi, detJ, n_ex = get_lin_trans_params(A, b, t)
    P_ex = Id(3)-OuterProduct(n_ex, n_ex)

    displ_ex = phi - CF((x,y,z))
    
    d_v = 1 + x**2
    c_v = cos(x)
    b_v = CF((2,1,0))

    d_s = 1 + y**2
    c_s = cos(y)
    b_s = P_ex*CF((0,0,0))

    alpha = 1.0
    beta = 1.0

    # I want a stationary solution on the moving mesh
    u_ex = cos(pi*x)*sin(3*y)*cos(t)
    v_ex = ((d_v*gradient(u_ex, Id(3)) - b_v*u_ex)*n_ex + alpha*u_ex)/beta

    flux1_v = (b_v*u_ex).Compile()
    flux2_v = (- d_v*gradient(u_ex, Id(3))).Compile()
    rel_flux_v = (w_phi*u_ex).Compile()
    flux_v = flux1_v + flux2_v + rel_flux_v
    rhs_v = (u_ex.Diff(t) + Trace(gradient(flux_v, Id(3))) + c_v*u_ex).Compile()

    bnd_cond={'convective_flux': [['membrane', flux2_v]],
              'diffusive_flux': [['membrane', flux1_v]]}

    flux1_s = b_s*v_ex
    flux2_s = - d_s*gradient(v_ex, P_ex)
    rel_flux_s = v_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(v_ex, Id(3))
    flux_s = flux1_s + flux2_s
    rhs_s = (v_ex.Diff(t) + rel_flux_s + Trace(gradient(flux_s, P_ex)) + c_s*v_ex + beta*v_ex - alpha*u_ex).Compile()

    fes_order = 1
    R0 = 1.0
    T = 1.0
    dt = Parameter(0.05)
    mesh, _ = generate_ball(maxh=0.1, R = R0, order_g =1)

    params = {'convection_v': b_v,
              'convection_s': b_s,
              'k_diffusion_v': d_v,
              'k_diffusion_s': d_s,
              'k_reaction_v': c_v,
              'k_reaction_s': c_s,
              'initial_c_v': u_ex,
              'initial_c_s': v_ex,
              'rhs_v': rhs_v,
              'rhs_s': rhs_s,
              'boundary_c': bnd_cond,
              'coupling_coefs': [alpha, beta],
              'displacement': displ_ex}
    
    solver = um_sb_ADR_CGSolver(mesh, fes_order=fes_order, dt=dt, t=t, T=T, params = params)

    params = {'exact_solutions': [u_ex, v_ex],
              'vol_or_bnd': [VOL, BND],
              'norms': ['L2L2', 'LinfL2']}
    errors = ErrorTools(solver, params=params)

    ERR_u, ERR_v = errors.compute_errors()

    assert (ERR_u<3.6e-2 and ERR_v<0.15)

