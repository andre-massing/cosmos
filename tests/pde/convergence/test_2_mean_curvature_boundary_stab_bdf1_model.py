# %%
from ngsolve import *
from cosmos import *
from cosmos.utils.generate_surface_meshes import generate_sphere
from cosmos.pdes.pde_tools import gradient
import numpy as np
import pandas as pd
from myngspy import *

def test_2_mean_curvature_boundary_stab_bdf1_model():

    R0 = 1
    R1 = 2
    kappa = 0.5
    delta = 0.4
    Tend = 1

    def sphere_sp_curv(mesh, dt):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))

        t = Parameter(0.0)
        dt = Parameter(dt)

        Rt = R0*R1/(R1*exp(-kappa*t) + R0*(1-exp(-kappa*t)))
        u0 =x*y
        u_ex = u0*exp(-6*t)
        v = Rt.Diff(t)
        n = Normalize(CF((x,y,z)))
        Vn = v*n
        P = Id(3) - OuterProduct(n, n)
        f = (u_ex.Diff(t) + Vn*gradient(u_ex, Id(3)) + u_ex*Trace(gradient(Vn, P)) \
            - Trace(gradient(gradient(u_ex, P), P))).Compile(True, True)
        g = ((v + 2/Rt - delta*u_ex)*n).Compile(True, True)

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend, t_coef=t)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime, iter = True)

        mean_curvature = MeanCurvatureBoundaryStabBDF1Model(solver, 1, name = 'mean_curvature_boundary_stab_bdf1')
        displacement = DisplacementBoundaryBDF1Coupling(solver, 2, mean_curvature.displacement)
        adr_input_params = {
            "u0": u0
        }
        adr = ADRBoundaryBDF1Model(solver, 3, name = 'adr_boundary_bdf1', input_params=adr_input_params)

        def mc_rhs():
            n = specialcf.normal(3)
            term1 = delta*adr.sol*n + g
            return term1
        mean_curvature.set_input_fields({
            "rhs": mc_rhs, 
        })
        adr.set_input_fields({
            "d": 1,
            "rhs": f
        })

        errs = []
        for _ in solver():

            gfu.Set(sqrt(x**2+y**2+z**2) - Rt, dual = True, definedon = mesh.Boundaries('.*'))
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy())))

        return errs

    power_t = 1.5
    dt0 = 0.01
    dt_refs = 2
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.2
    dh_refs = 2
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            name = 'error' + str(i) + str(j)
            mesh, _ = generate_sphere(maxh = dh, R = R0)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            errs = sphere_sp_curv(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = np.max(errs)
    dhs = true_hs

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv('./results_test_2_mean_curvature_boundary_stab_bdf1_model/data_dX.dat', sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv('./results_test_2_mean_curvature_boundary_stab_bdf1_model/data_dX_flipped.dat', sep='\t', index=False)

test_2_mean_curvature_boundary_stab_bdf1_model()