from ngsolve import *
from cosmos.pdes.pde_tools import gradient
from cosmos.utils.generate_surface_meshes import generate_sphere
from cosmos.pdes.pde_tools import SaveError
from cosmos.pdes.pde_adr_bnd import BndADR
import os
import numpy as np
import pandas as pd
from cosmos.solvers.solvers import Dynamic
from ngsolve.webgui import Draw
from myngspy import *
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.coupling_weak import WeakCoupling

testname = 'convergence_adr_mc_scalar'

def test_adr_mc_scalar(results_path, solver_name, pde_constructor, **init_kwargs):

    R0 = 1
    R1 = 2
    kappa = 0.5
    delta = 0.4

    def sphere_mean_c(mesh, dt, folderpath, filename):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))

        t = Parameter(0.0)
        dt = Parameter(dt)
        T = 1

        Rt = R0*R1/(R1*exp(-kappa*t) + R0*(1-exp(-kappa*t)))
        u_ex = x*y*exp(-6*t)
        v = Rt.Diff(t)
        n = Normalize(CF((x,y,z)))
        Vn = v*n
        P = Id(3) - OuterProduct(n, n)
        f = (u_ex.Diff(t) + Vn*gradient(u_ex, Id(3)) + u_ex*Trace(gradient(Vn, P)) \
            - Trace(gradient(gradient(u_ex, P), P))).Compile(True, True)
        g = ((v + 2/Rt - delta*u_ex)).Compile(True, True)

        mean_c = pde_constructor(name = [filename + 'dX', filename + 'mc'],
                        **init_kwargs)
        u = BndADR(d = 1, rhs = f, time_scheme=BDF1(), u0=u_ex)

        solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)

        cpl = WeakCoupling(tol = 1e-5, type = 'implicit')
        cpl.AddPDEs(mean_c, u)
        solver.AddPDE(cpl)

        solver.ale.deformation_field = mean_c.displacement
        solver.ale.velocity_field = lambda : mean_c.displacement/solver.dt
        solver.ale.mat_velocity_field = lambda : mean_c.displacement/solver.dt

        def mc_rhs():
            term1 = delta*u.solute + g
            return term1
        mean_c.rhs.value = mc_rhs

        errs = []
        scene = Draw(solver.ale.deformation, mesh, deformation = solver.ale.deformation)
        for _ in solver():

            mesh.SetDeformation(solver.ale.deformation)
            gfu.Set(sqrt(x**2+y**2+z**2)-Rt, dual = True, definedon = mesh.Boundaries('.*'))
            mesh.UnsetDeformation()
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy())))
            scene.Redraw()

        return errs


    folderpath = os.path.join(results_path, solver_name, testname)
    os.makedirs(folderpath, exist_ok=True)

    power_t = 1.5
    dt0 = 0.01
    dt_refs = 2
    dts = dt0/(power_t**(np.arange(dt_refs+1)))
    print('Convergence time-steps:', dts)

    power_h = 1.5
    dh0 = 0.2
    dh_refs = 2
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))
    print('Convergence mesh-sizes:', dhs)

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            name = 'error' + str(i) + str(j)
            mesh, _ = generate_sphere(maxh = dh, R = R0)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(MyMeshSize(), definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            errs = sphere_mean_c(mesh=mesh, dt=dt, folderpath=folderpath, filename=name)

            ERRORS_dX[i, j] = np.max(errs)
    dhs = true_hs

    print('Sneak look at overall convergence displacement')
    print(np.log(ERRORS_dX.diagonal()[:-1]/ERRORS_dX.diagonal()[1:])/np.log(np.maximum(power_h, power_t)))


    name = os.path.join(folderpath, 'data_dX.dat')
    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(name, sep='\t', index=False)

    name = os.path.join(folderpath, 'data_dX_flipped.dat')
    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(name, sep='\t', index=False)
