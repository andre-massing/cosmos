from ngsolve import *
from cosmos.pdes.pde_tools import gradient
from cosmos.utils.generate_surface_meshes import generate_half_sphere
import os
import numpy as np
import pandas as pd
from cosmos.solvers.solvers import Dynamic
from ngsolve.webgui import Draw
from myngspy import *

testname = 'convergence_half_sphere_neu'

def test_half_sphere_neu(results_path, solver_name, pde_constructor, **init_kwargs):

    def adr_b_u(mesh, dt, folderpath, filename):

        t = Parameter(0.0)
        dt = Parameter(dt)
        T = 1

        def A_B(t):
            A = CF((cos(t), -sin(t), 0,\
                        sin(t), cos(t), 0,\
                            0, 0, 1), dims = (3,3))
            B = CF((0, 0, 0))
            return A, B
        A, B = A_B(t)
        detJ = Det(A)
        invA = Cof(A).trans/detJ
        inv_phi = invA*(CF((x,y,z)) - B) # Inverse of deformation map
        w_phi = A.Diff(t)*inv_phi + B.Diff(t) # velocity of the moving domain

        A1, B1 = A_B(t+dt)
        # dX_ex = A1*CF((x,y,z))+B1 - CF((x,y,z))
        dX_ex = (A1-A)*inv_phi+(B1-B)

        n_ex = Cof(A)*inv_phi/Norm(Cof(A)*inv_phi)
        P_ex = Id(3) - OuterProduct(n_ex, n_ex)

        u_ex = cos(2*pi*x)*cos(t) # Exact solution
        c = 1 + t**2 # reaction coefficient
        b = P_ex*CF((2,-x, 1))
        flux_b = b*u_ex
        rel_flux = u_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_ex, Id(3))
        rhs = u_ex.Diff(t) + rel_flux + Trace(gradient(flux_b, P_ex)) + c*u_ex # manufactured solution right-hand side

        neu_b = {'bottom': flux_b}
        pde = pde_constructor(c = c, b = b, neu_b = neu_b, 
                        rhs = rhs, u0=u_ex, name = filename,
                        **init_kwargs)

        from cosmos.pdes.pde_tools import SaveError
        err_save = SaveError(ex_sol = u_ex, 
                        norm = 'L2', 
                        folderpath = folderpath,
                        filename =filename)
        pde.SaveErr(err_save)

        solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
        solver.AddPDE(pde)
        solver.ale.deformation_field = dX_ex

        scene = Draw(pde.solute, mesh, deformation = solver.ale.deformation)
        for sol in solver():
            scene.Redraw()

    folderpath = os.path.join(results_path, solver_name, testname)

    power_t = 1.5
    dt0 = 0.02
    dt_refs = 2
    dts = dt0/(power_t**(np.arange(dt_refs+1)))
    print('Convergence time-steps:', dts)

    power_h = 1.5
    dh0 = 0.1
    dh_refs = 2
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))
    print('Convergence mesh-sizes:', dhs)

    ERRORS = np.zeros((len(dts), len(dhs)))
    ERRORS.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            name = 'error' + str(i) + str(j)
            mesh, _ = generate_half_sphere(maxh = dh)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(MyMeshSize(), definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            adr_b_u(mesh=mesh, dt=dt, folderpath=folderpath, filename=name)

            file_path = os.path.join(folderpath, name)
            df = pd.read_csv(file_path)
            errs = df[name]

            ERRORS[i, j] = np.sqrt(np.sum(dt*errs**2))
    dhs = true_hs

    dict = {
        'dhs': dhs,
        'power_h': power_h,
        'dts': dts,
        'power_t': power_t,
        'errors': ERRORS
    }
    print('Sneak look at overall convergence')
    print(np.log(ERRORS.diagonal()[:-1]/ERRORS.diagonal()[1:])/np.log(np.maximum(power_h, power_t)))

    name = os.path.join(folderpath, 'data.dat')
    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(name, sep='\t', index=False)

    name = os.path.join(folderpath, 'data_flipped.dat')
    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(name, sep='\t', index=False)