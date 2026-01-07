from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import generate_volume_circle, generate_volume_ball
from cosmos.core.model import CosmosModel
from cosmos.pde.adr.volume.adr_volume_system_bdf1_model import ADRVolumeSystemBDF1Model
from cosmos.pde.adr.boundary.adr_boundary_system_bdf1_model import ADRBoundarySystemBDF1Model
from cosmos.config.parameters import get_config
from cosmos.utils.tools import gradient
import numpy as np
import logging
import pytest
import pandas as pd

logging.getLogger().setLevel(logging.INFO)

def test_convergence_sb_circle(
        request,
        artifacts_path):
    
    out = artifacts_path
    filename = request.function.__name__

    def solve(mesh, dt):

        t = Parameter(0.0)
        T = 1

        model_name = f'model_{mesh.nv}_{dt:.2e}'
        model = CosmosModel(name = model_name, parentmesh = mesh, dt = dt, t0 = 0, t1 = T, t = t,
                            root = out, sample_rate = 1, coupling_type = 'implicit')
        comp_vol = model.create_compartment(name = 'comp_vol', material = 'default', boundary = 'boundary')
        comp_bnd = model.create_compartment(name = 'comp_bnd', boundary = 'boundary', bboundary = '')
        ale = model.create_ale(name = 'ale', compartment=comp_vol)

        u_ex = sin(t)*cos(x*y) # Exact solution
        v_ex = sin(t)*sin(x*y) # Exact solution

        ## Coupling
        beta = 1
        alpha = 2

        def A_B(t):
            A = CF((cos(t), -sin(t),\
                        sin(t), cos(t)), dims = (2,2))
            B = CF((0, 0))
            return A, B
        A, B = A_B(t)
        detJ = Det(A)
        invA = Cof(A).trans/detJ
        inv_phi = invA*(CF((x,y)) - B) # Inverse of deformation map
        n_ex = Cof(A)*inv_phi/Norm(Cof(A)*inv_phi)
        P_ex = Id(2) - OuterProduct(n_ex, n_ex)
        w_phi = A.Diff(t)*inv_phi + B.Diff(t) # velocity of the moving domain

        d_vol = CF(1)
        flux_vol = -d_vol*gradient(u_ex, Id(2))
        rhs_vol = (u_ex.Diff(t) + Trace(gradient(w_phi*u_ex, Id(2))) + Trace(gradient(flux_vol, Id(2)))) # manufactured solution right-hand side
        d_bnd = CF(1)
        flux_bnd = -d_bnd*gradient(v_ex, P_ex)
        rhs_bnd = v_ex.Diff(t) + v_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(v_ex, Id(2)) +\
            Trace(gradient(flux_bnd, P_ex)) + beta*v_ex - alpha*u_ex  # manufactured solution right-hand side

        Ap, Bp = A_B(t+dt)
        displacement_ex = (Ap-A)*inv_phi+(Bp-B)
        ale.set_domain_velocity(displacement_ex/dt)
        # ale.set_domain_velocity(w_phi)

        pde_vol = model.create_pde(name = 'adr_vol', pde_model=ADRVolumeSystemBDF1Model, compartment=comp_vol, dim = 1)
        ns = specialcf.normal(2)
        def tot_flux_bnd_cf():
            return  -1*flux_vol*ns + alpha*(pde_vol.sol[0] - u_ex) - beta*(pde_bnd.sol[0] - v_ex)
        pde_vol.set_params(d_1 = d_vol, rhs_1 = rhs_vol, u0_1 = u_ex,
                        tot_flux_bnd_1 = tot_flux_bnd_cf)
        pde_bnd = model.create_pde(name = 'adr_bnd', pde_model=ADRBoundarySystemBDF1Model, compartment=comp_bnd, dim = 1)
        def rhs_bnd_cf():
            return rhs_bnd - beta*pde_bnd.sol[0] + alpha*pde_vol.sol[0]
        pde_bnd.set_params(d_1 = d_bnd, rhs_1 = rhs_bnd_cf, u0_1 = v_ex)

        def error_vol(): return sqrt(Integrate(InnerProduct(pde_vol.sol[0] - u_ex, pde_vol.sol[0] - u_ex), mesh, VOL))
        def error_bnd(): return sqrt(Integrate(InnerProduct(pde_bnd.sol[0] - v_ex, pde_bnd.sol[0] - v_ex), mesh, BND))
        output_callables = {'error_vol': error_vol,
                            'error_bnd': error_bnd}
        model.set_params(output_callables = output_callables)
        errs_vol = []
        errs_bnd = []
        for _ in model():
            errs_vol.append(error_vol())
            errs_bnd.append(error_bnd())

        return np.sqrt(dt*np.sum(np.array(errs_vol)**2)), np.sqrt(dt*np.sum(np.array(errs_bnd)**2))

    power_t = 1.5
    dt0 = 0.001
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.4
    dh_refs = 3
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_vol = np.zeros((len(dts), len(dhs)))
    ERRORS_vol.fill(np.inf)
    ERRORS_bnd = np.zeros((len(dts), len(dhs)))
    ERRORS_bnd.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh = generate_volume_circle(maxh = dh, R = 1)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err_vol, err_bnd = solve(mesh=mesh, dt=dt)

            ERRORS_vol[i, j] = err_vol
            ERRORS_bnd[i, j] = err_bnd
    dhs = true_hs

    os.makedirs(out, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_vol))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'adr_convergence_vol_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_vol.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'adr_convergence_vol_ht.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_bnd))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'adr_convergence_bnd_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_bnd.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'adr_convergence_bnd_ht.dat'), sep='\t', index=False)
    
    assert 1