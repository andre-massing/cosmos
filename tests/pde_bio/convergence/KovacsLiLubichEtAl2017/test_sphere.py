from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import generate_boundary_sphere
from cosmos.utils.tools import gradient
import numpy as np
import logging
import pytest
import pandas as pd

from cosmos.core.model import CosmosModel
from cosmos.pde.willmore.geometrical_flow_model import GeometricalFlowModel
from cosmos.pde.adr.boundary.adr_boundary_system_bdf1_model_nostab import ADRBoundarySystemBDF1Model

logging.getLogger().setLevel(logging.INFO)

@pytest.mark.parametrize("redistribute", [False, True])
def test_sphere_bio(
        request,
        artifacts_path,
        redistribute):
    
    out = artifacts_path
    filename = request.function.__name__

    R0 = 1
    R1 = 2
    kappa = 0.5
    delta = 0.4
    Tend = 1

    def solve(mesh, dt):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))

        t = Parameter(0.0)
        dt = Parameter(dt)

        Rt = R0*R1/(R1*exp(-kappa*t) + R0*(1-exp(-kappa*t)))
        
        u0 = x*y
        u_ex = u0*exp(-6*t)
        v = Rt.Diff(t)
        n = Normalize(CF((x,y,z)))
        Vn = v*n
        P = Id(3) - OuterProduct(n, n)
        f = (u_ex.Diff(t) + Vn*gradient(u_ex, Id(3)) + u_ex*Trace(gradient(Vn, P)) \
            - Trace(gradient(gradient(u_ex, P), P)))
        g = (v + 2/Rt - delta*u_ex)

        model = CosmosModel(name = 'test_sphere', parentmesh=mesh, t0 = 0, t1 = Tend,
                            dt = dt, t = t, coupling_type = 'implicit', redistribute=redistribute)
        comp1 = model.create_compartment(name='compartment', boundary = 'default', bboundary = '')
        mc = model.create_pde(name='mean_curvature', pde_model=GeometricalFlowModel, compartment=comp1, ale_type = 0)
        adr = model.create_pde(name='adr', pde_model=ADRBoundarySystemBDF1Model, compartment=comp1, ale_type = 1, dim=1)
        ale = model.create_ale(name='ale', compartment=comp1)

        ale.set_normal_velocity(mc.V_h)
        ale.set_tangential_velocity(CF((0,0,0)))

        mc.set_params(
            alpha = 0,
            beta = 0, 
            gamma = 1,
            rhs = lambda: delta*adr.sol[0] + g
        )

        adr.set_params(
            u0_1 = u0,
            d_1 = 1,
            rhs_1 = f,
            b_1 = lambda: model.ale.V,
        )

        errs = []
        for _ in model():
            gfu.Set(sqrt(x**2+y**2+z**2) - Rt, dual = True, definedon = mesh.Boundaries('.*'))
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy())))

        return np.max(errs)

    power_t = 1.5
    dt0 = 0.05
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.1
    dh_refs = 3
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh = generate_boundary_sphere(maxh = dh, R = R0)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = err
    dhs = true_hs

    os.makedirs(out, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'mc_adr_kovacs_duanli_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'mc_adr_kovacs_duanli_ht.dat'), sep='\t', index=False)
    
    assert 1