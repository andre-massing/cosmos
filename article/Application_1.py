"""Application 1 simulations.
"""

import numpy as np
from ngsolve import *
from cosmos import *
from cosmos.utils.generate_meshes import generate_boundary_1D_circle
from cosmos.core.model import CosmosModel
from cosmos.pde import (
    ADRBoundarySystemBDF1Model,
    ADRBoundarySystemBDF1StabModel
)

SOLVER = ADRBoundarySystemBDF1StabModel
ROOT = '.'

a12s = np.round(np.linspace(0, 3, 21, endpoint=True), 2)[1:]
a21s = np.round(np.linspace(0, 3, 21, endpoint=True), 2)[1:]

for a12 in a12s:
    for a21 in a21s:
        dt = 5e-2
        t0 = 0
        t1 = 100

        kappa1 = 0.4
        kappa2 = 0.4
        m = 4
        s = 8
        r1 = 1
        r2 = 1
        klim_A = 1
        klim_B = 1
        A0 = 2.2
        B0 = 2.2

        R = 10
        mesh = generate_boundary_1D_circle(r=R, N=100)
        ns = specialcf.normal(2)

        model_name = f"Application_1_a12_{a12}_a21_{a21}"
        model = CosmosModel(
            name=model_name,
            parentmesh=mesh,
            dt=dt,
            t0=t0,
            t1=t1,
            root=ROOT,
            samples=100,
            coupling_type="implicit",
            redistribute=True,
            adaptive_timestep=True,
        )

        comp1 = model.create_compartment(name="compartment", boundary="default", bboundary="")

        adr_system = model.create_pde(
            name="adr_system",
            pde_model=SOLVER,
            compartment=comp1,
            ale_type=1,
            dim=2,
        )

        ale = model.create_ale(name="ale", compartment=comp1)

        def V():
            term1 = (adr_system.sol[0]) ** m / ((adr_system.sol[0]) ** m + A0**m)
            term2 = (adr_system.sol[1]) ** m / ((adr_system.sol[1]) ** m + B0**m)
            return term1 - term2

        ale.set_normal_velocity(V)
        ale.set_tangential_velocity(CF((0, 0)))

        adr_system.add_nonlinearity(
            target=1,
            expression="klim_A*u1**2 + a12*u1*u2 - kappa1*V*ns*u1",
            map={
                "kappa1": kappa1,
                "klim_A": klim_A,
                "a12": a12,
                "V": model.ale.V,
                "ns": specialcf.normal(2),
            },
        )
        adr_system.add_nonlinearity(
            target=2,
            expression="klim_B*u2**2 + a21*u1*u2 + kappa2*V*ns*u2",
            map={
                "kappa2": kappa2,
                "klim_B": klim_B,
                "a21": a21,
                "V": model.ale.V,
                "ns": specialcf.normal(2),
            },
        )

        adr_system.set_params(
            u0_1=0.1,
            d_1=0.1,
            c_1=-r1,
            b_1=lambda: model.ale.V,
            bounds_1=[0, 1e100],
            u0_2=1,
            d_2=0.1,
            c_2=-r2,
            b_2=lambda: model.ale.V,
            bounds_2=[0, 1e100],
            printing=True,
        )

        output_callables = {
            "mass_A": lambda: Integrate(adr_system.sol[0], mesh, VOL_or_BND=BND),
            "mass_B": lambda: Integrate(adr_system.sol[1], mesh, VOL_or_BND=BND),
            "A_point": lambda: adr_system.sol[0].vec[0],
            "B_point": lambda: adr_system.sol[1].vec[0],
            "normal_velocity": lambda: ale.gfu_norm_vel.vec[0],
            "x_bary": lambda: (
                Integrate(x, mesh, VOL_or_BND=BND) / Integrate(1, mesh, VOL_or_BND=BND)
            ),
            "y_bary": lambda: (
                Integrate(y, mesh, VOL_or_BND=BND) / Integrate(1, mesh, VOL_or_BND=BND)
            ),
            "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
            "volume": lambda: Integrate(CF((x, 0)) * ns, mesh, VOL_or_BND=BND),
            "min_A": lambda: np.min(adr_system.sol[0].vec.FV().NumPy()),
            "max_A": lambda: np.max(adr_system.sol[0].vec.FV().NumPy()),
            "min_B": lambda: np.min(adr_system.sol[1].vec.FV().NumPy()),
            "max_B": lambda: np.max(adr_system.sol[1].vec.FV().NumPy()),
        }

        model.set_params(output_callables=output_callables)

        depleted = False
        for _ in model():
            if model.t.Get() >= 10 * dt and not depleted:
                adr_mod = GridFunction(H1(mesh))
                adr_mod.Set(adr_system.sol[1], definedon=comp1.domain)
                adr_system.sol[1].Set(
                    (1 - 1 / (1 + exp(-5 * x))) * adr_mod,
                    dual=True,
                    definedon=comp1.domain,
                )
                arr = adr_system.sol[1].vec.FV().NumPy()
                adr_system.sol[1].vec.data[:] = np.clip(arr, a_min=0, a_max=1e100)
                depleted = True


for a12 in a12s:
    for a21 in a21s:
        dt = 5e-2
        t0 = 0
        t1 = 100

        kappa1 = 0.4
        kappa2 = 0.4
        m = 4
        s = 8
        r1 = 1
        r2 = 1
        klim_A = 1
        klim_B = 1
        A0 = 2.2
        B0 = 2.2

        R = 10
        mesh = generate_boundary_1D_circle(r=R, N=100)
        ns = specialcf.normal(2)
        Area0 = Integrate(CF((x, 0)) * ns, mesh, VOL_or_BND=BND)

        from cosmos.core.model import CosmosModel

        model_name = f"Application_1_feedback_a12_{a12}_a21_{a21}"
        model = CosmosModel(
            name=model_name,
            parentmesh=mesh,
            dt=dt,
            t0=t0,
            t1=t1,
            root=ROOT,
            samples=100,
            coupling_type="implicit",
            redistribute=True,
            adaptive_timestep=True,
        )

        comp1 = model.create_compartment(name="compartment", boundary="default", bboundary="")

        adr_system = model.create_pde(
            name="adr_system",
            pde_model=ADRBoundarySystemBDF1Model,
            compartment=comp1,
            ale_type=1,
            dim=2,
        )

        ale = model.create_ale(name="ale", compartment=comp1)

        def Tension():
            Area = Integrate(CF((x, 0)) * ns, mesh, VOL_or_BND=BND)
            T = (Area ** (s + 1)) / (Area**s + Area0**s)
            return T

        def V():
            T = Tension()
            term1 = (adr_system.sol[0]) ** m / ((adr_system.sol[0]) ** m + (A0 * T) ** m)
            term2 = (adr_system.sol[1]) ** m / ((adr_system.sol[1]) ** m + B0**m)
            return term1 - term2

        ale.set_normal_velocity(V)
        ale.set_tangential_velocity(CF((0, 0)))

        adr_system.add_nonlinearity(
            target=1,
            expression="klim_A*u1**2 + a12*u1*u2 - kappa1*V*ns*u1",
            map={
                "kappa1": kappa1,
                "klim_A": klim_A,
                "a12": a12,
                "V": model.ale.V,
                "ns": specialcf.normal(2),
            },
        )
        adr_system.add_nonlinearity(
            target=2,
            expression="klim_B*u2**2 + a21*u1*u2 + kappa2*V*ns*u2",
            map={
                "kappa2": kappa2,
                "klim_B": klim_B,
                "a21": a21,
                "V": model.ale.V,
                "ns": specialcf.normal(2),
            },
        )

        adr_system.set_params(
            u0_1=0.1,
            d_1=0.1,
            c_1=-r1,
            b_1=lambda: model.ale.V,
            bounds_1=[0, 1e100],
            u0_2=1,
            d_2=0.1,
            c_2=-r2,
            b_2=lambda: model.ale.V,
            bounds_2=[0, 1e100],
            printing=True,
        )

        output_callables = {
            "mass_A": lambda: Integrate(adr_system.sol[0], mesh, VOL_or_BND=BND),
            "mass_B": lambda: Integrate(adr_system.sol[1], mesh, VOL_or_BND=BND),
            "tension": lambda: Tension(),
            "A_point": lambda: adr_system.sol[0].vec[0],
            "B_point": lambda: adr_system.sol[1].vec[0],
            "normal_velocity": lambda: ale.gfu_norm_vel.vec[0],
            "x_bary": lambda: (
                Integrate(x, mesh, VOL_or_BND=BND) / Integrate(1, mesh, VOL_or_BND=BND)
            ),
            "y_bary": lambda: (
                Integrate(y, mesh, VOL_or_BND=BND) / Integrate(1, mesh, VOL_or_BND=BND)
            ),
            "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
            "volume": lambda: Integrate(CF((x, 0)) * ns, mesh, VOL_or_BND=BND),
            "min_A": lambda: np.min(adr_system.sol[0].vec.FV().NumPy()),
            "max_A": lambda: np.max(adr_system.sol[0].vec.FV().NumPy()),
            "min_B": lambda: np.min(adr_system.sol[1].vec.FV().NumPy()),
            "max_B": lambda: np.max(adr_system.sol[1].vec.FV().NumPy()),
        }

        model.set_params(output_callables=output_callables)

        depleted = False
        for _ in model():
            if model.t.Get() >= 10 * dt and not depleted:
                adr_mod = GridFunction(H1(mesh))
                adr_mod.Set(adr_system.sol[1], definedon=comp1.domain)
                adr_system.sol[1].Set(
                    (1 - 1 / (1 + exp(-5 * x))) * adr_mod,
                    dual=True,
                    definedon=comp1.domain,
                )
                arr = adr_system.sol[1].vec.FV().NumPy()
                adr_system.sol[1].vec.data[:] = np.clip(arr, a_min=0, a_max=1e100)
                depleted = True
