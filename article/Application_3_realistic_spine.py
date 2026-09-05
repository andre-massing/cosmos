"""Application 3 realistic spine simulations.
"""

from ngsolve import *
from cosmos import *
from dendritic_spine_geom import generate_synapse3d
from cosmos.core.model import CosmosModel
from cosmos.pde import (
    ADRVolumeSystemBDF1Model,
    DistanceVolumeModel,
    GeometricalFlowStationaryModel
)

INITIAL_TIMESTEP = 0.01 # INITIAL_TIMESTEP = 0.001
T1 = 70 # T1 = -59.75 or T1 = -59.999
ROOT = '.'

mesh = Mesh("./vol/spine_sliced_intermed/closed/filled/spine_intermed_cut_fixed.vol")
# mesh = Mesh("./vol/spine_sliced_fine/closed/filled/spine_refined_cut_fixed.vol")

A0 = 20
B0 = 3000 * 3.6
C0 = 40
K_A = 0.0013
K_B = 0.0081
K_C = 0.0006
I_A = 0.0255
I_B = 24.4284
I_C = 0.0237
I_SA = 0.0293
I_SB = 25.6684
I_SC = 0.4384
K_nuc = 0.0153
K_sev = 0.012
K_n = 0.6
psi0 = 3.6
psi1 = 0.02
N = 3

dt = Parameter(INITIAL_TIMESTEP)
t = Parameter(-60)

from cosmos.core.model import CosmosModel

model_name = (
    f"Application_3_NE_{mesh.ne}_DT_{INITIAL_TIMESTEP}_T1_{T1}"
)
model = CosmosModel(
    parentmesh=mesh,
    dt=dt,
    t=t,
    t0=t.Get(),
    t1=T1,
    root=ROOT,
    samples=400,
    name=model_name,
    coupling_type="implicit",
    redistribute=True,
    adaptive_timestep=True,
)

model.print_model_data()

comp1 = model.create_compartment("bulk", material="cd0_1", boundary="boundary2|default")
dist_fct = model.create_pde(
    "distance_function",
    pde_model=DistanceVolumeModel,
    compartment=comp1,
    zero_bnd="boundary2|default",
    ale_type=-1,
)
dist_fct.set_params(printing=True)
adr_sys = model.create_pde(
    "adr_system", pde_model=ADRVolumeSystemBDF1Model, compartment=comp1, ale_type=1, dim=3
)

comp2 = model.create_compartment(
    "surface", boundary="boundary2|default", bboundary="bboundary1", clamped_bbnd="bboundary1"
)
geom_flow = model.create_pde(
    "willmore", pde_model=GeometricalFlowStationaryModel, compartment=comp2, ale_type=0
)
geom_flow.set_params(rhs=lambda: adr_sys.sol[1] * 1e-3, alpha=1, printing=True)

ale = model.create_ale("ale", compartment=comp2)
ale.set_normal_velocity(geom_flow.V_h)
ale.set_tangential_velocity(CF((0, 0, 0)))

adr_sys.add_nonlinearity(
    target=1, expression="K_nuc*psi1*u1*u2", map={"K_nuc": K_nuc, "psi1": psi1}
)

adr_sys.add_nonlinearity(
    target=2,
    expression="-1*psi0*(K_nuc*psi1*u1*u2 + K_sev*u3**N/(K_n + u3**N)*psi1*u2)",
    map={"K_nuc": K_nuc, "psi1": psi1, "psi0": psi0, "K_sev": K_sev, "K_n": K_n, "N": N},
)

adr_sys.add_nonlinearity(
    target=3,
    expression="K_sev*u3**N/(K_n + u3**N)*psi1*u2",
    map={"psi1": psi1, "K_sev": K_sev, "K_n": K_n, "N": N},
)

model.initialize()
dist_fct.Initialize()
dist_fct.Solve()
id_funct = IfPos(dist_fct.sol[0] - 0.02, 1, 0) * IfPos(z - 0.5, 1, 0)
impulse = IfPos(t, 1, 0) * IfPos(60 - t, 1, 0)

adr_sys.set_params(
    Neu_bnd="membrane|default",
    u0_1=A0 * id_funct,
    b_1=lambda: model.ale.V,
    c_1=K_A,
    rhs_1=I_A + I_SA * impulse,
    gradu_bnd_1=CF((0, 0, 0)),
    bounds_1=[0, 1e100],
    u0_2=B0 * id_funct,
    c_2=K_B,
    b_2=lambda: (
        model.ale.V
        + dist_fct.sol[1]
        * 1e-3
        * sinh(50 * dist_fct.sol[0])
        / cosh(50 * dist_fct.sol[0])
    ),
    rhs_2=psi0 * (I_B + I_SB * impulse),
    gradu_bnd_2=CF((0, 0, 0)),
    bounds_2=[0, 1e100],
    u0_3=C0 * id_funct,
    b_3=lambda: model.ale.V,
    c_3=K_C,
    rhs_3=I_C + I_SC * impulse,
    gradu_bnd_3=CF((0, 0, 0)),
    bounds_3=[0, 1e100],
    printing=True,
)

output_callables = {
    "mass_A": lambda: Integrate(adr_sys.sol[0], mesh),
    "mass_B": lambda: Integrate(adr_sys.sol[1], mesh),
    "mass_C": lambda: Integrate(adr_sys.sol[2], mesh),
    "energy": lambda: Integrate(
        0.5 * (geom_flow.kappa_h - geom_flow.sp_curv_h) ** 2, mesh, VOL_or_BND=BND
    ),
    "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
    "volume": lambda: Integrate(1, mesh, VOL_or_BND=VOL),
}
model.set_params(output_callables=output_callables)

model.run()