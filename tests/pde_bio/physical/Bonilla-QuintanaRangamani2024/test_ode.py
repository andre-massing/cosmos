# %%
from ngsolve import *
from ngsolve.webgui import Draw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cosmos import *

from dendritic_spine_geom import generate_synapse2d

mesh = generate_synapse2d(maxh=0.02)
mip = mesh(0, 0.5)

###################  PARAMETERS  ##################################

A0 = 20
B0 = 3000*3.6
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
N = 3.5
N = 3

T = 4*60
dt = 1e-1
n = 200
sample_rate = np.maximum(int(abs(T/dt/n)), 1)
dt = Parameter(dt)
t = Parameter(-4*60)

from cosmos.core.model import CosmosModel

model = CosmosModel(parentmesh=mesh, dt=dt, t=t, t0 = t.Get(), t1 = T,
                    root = '.', sample_rate = 10, name = 'test_ode',
                    coupling_type = 'implicit')

###################  BULK REACTIONS  ##################################
from cosmos.pde.adr.volume.adr_volume_system_bdf1_model import ADRVolumeSystemBDF1Model

comp1 = model.create_compartment('bulk', material = 'default', boundary = 'membrane|default')
adr_sys = model.create_pde('adr_system', pde_model=ADRVolumeSystemBDF1Model, compartment=comp1, ale_type = 1, dim = 3)

impulse = IfPos(t, 1, 0)*IfPos(60-t, 1, 0)

adr_sys.add_nonlinearity(target = 1, expression = 'K_nuc*psi1*u1*u2', map={'K_nuc': K_nuc, 'psi1': psi1})

adr_sys.add_nonlinearity(target = 2, expression = '-1*psi0*(K_nuc*psi1*u1*u2 + K_sev*u3**N/(K_n + u3**N)*psi1*u2)', 
                         map={'K_nuc': K_nuc, 'psi1': psi1, 'psi0': psi0, 'K_sev': K_sev, 'K_n': K_n, 'N': N})

adr_sys.add_nonlinearity(target = 3, expression = 'K_sev*u3**N/(K_n + u3**N)*psi1*u2',
                         map={'psi1': psi1, 'K_sev': K_sev, 'K_n': K_n, 'N': N})

# id_funct = IfPos(y-0.5, 1, 0)
id_funct = CF(1)

adr_sys.set_params(
    Neu_bnd = 'membrane|default',
    u0_1 = A0*id_funct,
    c_1 = K_A,
    rhs_1 = I_A + I_SA*impulse,
    gradu_bnd_1 = CF((0,0)),
    u0_2 = B0*id_funct,
    c_2 = K_B,
    rhs_2 = psi0*(I_B + I_SB*impulse),
    gradu_bnd_2 = CF((0,0)),
    u0_3 = C0*id_funct,
    c_3 = K_C,
    rhs_3 = I_C + I_SC*impulse,
    gradu_bnd_3 = CF((0,0)),
    printing = True
)

output_callables = {
    'mass_A': lambda: Integrate(adr_sys.sol[0], mesh),
    'mass_B': lambda: Integrate(adr_sys.sol[1], mesh),
    'mass_C': lambda: Integrate(adr_sys.sol[2], mesh),
    'energy': lambda: 0,
    'area': lambda: Integrate(1, mesh, VOL_or_BND = BND),
    'volume': lambda: Integrate(1, mesh, VOL_or_BND = VOL),
    'control1_A': lambda: adr_sys.sol[0].vec.data[5],
    'control1_B': lambda: adr_sys.sol[1].vec.data[5],
    'control1_C': lambda: adr_sys.sol[2].vec.data[5],
    'control2_A': lambda: adr_sys.sol[0].vec.data[6],
    'control2_B': lambda: adr_sys.sol[1].vec.data[6],
    'control2_C': lambda: adr_sys.sol[2].vec.data[6],
    'control3_A': lambda: adr_sys.sol[0].vec.data[7],
    'control3_B': lambda: adr_sys.sol[1].vec.data[7],
    'control3_C': lambda: adr_sys.sol[2].vec.data[7],
    'control4_A': lambda: adr_sys.sol[0](mip),
    'control4_B': lambda: adr_sys.sol[1](mip),
    'control4_C': lambda: adr_sys.sol[2](mip),
}
model.set_params(
    output_callables = output_callables
)

# sceneA = Draw(adr_sys.sol[0], mesh)
sceneB = Draw(adr_sys.sol[1], mesh)
# sceneC = Draw(adr_sys.sol[2], mesh)
for i, sol in enumerate(model()):
    # sceneA.Redraw()
    sceneB.Redraw()
    # sceneC.Redraw()