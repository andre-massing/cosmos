# %%
from ngsolve import *
from ngsolve.webgui import Draw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cosmos import *

from test_dendritic_spine_geom import generate_synapse2d

mesh, _ = generate_synapse2d(maxh=0.02)
mip = mesh(0.3, 0.7)

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

T = 4*60
dt = 1e-1
n = 200
sample_rate = np.maximum(int(abs(T/dt/n)), 1)
dt = Parameter(dt)
t = Parameter(-4*60)

folderpath = './quintana_2d'

from cosmos.core.model import CosmosModel

model = CosmosModel(parentmesh=mesh, dt=dt, t=t, t0 = t.Get(), t1 = T,
                    root = '.', sample_rate = 10, name = 'adr_system_Mayte')

###################  BULK REACTIONS  ##################################
from cosmos.pde.adr.volume.adr_volume_system_bdf1_model import ADRVolumeSystemBDF1Model

comp1 = model.create_compartment('bulk', material = 'default', boundary = 'membrane|default')
adr_sys = model.create_pde('adr_system', pde_model=ADRVolumeSystemBDF1Model, compartment=comp1, dim = 3)

impulse = IfPos(t, 1, 0)*IfPos(60-t, 1, 0)

def coupA():
    f_nuc = K_nuc*psi1*adr_sys.sol[0]*adr_sys.sol[1]
    f_A = I_A + I_SA*impulse
    return - f_nuc + f_A

def coupB():
    f_nuc = K_nuc*psi1*adr_sys.sol[0]*adr_sys.sol[1]
    f_sev = K_sev*adr_sys.sol[2]**N/(K_n + adr_sys.sol[2]**N)*psi1*adr_sys.sol[1]
    f_B = I_B + I_SB*impulse
    return psi0*(f_nuc+f_sev+f_B)

def coupC():
    f_sev = K_sev*adr_sys.sol[2]**N/(K_n + adr_sys.sol[2]**N)*psi1*adr_sys.sol[1]
    f_C = I_C + I_SC*impulse
    return - f_sev + f_C

adr_sys.set_params(
    u0_1 = A0,
    c_1 = K_A,
    rhs_1 = coupA,
    u0_2 = B0,
    c_2 = K_B,
    rhs_2 = coupB,
    u0_3 = C0,
    c_3 = K_C,
    rhs_3 = coupC,
)

xdata = []
ydata = [[], [], []]

# sceneA = Draw(adr_sys.sol[0], mesh)
# sceneB = Draw(adr_sys.sol[1], mesh)
# sceneC = Draw(adr_sys.sol[2], mesh)
for i, sol in enumerate(model()):
    # sceneA.Redraw()
    # sceneB.Redraw()
    # sceneC.Redraw()

    xdata.append(t.Get()/60)
    ydata[0].append(adr_sys.sol[0](mip))
    ydata[1].append(adr_sys.sol[1](mip))
    ydata[2].append(adr_sys.sol[2](mip))

plt.plot(xdata, ydata[0], label = 'A')
plt.plot(xdata, ydata[2], label = 'C')
plt.ylim([0, 0.7])
plt.legend()
plt.show()

plt.plot(xdata, np.array(ydata[1])/1e4, label = 'B')
plt.ylim([1, 2])
plt.legend()
plt.show()

print(adr_sys.sol[0](mip))
print(adr_sys.sol[1](mip))
print(adr_sys.sol[2](mip))
# %%

# %%
