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

solvermesh = SolverMesh(mesh)
solvertime = SolverTime(dt, t.Get(), T, t_coef=t)
solver = Solver(solvermesh, solvertime, printing=True)

###################  SURFACE REACTIONS  ##################################

### Actin
A = ADRVolumeBDF1Model(solver, 1, name = 'A', input_params={'u0': A0})

### Barbed ends
B = ADRVolumeBDF1Model(solver, 2, name = 'B', input_params={'u0': B0})

### Cofilin
C = ADRVolumeBDF1Model(solver, 3, name = 'C', input_params={'u0': C0})


# ###################  MEAN CURVATURE FLOW  ##################################

###################  ALE  ##################################

###################  Couplings  ##################################

impulse = IfPos(t, 1, 0)*IfPos(60-t, 1, 0)

def coupA():
    f_nuc = K_nuc*psi1*A.sol*B.sol
    f_A = I_A + I_SA*impulse
    return - f_nuc + f_A
A.set_input_fields({
    'c': K_A,
    'rhs': coupA
})

def coupB():
    f_nuc = K_nuc*psi1*A.sol*B.sol
    f_sev = K_sev*C.sol**N/(K_n + C.sol**N)*psi1*B.sol
    f_B = I_B + I_SB*impulse
    return psi0*(f_nuc+f_sev+f_B)
B.set_input_fields({
    'c': K_B,
    'rhs': coupB
})

def coupC():
    f_sev = K_sev*C.sol**N/(K_n + C.sol**N)*psi1*B.sol
    f_C = I_C + I_SC*impulse
    return - f_sev + f_C
C.set_input_fields({
    'c': K_C,
    'rhs': coupC
})

# solver.ale.deformation_field = lambda: mc.displacement
# solver.ale.velocity_field = lambda : mc.displacement/dt
# solver.ale.mat_velocity_field = lambda : mc.displacement/dt

xdata = []
ydata = [[], [], []]

# sceneA = Draw(A.sol, mesh)
# sceneB = Draw(B.sol, mesh)
# sceneC = Draw(C.sol, mesh)
for i, sol in enumerate(solver()):
    # sceneA.Redraw()
    # sceneB.Redraw()
    # sceneC.Redraw()

    xdata.append(t.Get()/60)
    ydata[0].append(A.sol(mip))
    ydata[1].append(B.sol(mip))
    ydata[2].append(C.sol(mip))

plt.plot(xdata, ydata[0], label = 'A')
plt.plot(xdata, ydata[2], label = 'C')
plt.ylim([0, 0.7])
plt.legend()
plt.show()

plt.plot(xdata, np.array(ydata[1])/1e4, label = 'B')
plt.ylim([1, 2])
plt.legend()
plt.show()

print(A.sol(mip))
print(B.sol(mip))
print(C.sol(mip))
# %%
