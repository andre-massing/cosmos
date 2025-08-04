# %%
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *   # Opencascade for geometry modeling
import pandas as pd
import matplotlib.pyplot as plt
import netgen.occ as occ
from cosmos.pdes.pde_willmore_dziuk import WillmoreDziuk
from cosmos.pdes.pde_willmore_dziuk_stab import WillmoreDziukStab
from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import Steady, BDF1, BDF2
from cosmos.pdes.pde_tools import SaveSolution
from cosmos.pdes.pde_adr_vol_ad import VolADR
import numpy as np

def generate_synapse2d(maxh, order_g = 1, external = False):
    wp = occ.WorkPlane()
    wp.Rotate(90).Line(0.1).Rotate(-90)
    wp.Line(0.15).Arc(0.1, 90)
    wp.Line(0.3).Arc(0.1, 90)
    wp.Arc(0.1, -180).Line(0.3).Arc(0.1,-180)
    wp.Arc(0.1, 90).Line(0.3).Arc(0.1, 90)
    wp.Line(0.15)
    wp.Rotate(-90).Line(0.1).Rotate(-90)
    wp.Line(0.6)
    wp.Reverse()
    synapse = wp.Face()

    # synapse.edges.col = (0,0,0)
    # synapse.edges.name = "inner_bnd"

    # for i in range(11):
    #     synapse.edges[i+1].name = "membrane"
    # synapse.faces.name = "inner_space"
    # synapse.vertices[1].name = 'membrane_bnd'
    # synapse.vertices[24].name = 'membrane_bnd'
    # synapse.vertices[2].name = 'membrane_bnd'
    # synapse.vertices[23].name = 'membrane_bnd'

    for i in range(7):
        synapse.edges[i+3].name = "membrane"
    synapse.faces.name = "inner_space"
    synapse.vertices[5].name = 'membrane_bnd'
    synapse.vertices[20].name = 'membrane_bnd'
    synapse.vertices[6].name = 'membrane_bnd'
    synapse.vertices[19].name = 'membrane_bnd'

    if external:
        # Using a 1x1 square as background and subtract the synapse profile to
        # create the external space and naming the components

        wp2 = occ.WorkPlane()
        wp2.Rectangle(1, 1)
        face = wp2.Face()
        face.edges.Min(occ.X).name = "outer_bnd"
        face.edges.Max(occ.X).name = "outer_bnd"
        face.edges.Max(occ.Y).name = "outer_bnd"

        # subtraction of the background to create the outer space
        ambient = face - synapse
        ambient.faces.name = "outer_space"

        # Merging the two spaces mantaining the interface between the two
        total = occ.Glue([synapse, ambient])

    else:

        total = synapse

    geo = occ.OCCGeometry(total, dim = 2)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(order_g)
    return mesh, geo

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

###################  SURFACE REACTIONS  ##################################

### Actin
A = VolADR(time_scheme=BDF2(), c=K_A, u0 = A0)
sol_save_A = SaveSolution(folderpath = folderpath,
                        filename = 'quintana_2d_A',
                        sample_rate = sample_rate)
A.SaveSol(sol_save_A)

### Barbed ends
B = VolADR(time_scheme=BDF2(), c = K_B, u0 = B0)
sol_save_B = SaveSolution(folderpath = folderpath,
                        filename = 'quintana_2d_B',
                        sample_rate = sample_rate)
B.SaveSol(sol_save_B)

### Cofilin
C = VolADR(time_scheme=BDF2(), c = K_C, u0 = C0)
sol_save_C = SaveSolution(folderpath = folderpath,
                        filename = 'quintana_2d_C',
                        sample_rate = sample_rate)
C.SaveSol(sol_save_C)

# ###################  MEAN CURVATURE FLOW  ##################################
# willmore = WillmoreDziuk(postprocess = True, domain = 'membrane', clamped_bnd = 'membrane_bnd',
#                          time_scheme=BDF1())
# # mc = MCBGN(time_scheme=BDF1(), postprocess=True, alpha=0.01)
# mc = WillmoreDziuk(time_scheme=BDF1(), postprocess=True, kappa=1)
# sol_save = SaveSolution(folderpath = folderpath,
#                         filename = 'quintana_2d_mc',
#                         sample_rate = sample_rate)
# mc.SaveSol(sol_save)

###################  Solver  ##################################
from cosmos.solvers.solvers import Dynamic
solver = Dynamic(mesh = mesh, t = t, T = T, dt = dt)

###################  ALE  ##################################

###################  Couplings  ##################################
from cosmos.pdes.coupling_weak import WeakCoupling
cpl = WeakCoupling(tol = 1e-4, type = 'implicit')
cpl.AddPDEs(A, B, C)
solver.AddPDE(cpl)

impulse = IfPos(t, 1, 0)*IfPos(60-t, 1, 0)

def coupA():
    f_nuc = K_nuc*psi1*A.solute*B.solute
    f_A = I_A + I_SA*impulse
    return - f_nuc + f_A
A.rhs.value = coupA

def coupB():
    f_nuc = K_nuc*psi1*A.solute*B.solute
    f_sev = K_sev*C.solute**N/(K_n + C.solute**N)*psi1*B.solute
    f_B = I_B + I_SB*impulse
    return psi0*(f_nuc+f_sev+f_B)
B.rhs.value = coupB

def coupC():
    f_sev = K_sev*C.solute**N/(K_n + C.solute**N)*psi1*B.solute
    f_C = I_C + I_SC*impulse
    return - f_sev + f_C
C.rhs.value = coupC

# solver.ale.deformation_field = lambda: mc.displacement
# solver.ale.velocity_field = lambda : mc.displacement/dt
# solver.ale.mat_velocity_field = lambda : mc.displacement/dt

xdata = []
ydata = [[], [], []]

# sceneA = Draw(A.solute, mesh)
# sceneB = Draw(B.solute, mesh)
# sceneC = Draw(C.solute, mesh)
for i, sol in enumerate(solver()):
    # sceneA.Redraw()
    # sceneB.Redraw()
    # sceneC.Redraw()

    xdata.append(t.Get()/60)
    ydata[0].append(A.solute(mip))
    ydata[1].append(B.solute(mip))
    ydata[2].append(C.solute(mip))

plt.plot(xdata, ydata[0], label = 'A')
plt.plot(xdata, ydata[2], label = 'C')
plt.ylim([0, 0.7])
plt.legend()
plt.show()

plt.plot(xdata, np.array(ydata[1])/1e4, label = 'B')
plt.ylim([1, 2])
plt.legend()
plt.show()

print(A.solute(mip))
print(B.solute(mip))
print(C.solute(mip))
# %%
