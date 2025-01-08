# %%

from cosmos.solvers.willmore_solver import WillmoreSolver
import numpy as np
from netgen import stl
from netgen.meshing import MeshingStep
from ngsolve import *
from ngsolve.webgui import Draw

from ngsolve import ngsglobals
ngsglobals.msg_level = 2

# spine = stl.STLGeometry('./spine_closed.stl')

# mesh = Mesh(spine.GenerateMesh(maxh = 0.05, perfstepsend=MeshingStep.MESHSURFACE,
#             quad_dominated=False,
#             grading=0.7,
#             optimize2d = "smsmsmSmSmSm",
#             yangle=50,
#             contyangle=50,
#             edgecornerangle=30))
mesh = Mesh('./spine_closed.vol')

Draw(mesh)

t = Parameter(0.0)
rhs = CF((0,0,0))

fes_order = 1

T = 0.1
dt = Parameter(1e-4)

params = {'rhs': rhs,
            'domain_name': '.*',
            'initial_mc_type': 1,
            'stabilized_mc': True,
            'postprocessing_type': 'harmonic_step'}

solver = WillmoreSolver(mesh, fes_order=fes_order, dt=dt, t=t, T=T, params = params)

# ramping up the initial values
nsteps = int((T-t.Get())/dt.Get())
ramp_steps = int(nsteps*0.05)
exp0 = -7
exp1 = int(np.log10(dt.Get()))
ramp_vals = np.logspace(exp0, exp1, num=ramp_steps)

time_vals = np.concatenate((ramp_vals, np.ones(nsteps)*dt.Get()))

settings={"camera": {"transformations": [{"type": "rotateX", "angle": 0}]}}

fes = VectorH1(mesh, order = 1, definedon = mesh.Boundaries('.*'))
gfu0 = GridFunction(fes)
gfu0.Set(solver.displ_h, definedon = mesh.Boundaries('.*'))
gfu1 = GridFunction(fes)
gfu1.Set(solver.sol_h[1], definedon = mesh.Boundaries('.*'))
scene = Draw(gfu1, mesh, deformation = gfu0)

solver.dt.Set(time_vals[0])

vtkout = VTKOutput(mesh,coefs=[gfu0, gfu1],names=["displacement", "mean_curvature"],filename="./realspine3D/realspine3D")

vtkout.Do(time=solver.t.Get(), vb = BND)

for i, sol in enumerate(solver()):

    if i%10 == 0:
        vtkout.Do(time=solver.t.Get(), vb = BND)

    solver.dt.Set(time_vals[i+1])

    gfu1.Set(solver.sol_h[1], definedon = mesh.Boundaries('.*'))
    gfu0.Set(solver.displ_h, definedon = mesh.Boundaries('.*'))

    scene.Redraw()
# %%

# %%
