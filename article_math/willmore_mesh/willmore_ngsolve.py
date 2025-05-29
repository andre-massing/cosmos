# %%

from cosmos.solvers.solver_unsteady import UnsteadySolver
from cosmos.solvers.schemes import BDF1, BDF2
from cosmos.pdes.pde_willmore_dziuk import WillmoreDziuk
from cosmos.pdes.pde_willmore_dziuk_stab import WillmoreDziukStab
from cosmos.pdes.pde_tools import SaveError, SaveSolution
from netgen.occ import *
from ngsolve import *
from netgen.meshing import MeshingStep
from ngsolve.webgui import Draw
import numpy as np
import pandas as pd

t = Parameter(0.0)
dt = Parameter(1e-5)
T = 0.01

ngsolve = Sphere( (0,0,0), 0.8) - Sphere( (0,0,0), 0.5)  - Cylinder((-1,0.5,0.5),X, r=0.4, h=2) - Cylinder((-0.5,-1,-0.5),Y, r=0.4, h=2) - Cylinder((0.5,-0.5,-1),Z, r=0.4, h=2)

will_sol = [
    # WillmoreDziuk(postprocess = None, mc_autoupdate = True),
    WillmoreDziukStab(postprocess = 1, mc_autoupdate = True)
]

for i, sol in enumerate(will_sol):

    sol_save = SaveSolution(folderpath = './willmore_ngsolve',
                        filename = 'ngsolve' + str(i),
                        sample_rate = 1)
    sol.SaveSol(sol_save)

    t = Parameter(0.0)
    dt = Parameter(1e-4)
    T = 0.01

    mesh = Mesh(OCCGeometry(ngsolve).GenerateMesh(maxh=0.1, perfstepsend=MeshingStep.MESHSURFACE))

    solver = UnsteadySolver(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(sol, BDF1(conservative=False))
    solver.ale.SetMeshDeformation(sol.gfu.components[0])

    scene = Draw(sol.kappa_h, mesh, deformation = solver.ale.deformation, min = 0)
    for sol in solver():
        scene.Redraw()