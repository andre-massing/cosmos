# %%

from cosmos.solvers.solver_unsteady import UnsteadySolver
from cosmos.solvers.schemes import BDF1, BDF2
from cosmos.pdes.pde_willmore_bgn import WillmoreBGN
from cosmos.pdes.pde_willmore_bgn_stab import WillmoreBGNStab
from cosmos.pdes.pde_willmore_dziuk import WillmoreDziuk
from cosmos.pdes.pde_willmore_dziuk_stab import WillmoreDziukStab
from cosmos.utils.generate_surface_meshes import generate_box
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import pandas as pd

will_sol = [
    # WillmoreBGN(postprocess = None, mc_autoupdate = True),
    # WillmoreBGNStab(postprocess = None, mc_autoupdate = True),
    # WillmoreDziuk(postprocess = 0, mc_autoupdate = True),
    WillmoreDziukStab(postprocess = 0, mc_autoupdate = True)
]

for i, sol in enumerate(will_sol):

    t = Parameter(0.0)
    dt = Parameter(1e-2)
    T = 1

    mesh, _ = generate_box(maxh=0.2, a = 1, b = 6, c = 1, vol_or_bnd='BND')

    solver = UnsteadySolver(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(sol, BDF1(conservative=False))
    solver.ale.SetMeshDeformation(sol.gfu.components[0])

    scene = Draw(sol.kappa_h, mesh, deformation = solver.ale.deformation, min = 0)
    for sol in solver():
        scene.Redraw()