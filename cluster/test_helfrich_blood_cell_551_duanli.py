# %%
from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from netgen.occ import *
from cosmos.utils.generate_surface_meshes import generate_ellipse
import numpy as np
import logging
from test_helfrich_blood_cell_config import CFG

import os
filename = os.path.splitext(os.path.basename(__file__))[0]

Id = CF((x, 0, 0))
n = specialcf.normal(3)

maxh = CFG.maxh
mesh, _ = generate_ellipse(maxh=maxh, a = 5, b=5, c=1)

logging.getLogger().setLevel(logging.DEBUG)

dt = CFG.dt
Tend = CFG.Tend
solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)

solvermesh = SolverMesh(mesh)
solver = Solver(solvermesh, solvertime, iter=False,
                name = filename)
solver.output_params(CFG.output_folder, sample_rate=2)

input_params_willmore = {
    "autoupdate": True,
}
willmore = WillmoreBoundaryAPVPBDF1Model(solver, 1, name = 'willmore_boundary_apvp_bdf1', 
                                     input_params = input_params_willmore)

ale = ALEModel(solver, 3)
ale.set_bnd_displacement(willmore.displacement, 'default', redistribute=True)

solver.save_model_solution("willmore_boundary_apvp_bdf1", "displacement")
solver.save_model_solution("willmore_boundary_apvp_bdf1", "mean_curvature")

try:
    with open(CFG.output_folder + filename + '/simulation.txt', "w") as f:
        f.write('Area\tVolume\tEnergy\n')
        for _ in solver():
            area = Integrate(1, mesh, VOL_or_BND = BND)
            volume = Integrate(Id*n, mesh, VOL_or_BND = BND)
            energy = Integrate(InnerProduct(willmore.mean_curvature, willmore.mean_curvature), mesh, VOL_or_BND = BND)
            f.write(str(area) + '\t' + str(volume) + '\t' + str(energy) +'\n')
            pass
except Exception as e:
    with open(CFG.output_folder + filename + '/error_file.txt', "w") as f:
        f.write('Simulation terminated with error\n')
        f.write('Time: ' +  str(solver.time.t.Get()) +', iter: '+ str(solver.time.iter) + '\n')
        f.write('Cause: ' + str(e))