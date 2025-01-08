
# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_surface_meshes import generate_sphere
from cosmos.solvers.willmore_solver import WillmoreSolver
from cosmos.utils.generate_surface_meshes import ErrorTools

import os
import numpy as np
import pandas as pd
directory = "./moving_sphere_tests"
os.makedirs(directory, exist_ok=True)

t = Parameter(0.0)

R = CF(1 + t)
displ_ex = CF((x,y,z))/CF((x,y,z))*t
mc_exact = -2/R*CF((x,y,z))

initial_mc_type = [0, 1]
postprocessing_type = [None, 'harmonic', 'harmonic_step']
stabilized_mc = [False, True]

fes_order = 1
T = 1.0
dt0 = 0.05
dh0 = 0.2
h_power = 1.5
h_n_ref = 1
t_power = 1.5
t_n_ref = 1
dh_vals = dh0/np.power(h_power, np.arange(h_n_ref+1))
dt_vals = dt0/np.power(t_power, np.arange(t_n_ref+1))


dt = Parameter(dt0)
mesh, geo = generate_sphere(maxh = dh0, order_g=1, R=1)
rhs = CF(0.0)

params = {'rhs': rhs,
            'domain_name': '.*',
            'spontaneous_curv': -2/R}
solver = WillmoreSolver(mesh, fes_order=fes_order, dt=dt, t=t, T=T, params = params, verbose = 0)


for i, ic in enumerate(initial_mc_type):

    for j, stab in enumerate(stabilized_mc):


        for k, pp in enumerate(postprocessing_type):

            solver.t.Set(0.0)

            solver.params['initial_mc_type'] = ic
            solver.params['postprocessing_type'] = pp 
            solver.params['stabilized_mc'] = stab

            V = VectorH1(mesh, order = 1)
            gfu1 = GridFunction(V)
            gfu2 = GridFunction(V)
            gfu1.Set(solver.sol_h[0], definedon=mesh.Boundaries('.*'))
            gfu2.Set(solver.sol_h[1], definedon=mesh.Boundaries('.*'))

            scene = Draw(gfu2, mesh, deformation = gfu1)
            import time

            for sol in solver():

                gfu1.Set(solver.sol_h[0], definedon=mesh.Boundaries('.*'))
                gfu2.Set(solver.sol_h[1], definedon=mesh.Boundaries('.*'))
                scene.Redraw()

            # err_params = {'exact_solutions': [displ_ex, mc_exact],
            #   'vol_or_bnd': [BND, BND],
            #   'norms': ['L2L2', 'L2L2']}
            # errors = ErrorTools(solver, params=err_params)

            # conv_params = {
            #         'geometry': geo,
            #         'space_params': [dh0, h_power, h_n_ref],
            #         'time_params': [dt0, t_power, t_n_ref]
            #     }

            # errs = errors.compute_eoc(conv_params)

            # # Creating sub-directory for results
            # sub_dir = directory + '/' + str(i) + str(j) + str(k)
            # os.makedirs(sub_dir, exist_ok=True)

            # # Printing the errors for the displacement
            
            # name = sub_dir + "/displacement.dat"

            # horizontal_labels =  [f'{x:.2e}' for x in dh_vals]
            # horizontal_labels = ["dt-dh"] + horizontal_labels
            # output = np.column_stack(([f'{m:.2e}' for m in dt_vals], errs[0]))

            # df = pd.DataFrame(output, columns=horizontal_labels)
            # df.to_csv(name, sep='\t', index=False)

            # # Printing the errors for the mean curvature

            # name = sub_dir + "/mean.dat"

            # horizontal_labels =  [f'{x:.2e}' for x in dh_vals]
            # horizontal_labels = ["dt-dh"] + horizontal_labels
            # output = np.column_stack(([f'{m:.2e}' for m in dt_vals], errs[1]))

            # df = pd.DataFrame(output, columns=horizontal_labels)
            # df.to_csv(name, sep='\t', index=False)
            
# %%

# %%

# %%
