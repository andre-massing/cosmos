# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.solvers.willmore_solver import WillmoreSolver
from cosmos.solvers.vector_poisson_solver import VectorPoissonSolver
from cosmos.solvers.distance_solver import DistanceSolver
import time
import numpy as np
import pandas as pd
import os
directory = "./square_tests"
os.makedirs(directory, exist_ok=True)

mesh = Mesh(unit_square.GenerateMesh(maxh = 0.05))

initial_mc_type = [0, 1]
postprocessing_type = [None, 'harmonic', 'harmonic_step']
stabilized_mc = [False, True]

fes_order = 1
T = 0.016
dt0 = 0.0001
dh0 = 0.2

dt = Parameter(dt0)
t = Parameter(0.0)
rhs = CF((0, 0))

params = {'rhs': rhs,
            'domain_name': '.*'}
w_solver = WillmoreSolver(mesh, fes_order=fes_order, dt=dt, t=t, T=T, params = params, verbose = 0)

bnd_cond={'dirichlet': [None]}
v_p_params = {'boundary_c': bnd_cond}
v_poisson_solver = VectorPoissonSolver(mesh, fes_order=fes_order, params = v_p_params)

distance_params = {'zero_bnd': w_solver.params['domain_name']}
distance_solver = DistanceSolver(mesh, fes_order=fes_order, params = distance_params)

# Grid-functions for printing purposes
fes2 = VectorH1(mesh, order = 1)
fes3 = H1(mesh, order = 1)

gfu_displacement = GridFunction(fes2)
gfu_mean_curvature = GridFunction(fes2)
gfu_mesh_moving = GridFunction(fes2)
gfu_velocity = GridFunction(fes2)
gfu_distance = GridFunction(fes3)


for i, ic in enumerate(initial_mc_type):

    for j, stab in enumerate(stabilized_mc):

        for k, pp in enumerate(postprocessing_type):

            # Creating sub-directory for results
            sub_dir = directory + '/' + str(i) + str(j) + str(k)
            os.makedirs(sub_dir, exist_ok=True)

            w_solver.t.Set(0.0)

            w_solver.params['initial_mc_type'] = ic
            w_solver.params['postprocessing_type'] = pp 
            w_solver.params['stabilized_mc'] = stab
            
            willmore_energy = []
            surface_area = []
            t_vec = []

            vtkout_s = VTKOutput(mesh, coefs=[gfu_displacement, gfu_mean_curvature],
                        names=['surface_displacement', 'mean_curvature'], filename=sub_dir + '/results_surf' )
            vtkout_v = VTKOutput(mesh, coefs=[gfu_mesh_moving, gfu_distance, gfu_velocity],
                                    names=['volume_displacement', 'distance_function', 'distance_f_gradient'], filename=sub_dir + '/results_vol' )
            
            scene = Draw(gfu_distance, mesh, deformation = gfu_mesh_moving)

            try:

                for sol in w_solver():

                    v_poisson_solver.params['boundary_c']['dirichlet'] =[[w_solver.params['domain_name'], w_solver.sol_h[0]],
                            ['default', CF((0,)*mesh.dim)]]
                    next(v_poisson_solver())

                    distance_solver.params['displacement'] = v_poisson_solver.sol_h[0]
                    next(distance_solver())

                    mesh.SetDeformation(w_solver.sol_h[0])
                    willmore_energy.append(Integrate(InnerProduct(w_solver.sol_h[1], w_solver.sol_h[1]),
                                                      mesh, order = 2*fes_order, VOL_or_BND=BND))
                    surface_area.append(Integrate(1, mesh, order = 2*fes_order, VOL_or_BND=BND))
                    mesh.UnsetDeformation()

                    t_vec.append(w_solver.t.Get())

                    gfu_displacement.Set(w_solver.sol_h[0], definedon=mesh.Boundaries(w_solver.params['domain_name']))
                    gfu_mean_curvature.Set(w_solver.sol_h[1], definedon=mesh.Boundaries(w_solver.params['domain_name']))
                    gfu_mesh_moving.Set(v_poisson_solver.sol_h[0])
                    gfu_distance.Set(distance_solver.sol_h[0])
                    gfu_velocity.Set(distance_solver.sol_h[1])

                    vtkout_s.Do(time=w_solver.t.Get(), vb=VOL)
                    vtkout_v.Do(time=w_solver.t.Get(), vb=VOL)

                    scene.Redraw()

                    print('time:', w_solver.t.Get(), end="\r")

            except:

                pass

            # Printing the results
            
            name = sub_dir + "/results.dat"

            horizontal_labels = ['name-dt', 'willmore_energy', 'surface_area']
            vertical_labels =  [f'{x:.2e}' for x in t_vec]
            output = np.column_stack((vertical_labels, np.column_stack([willmore_energy, surface_area])))

            df = pd.DataFrame(output, columns=horizontal_labels)
            df.to_csv(name, sep='\t', index=False)

       
# %%