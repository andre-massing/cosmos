# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.solvers.willmore_solver import WillmoreSolver
from cosmos.solvers.vector_poisson_solver import VectorPoissonSolver
from cosmos.solvers.distance_solver import DistanceSolver
import netgen.occ as occ
import time
import numpy as np
import pandas as pd
import os
directory = "./proto2D_tests"
os.makedirs(directory, exist_ok=True)

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

mesh, geo = generate_synapse2d(maxh=0.02)

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
            'domain_name': 'membrane',
            'clamped_bnd': 'membrane_bnd'}
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
            max_y = []
            t_vec = []

            vtkout_s = VTKOutput(mesh, coefs=[gfu_displacement, gfu_mean_curvature],
                        names=['surface_displacement', 'mean_curvature'], filename=sub_dir + '/results_surf' )
            vtkout_v = VTKOutput(mesh, coefs=[gfu_mesh_moving, gfu_distance, gfu_velocity],
                                    names=['volume_displacement', 'distance_function', 'distance_f_gradient'], filename=sub_dir + '/results_vol' )
            
            scene = Draw(gfu_distance, mesh, deformation = gfu_mesh_moving)

            try:

                out_step = int(int(w_solver.T/w_solver.dt.Get())/200)+1

                for count, sol in enumerate(w_solver()):

                    v_poisson_solver.params['boundary_c']['dirichlet'] =[[w_solver.params['domain_name'], w_solver.sol_h[0]],
                            ['default', CF((0,)*mesh.dim)]]
                    next(v_poisson_solver())

                    distance_solver.params['displacement'] = v_poisson_solver.sol_h[0]
                    next(distance_solver())

                    mesh.SetDeformation(w_solver.sol_h[0])
                    willmore_energy.append(Integrate(InnerProduct(w_solver.sol_h[1], w_solver.sol_h[1]),
                                                      mesh, order = 2*fes_order, VOL_or_BND=BND))
                    surface_area.append(Integrate(1, mesh, order = 2*fes_order, VOL_or_BND=BND))
                    m_y = -np.inf
                    for v in mesh.vertices:
                        aux = np.array(v_poisson_solver.sol_h[0](mesh(*v.point))[1])
                        m_y = np.max([m_y, v.point[1]+aux])
                    max_y.append(m_y)
                    mesh.UnsetDeformation()

                    t_vec.append(w_solver.t.Get())

                    gfu_displacement.Set(w_solver.sol_h[0], definedon=mesh.Boundaries(w_solver.params['domain_name']))
                    gfu_mean_curvature.Set(w_solver.sol_h[1], definedon=mesh.Boundaries(w_solver.params['domain_name']))
                    gfu_mesh_moving.Set(v_poisson_solver.sol_h[0])
                    gfu_distance.Set(distance_solver.sol_h[0])
                    gfu_velocity.Set(distance_solver.sol_h[1])

                    if count%out_step == 0:

                        vtkout_s.Do(time=w_solver.t.Get(), vb=VOL)
                        vtkout_v.Do(time=w_solver.t.Get(), vb=VOL)

                    scene.Redraw()

            except:

                pass

            # Printing the results
            
            name = sub_dir + "/results.dat"

            horizontal_labels = ['name_dt', 'willmore_energy', 'surface_area', 'max_y']
            vertical_labels =  [f'{x:.2e}' for x in t_vec]
            output = np.column_stack((vertical_labels, np.column_stack([willmore_energy, surface_area, max_y])))

            df = pd.DataFrame(output, columns=horizontal_labels)
            df.to_csv(name, sep='\t', index=False)

       
# %%
