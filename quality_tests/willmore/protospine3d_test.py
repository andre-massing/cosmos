# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.solvers.willmore_solver import WillmoreSolver
import netgen.occ as occ
import time
import numpy as np
import pandas as pd
import os
directory = "./proto3D_tests"
os.makedirs(directory, exist_ok=True)

def generate_synapse3d_og(maxh, order_g = 1, external = False):
    wp = occ.WorkPlane()
    wp.Rotate(90).Line(0.1).Rotate(-90)
    wp.Line(0.2).Line(0.15).Arc(0.1, 90)
    wp.Line(0.3).Arc(0.1, 90)
    wp.Arc(0.1, -180).Line(0.15)
    wp.Rotate(-90).Line(0.8)
    wp.Close()
    wp.Reverse()
    synapse = wp.Face()
    synapse = synapse.Move((-0.5,0,0))
    synapse = synapse.Rotate(occ.Axis((0,0,0), occ.X), 90)

    synapse.edges.name = "membrane"
    synapse.edges[occ.X<-0.3].name = "membrane_bnd"
    synapse.edges.Min(occ.Z).name = "inner_bnd"
    synapse.edges.Min(occ.X).name = "inner_bnd"
    synapse.edges.Max(occ.X).name = "default"

    synapse3d = synapse.Revolve(occ.Axis((0, 0, 0),occ.Z), 360)

    if external:

        box = occ.Box(occ.Pnt(-0.5, -0.5, 0), occ.Pnt(0.5, 0.5, 0.1))
        box = box - synapse3d
        box.faces.name = "inner_bnd"
        box.faces[occ.Z>=0.1].name = "membrane_bnd"

        synapse3d = occ.Glue([synapse3d, box])
        synapse3d.mat("inner_space")

        ambient3d = occ.Box(occ.Pnt(-0.5, -0.5, 0), occ.Pnt(0.5, 0.5, 1.0))
        ambient3d = ambient3d - synapse3d
        ambient3d.mat("outer_space")
        ambient3d.faces.Min(occ.X).name = "outer_bnd"
        ambient3d.faces.Max(occ.X).name = "outer_bnd"
        ambient3d.faces.Min(occ.Y).name = "outer_bnd"
        ambient3d.faces.Max(occ.Y).name = "outer_bnd"
        ambient3d.faces.Max(occ.Z).name = "outer_bnd"

        total = occ.Glue([synapse3d, ambient3d])

    else:

        total = synapse3d

    geo = occ.OCCGeometry(total)

    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(order_g)
    return mesh, geo


def generate_synapse3d(maxh, order_g = 1):

    pnt1 = occ.Pnt(0.5, 0, 0)
    pnt2 = occ.Pnt(0.2, 0, 0)
    pnt3 = occ.Pnt(0.2, 0, 0.5)
    pnt4 = occ.Pnt(0.4, 0, 0.7)
    pnt5 = occ.Pnt(0.2, 0, 0.9)
    pnt6 = occ.Pnt(0 , 0 , 0.9)

    seg1 = occ.Segment(pnt1, pnt2)
    seg2 = occ.Segment(pnt2, pnt3)
    arc1 = occ.ArcOfCircle(pnt3, pnt4, pnt5)
    seg3 = occ.Segment(pnt5, pnt6)

    w = occ.Wire([seg1, seg2, arc1, seg3])
    body = w.Revolve(occ.Axis((0,0,0),occ.Z), 360).Rotate(occ.Axis((0,0,0),occ.X), -90)

    body.edges[0].name = "boundary"

    geo = occ.OCCGeometry(body)

    mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)

    return mesh, geo

mesh, _ = generate_synapse3d(0.05, order_g = 1)

initial_mc_type = [0, 1]
postprocessing_type = [None, 'harmonic', 'harmonic_step']
stabilized_mc = [False, True]

fes_order = 1
T = 0.016
dt0 = 0.00002
dh0 = 0.2

dt = Parameter(dt0)
t = Parameter(0.0)
rhs = CF((0, 0, 0))

params = {'rhs': rhs,
            'domain_name': '.*',
            'clamped_bnd': 'boundary',
            'gamma_stab': 1e-3}
solver = WillmoreSolver(mesh, fes_order=fes_order, dt=dt, t=t, T=T, params = params, verbose = 0)

# Grid-functions for printing purposes
fes2 = VectorH1(mesh, order = 1)
fes3 = H1(mesh, order = 1)

gfu_displacement = GridFunction(fes2)
gfu_mean_curvature = GridFunction(fes2)


for i, ic in enumerate(initial_mc_type):

    for j, stab in enumerate(stabilized_mc):

        for k, pp in enumerate(postprocessing_type):

            # Creating sub-directory for results
            sub_dir = directory + '/' + str(i) + str(j) + str(k)
            os.makedirs(sub_dir, exist_ok=True)

            solver.t.Set(0.0)

            solver.params['initial_mc_type'] = ic
            solver.params['postprocessing_type'] = pp 
            solver.params['stabilized_mc'] = stab

            vtkout_s = VTKOutput(mesh, coefs=[solver.sol_h[0], solver.sol_h[1]],
                                  names=['surface_displacement', 'mean_curvature'], filename=sub_dir + '/results_surf' )
            
            willmore_energy = []
            surface_area = []
            max_y = []
            t_vec = []

            scene = Draw(gfu_mean_curvature, mesh, deformation = gfu_displacement)

            try: 

                out_step = int(int(solver.T/solver.dt.Get())/200)+1

                for count, sol in enumerate(solver()):

                    gfu_displacement.Set(solver.sol_h[0], definedon=mesh.Boundaries(solver.params['domain_name']))
                    gfu_mean_curvature.Set(solver.sol_h[1], definedon=mesh.Boundaries(solver.params['domain_name']))

                    if count%out_step == 0:

                        vtkout_s.Do(time=solver.t.Get(), vb=BND)

                    mesh.SetDeformation(solver.sol_h[0])
                    willmore_energy.append(Integrate(InnerProduct(solver.sol_h[1], solver.sol_h[1]),
                                                        mesh, order = 2*fes_order, VOL_or_BND=BND))
                    surface_area.append(Integrate(1, mesh, order = 2*fes_order, VOL_or_BND=BND))
                    m_y = -np.inf
                    for v in mesh.vertices:
                        aux = np.array(solver.sol_h[0](mesh(*v.point))[1])
                        m_y = np.max([m_y, v.point[1]+aux])
                    max_y.append(m_y)
                    mesh.UnsetDeformation()

                    t_vec.append(solver.t.Get())

                    scene.Redraw()

            except:
                
                pass

            # Printing the results
            
            name = sub_dir + "/results.dat"

            horizontal_labels = ['name-dt', 'willmore_energy', 'surface_area', 'max_y']
            vertical_labels =  [f'{x:.2e}' for x in t_vec]
            output = np.column_stack((vertical_labels, np.column_stack([willmore_energy, surface_area, max_y])))

            df = pd.DataFrame(output, columns=horizontal_labels)
            df.to_csv(name, sep='\t', index=False)

# settings={"camera": {"transformations": [{"type": "rotateX", "angle": 0}]}}