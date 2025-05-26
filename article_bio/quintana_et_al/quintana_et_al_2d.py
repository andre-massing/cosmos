# %%
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *   # Opencascade for geometry modeling
import pandas as pd
import matplotlib.pyplot as plt
import netgen.occ as occ
from cosmos.pdes.pde_willmore_bgn import WillmoreBGN
from cosmos.pdes.pde_willmore_bgn_stab import WillmoreBGNStab
from cosmos.pdes.pde_willmore_dziuk import WillmoreDziuk
from cosmos.pdes.pde_willmore_dziuk_stab import WillmoreDziukStab
from cosmos.solvers.solver_unsteady import UnsteadySolver
from cosmos.solvers.schemes import Steady, BDF1, BDF2
from cosmos.pdes.pde_distance_vol import DistanceVol
from cosmos.pdes.pde_adr_vol_ad import VolADR

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

will_sol = [
    # WillmoreBGN(postprocess = None, mc_autoupdate = True, domain = 'membrane', clamped_bnd = 'membrane_bnd'),
    # WillmoreBGNStab(postprocess = None, mc_autoupdate = True, domain = 'membrane', clamped_bnd = 'membrane_bnd'),
    # WillmoreDziuk(postprocess = None, mc_autoupdate = True, domain = 'membrane', clamped_bnd = 'membrane_bnd'),
    WillmoreDziukStab(postprocess = None, mc_autoupdate = True, domain = 'membrane', clamped_bnd = 'membrane_bnd')
]

for i, willmore in enumerate(will_sol):

    mesh, _ = generate_synapse2d(maxh=0.02)

    t = Parameter(0.0)
    dt = Parameter(1e-3)
    T = 1

    solver = UnsteadySolver(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(willmore, BDF1(conservative=False))
    # solver.AddPDE(willmore, Steady())

    from cosmos.pdes.pde_neohook import NeoHook
    from cosmos.pdes.pde_elastic import Elastic
    bnd_funct = GridFunction(VectorH1(mesh))
    dir_bnd = {'.*': bnd_funct}
    ale_ext = Elastic(lam = 1, mu = 1, rho = 1, steady = True, dir=dir_bnd)
    solver.AddPDE(ale_ext, BDF1())
    solver.ale.SetMeshDeformation(ale_ext.d_h)

    dist = DistanceVol(dirichlet = 'membrane')
    solver.AddPDE(dist, Steady())

    u0 = IfPos(y - 0.6, 1, 0)
    flux_b = {'.*': CF((0, 0))}
    flux_d = {'.*': CF((0, 0))}
    actin = VolADR(u0 = u0, b = - dist.X, d = 0.01, domain = 'inner_space', MP = True, BP = [0, 1e5], Fneu_b = flux_b, neu_d = flux_d)
    solver.AddPDE(actin, BDF1())

    import time
    scene1 = Draw(bnd_funct, mesh, deformation = solver.ale.deformation)
    scene2 = Draw(actin.gfu, mesh, deformation = solver.ale.deformation)
    ns = specialcf.normal(mesh.dim)
    for sol in solver():
        actin.params['b'] = - dist.X
        bnd_funct.Set(willmore.dX_h, definedon=mesh.Boundaries('.*'))
        scene1.Redraw()
        scene2.Redraw()
# %%
