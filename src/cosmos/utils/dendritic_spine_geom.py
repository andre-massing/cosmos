import netgen.occ as occ
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.webgui import Draw as DrawGeo
from netgen.meshing import MeshingParameters

def generate_synapse2d(maxh, order_g = 1):

    wp = occ.WorkPlane()
    wp.Line(0.05).Rotate(90)
    wp.Line(0.3).Arc(0.1, -90)
    wp.Arc(0.1, 180).Line(0.3).Arc(0.1,180)
    wp.Arc(0.1, -90).Line(0.3)
    wp.Rotate(90).Line(0.05)

    synapse = wp.Face()

    synapse.edges.maxh = maxh/4

    for i in range(7):
        synapse.edges[i+1].name = "membrane"
    synapse.vertices[1].name = 'membrane_bnd'
    synapse.vertices[2].name = 'membrane_bnd'
    synapse.vertices[15].name = 'membrane_bnd'
    synapse.vertices[16].name = 'membrane_bnd'

    total = synapse

    geo = occ.OCCGeometry(total, dim = 2)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh, uselocalh=True,
        optsteps2d=3 ))
    mesh.Curve(order_g)

    return mesh

def generate_synapse3d(maxh, order_g = 1):

    wp = occ.WorkPlane()
    wp.Line(0.05).Rotate(90)
    wp.Line(0.3).Arc(0.1, -90)
    wp.Arc(0.1, 180).Line(0.15).Rotate(90)
    wp.Line(0.6)

    synapse = wp.Face()

    for i in range(4):
        synapse.edges[i+1].name = "membrane"
    synapse.vertices[1].name = 'membrane_bnd'
    synapse.vertices[2].name = 'membrane_bnd'
    synapse.edges.maxh = maxh

    total = synapse.Revolve(occ.Axis((0,0,0),occ.Y),360).Rotate(occ.Axis((0,0,0),occ.X), 90)

    mp = MeshingParameters(
        grading=0.1,
        optsteps3d=5,
        optimize3d="cmdmstmcmdmstm",
        optsteps2d=5,
        closeedgefac=2,
        quality = 4
    )

    geo = occ.OCCGeometry(total, dim = 3)
    ngmesh = geo.GenerateMesh(mp = mp)
    mesh = Mesh(ngmesh)
    mesh.Curve(order_g)

    return mesh