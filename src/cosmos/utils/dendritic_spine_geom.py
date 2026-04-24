import netgen.occ as occ
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.webgui import Draw as DrawGeo

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
    synapse.edges.maxh = maxh/4

    for i in range(4):
        synapse.edges[i+1].name = "membrane"
    synapse.vertices[1].name = 'membrane_bnd'
    synapse.vertices[2].name = 'membrane_bnd'

    total = synapse.Revolve(occ.Axis((0,0,0),occ.Y),360).Rotate(occ.Axis((0,0,0),occ.X), 90)

    geo = occ.OCCGeometry(total, dim = 3)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh, uselocalh=True,
        grading = 0.1
        ))
    mesh.Curve(order_g)

    return mesh