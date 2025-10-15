import netgen.occ as occ
from ngsolve import *

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
    # synapse.faces.name = "inner_space"
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