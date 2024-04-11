# %%
from ngsolve import *
import netgen.occ as occ
from netgen.meshing import MeshingStep
from ngsolve.webgui import Draw

def generate_circle(maxh, order_g = 1, center=occ.Pnt(0,0,0), R = 0.5):

    wp = occ.WorkPlane()
    wp.Arc(R, 180)
    wp.Arc(R, 180)
    synapse = wp.Face()
    synapse.edges.name = "membrane"
    synapse.edges.col = (0, 0, 0)
    synapse = synapse.Move((0,-R,0))
    synapse = synapse.Move((center[0], center[1], center[2]))
    synapse.faces.name = "inner_space"

    geo = occ.OCCGeometry(synapse, dim = 2)
    mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)
    return mesh

def generate_sphere(maxh, order_g = 1, center=occ.Pnt(0,0,0), R = 0.5):

    body = occ.Sphere(center, R).faces[0]
    body.faces.name = "membrane"

    geo = occ.OCCGeometry(body)
    mesh = geo.GenerateMesh(maxh=maxh)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)
    return mesh 

def generate_ball(maxh, order_g = 1, center=occ.Pnt(0,0,0), R = 0.5):

    body = occ.Sphere(center, R)
    body.faces.name = "membrane"
    body.mat("inner_space")

    geo = occ.OCCGeometry(body)
    mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)
    return mesh

def generate_torus(maxh, order_g = 1, center=occ.Pnt(0,0,0), R = 1.0, r = 0.4):

    pnt1 = occ.Pnt(R-r + center[0], 0 + center[1], 0 + center[2])
    pnt2 = occ.Pnt(R + center[0], 0 + center[1], r + center[2])
    pnt3 = occ.Pnt(R+r + center[0], 0 + center[1], 0 + center[2])
    pnt4 = occ.Pnt(R + center[0], 0 + center[1], -r + center[2])

    arc1 = occ.ArcOfCircle(pnt1, pnt2, pnt3)
    arc2 = occ.ArcOfCircle(pnt3, pnt4, pnt1)

    w = occ.Wire([arc1, arc2])
    f = occ.Face(w)
    body = f.Revolve(occ.Axis((0,0,0),occ.Z), 360)

    geo = occ.OCCGeometry(body)
    mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)
    return mesh

def generate_cube(maxh, order_g = 1, center=occ.Pnt(0,0,0), R = 0.5):

    body = occ.Box(occ.Pnt(-R/2,-R/2,-R/2), occ.Pnt(R/2, R/2, R/2))
    body = body.Move((center[0], center[1], center[2]))

    geo = occ.OCCGeometry(body)
    mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)
    return mesh

def generate_cube_g5(maxh, order_g = 1, center=occ.Pnt(0,0,0), R = 1.0):

    body = occ.Box(occ.Pnt(-R/2,-R/2,-R/2), occ.Pnt(R/2, R/2, R/2))
    body = body - occ.Box(occ.Pnt(-R/6,-R/6,-R), occ.Pnt(R/6, R/6, R))
    body = body - occ.Box(occ.Pnt(-R/6,-R,-R/6), occ.Pnt(R/4, R, R/6))
    body = body - occ.Box(occ.Pnt(-R,-R/6,-R/6), occ.Pnt(R, R/6, R/6))

    body = body.Move((center[0], center[1], center[2]))

    geo = occ.OCCGeometry(body)
    mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)
    return mesh 

def generate_synapse2d(maxh, order_g = 1, external = False):
    wp = occ.WorkPlane()
    wp.Rotate(90).Line(0.1).Rotate(-90)
    wp.Line(0.2).Line(0.15).Arc(0.1, 90)
    wp.Line(0.3).Arc(0.1, 90)
    wp.Arc(0.1, -180).Line(0.3).Arc(0.1,-180)
    wp.Arc(0.1, 90).Line(0.3).Arc(0.1, 90)
    wp.Line(0.15).Line(0.2)
    wp.Rotate(-90).Line(0.1).Rotate(-90)
    wp.Line(1.0)
    wp.Reverse()
    synapse = wp.Face()

    synapse.edges.name = "membrane"
    synapse.edges.col = (0,0,0)
    synapse.edges[occ.X<0.2].name = "membrane_bnd"
    synapse.edges[occ.X>0.8].name = "membrane_bnd"
    synapse.edges.Min(occ.Y).name = "inner_bnd"
    synapse.edges.Min(occ.X).name = "inner_bnd"
    synapse.edges.Max(occ.X).name = "inner_bnd"
    synapse.faces.name = "inner_space"

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
    return mesh

def generate_synapse3d(maxh, order_g = 1, external = False):
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
    return mesh

if __name__ == "__main__":
    order_g = 3

    circle = generate_circle(maxh=0.05, order_g = order_g)
    Draw(circle)

    sphere = generate_sphere(maxh=0.05, order_g = order_g)
    Draw(sphere, clipping={"x": 0, "y": 1, "z": 0, "dist": 0.0})

    ball = generate_ball(maxh=0.05, order_g = order_g)
    Draw(ball, clipping={"x": 0, "y": 1, "z": 0, "dist": 0.0})

    cube = generate_cube(maxh=0.05, order_g = order_g)
    Draw(cube, clipping={"x": 0, "y": 1, "z": 0, "dist": 0.0})

    cube_g5 = generate_cube_g5(maxh=0.05, order_g = order_g)
    Draw(cube_g5, clipping={"x": 0, "y": 1, "z": 0, "dist": 0.0})

    torus = generate_torus(maxh=0.1, order_g = order_g, R = sqrt(2), r = 1.0)
    Draw(torus, clipping={"x": 0, "y": 1, "z": 0, "dist": 0.0})

    synapse2d = generate_synapse2d(maxh=0.05, order_g = order_g)
    Draw(synapse2d)

    synapse3d = generate_synapse3d(maxh=0.1, order_g = order_g)
    Draw(synapse3d, draw_surf = False, clipping={"x": 0, "y": 1, "z": 0, "dist": 0.0})

__all__ = [
    'generate_circle',
    'generate_sphere',
    'generate_ball',
    'generate_cube',
    'generate_cube_g5',
    'generate_torus',
    'generate_synapse2d',
    'generate_synapse3d'
]
# %%
