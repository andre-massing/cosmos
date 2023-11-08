# %%
from ngsolve import *
import netgen.csg as csg
import netgen.occ as occ
from netgen.meshing import MeshingStep
from ngsolve.webgui import Draw
from netgen import stl

def generate_sphere_mesh(maxh, order_g = 1, center=csg.Pnt(0,0,0), r = 1.0):
    geo          = csg.CSGeometry()
    sphere       = csg.Sphere(center, r)
    geo.Add(sphere)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE))
    mesh.Curve(order_g)
    return mesh

def generate_cylinder_mesh(maxh, order_g=1):
    geo       = csg.CSGeometry()
    cyl       = csg.Cylinder(csg.Pnt(0,0,0), csg.Pnt(1,0,0), 1)
    right     = csg.Plane(csg.Pnt(2,0,0), csg.Vec(1,0,0))
    left      = csg.Plane(csg.Pnt(-2,0,0), csg.Vec(-1,0,0))
    finitecyl = cyl * left * right

    geo.AddSurface(cyl, finitecyl.bc("surface"))
    geo.NameEdge(cyl, left, "left")
    geo.NameEdge(cyl, right, "right")

    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(order_g)
    return mesh


# TODO: Other possibilities to generate
def generate_torus_mesh(maxh, order_g = 1, center=csg.Pnt(0,0,0), R = 1.0, r = 0.4):

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

def generate_n_torus_mesh(maxh, order_g = 1, char_len=1, n=2):
    # char_len is the thickness of the thorus and coincides with the inner radius of 
    # the holes, the width is automatically deduced in order to fit the n holes 
    # disposed tangential to the outer circumference in a way that they don't
    # intersect eachother

    char_len = 1
    if n<2:
        raise RuntimeError("At least 2 holes must be made, otherwise opt for the torus geometry")
    R = 3*char_len/4
    width = R/sin(pi/n) + R + char_len/2

    pnt1 = occ.Pnt(0, 0, char_len/2)
    pnt2 = occ.Pnt(width-char_len/2, 0, char_len/2)
    pnt3 = occ.Pnt(width, 0, 0)
    pnt4 = occ.Pnt(width-char_len/2, 0, -char_len/2)
    pnt5 = occ.Pnt(0, 0, -char_len/2)

    segm1 = occ.Segment(pnt1, pnt2)
    arc= occ.ArcOfCircle(pnt2, pnt3, pnt4)
    segm2 = occ.Segment(pnt4, pnt5)
    segm3 = occ.Segment(pnt5, pnt1)

    w = occ.Wire([segm1, arc, segm2, segm3])
    f = occ.Face(w)
    body = f.Revolve(occ.Axis((0,0,0),occ.Z), 360)

    pnt1 = occ.Pnt(0, 0, char_len/2)
    pnt2 = occ.Pnt(3*char_len/4, 0, char_len/2)
    pnt3 = occ.Pnt(char_len/4, 0, 0)
    pnt4 = occ.Pnt(3*char_len/4, 0, -char_len/2)
    pnt5 = occ.Pnt(0, 0, -char_len/2)

    segm1 = occ.Segment(pnt1, pnt2)
    arc= occ.ArcOfCircle(pnt2, pnt3, pnt4)
    segm2 = occ.Segment(pnt4, pnt5)
    segm3 = occ.Segment(pnt5, pnt1)

    w = occ.Wire([segm1, arc, segm2, segm3])
    f = occ.Face(w)
    body2 = f.Revolve(occ.Axis((0,0,0),occ.Z), 360)
    body2 = body2.Move((R/sin(pi/n), 0, 0))

    for i in range(n):
        body = body-body2.Rotate(occ.Axis((0,0,0), occ.Z), 360/n*i)

    geo = occ.OCCGeometry(body)
    mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)	
    return mesh

def generate_plane(maxh, order_g = 1, a =1.0, b=2.0):
    # order_g is mainteined just for compatibility, but it does not improve the 
    # description of the plane. a,b are the side length. The boundary of the plane
    # has been named to facilitate handling of boundary conditions

    wp = occ.WorkPlane()
    wp.Rectangle(a,b)
    face = wp.Face()

    face.edges.name = "boundary"

    geo = occ.OCCGeometry(face)

    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(order_g);	
    return mesh 


def import_stl_mesh(fname, maxh):
    geo = stl.STLGeometry(fname)
    ngmesh = geo.GenerateMesh(perfstepsend=MeshingStep.MESHSURFACE,
                          quad=False,
                          maxh=maxh,
                          grading=0.7,
                          optimize2d = "smsmsmSmSmSm",
                          yangle=50,
                          contyangle=50,
                          edgecornerangle=30)
    mesh = Mesh(ngmesh)
    # mesh.Curve(4)
    return mesh, geo

def generate_cylinder():
    pass

# %%
if __name__ == "__main__":
    maxh = 0.5
    order_g = 2

    sphere_mesh = generate_sphere_mesh(maxh=maxh, order_g=order_g)
    Draw(sphere_mesh)

    torus_mesh = generate_torus_mesh(maxh=maxh, order_g = order_g)
    Draw(torus_mesh)

    #%% Test bunny
    fname = "../data/geometries/bunny.stl"
    bunny_mesh = import_bunny_mesh(fname)

    #%% Test bunny
    bunny_mesh.Curve(1)
    Draw(bunny_mesh)

    bunny_mesh.Curve(4)
    Draw(bunny_mesh)