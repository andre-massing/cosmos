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
    spline = csg.SplineCurve2d()
    R = 1
    r = 0.4
    eps = r*1e-8

    # define the control points
    pnts = [ (0,R-r), (-r+eps,R-r+eps), (-r,R),
            (-r+eps,R+r-eps), (0,R+r), (r-eps,R+r-eps), (r,R), (r-eps,R-r+eps) ]
    # define the splines using the control points
    segs = [ (0,1,2), (2,3,4), (4,5,6), (6,7,0) ]

    # add the points and segments to the spline
    for pnt in pnts:
        spline.AddPoint(*pnt)

    for seg in segs:
        spline.AddSegment(*seg)

    rev = csg.Revolution(csg.Pnt(0,0,-1), csg.Pnt(0,0,1), spline)
    geo = csg.CSGeometry()
    geo.Add(rev.col([1,0,0]))

    mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)	
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
    return mesh

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