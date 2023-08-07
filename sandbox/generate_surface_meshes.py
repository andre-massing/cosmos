# %%
from ngsolve import *
from netgen.csg import *
from netgen.meshing import MeshingStep
from ngsolve.webgui import Draw

def generate_sphere_mesh(maxh, order_g = 1, center=Pnt(0,0,0), r = 1.0):
    geo          = CSGeometry()
    sphere       = Sphere(center, r)
    geo.Add(sphere)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE))
    mesh.Curve(order_g)
    return mesh

# TODO: Other possibilities to generate
def generate_torus_mesh(maxh, order_g = 1, center=Pnt(0,0,0), R = 1.0, r = 0.4):
    spline = SplineCurve2d()
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
        spline.AddPoint (*pnt)

    for seg in segs:
        spline.AddSegment (*seg)

    rev = Revolution(Pnt(0,0,-1), Pnt(0,0,1), spline)
    geo = CSGeometry()
    geo.Add(rev.col([1,0,0]))

    mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)	
    return mesh 

# %%
# if __name__ == "__main__":
maxh = 0.5
order_g = 2

# sphere_mesh = generate_sphere_mesh(maxh=maxh, order_g=order_g)
# Draw(sphere_mesh)

torus_mesh = generate_torus_mesh(maxh=maxh, order_g = order_g)
Draw(torus_mesh)


# %%
