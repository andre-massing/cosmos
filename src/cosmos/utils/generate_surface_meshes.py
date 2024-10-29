# %%
# ----------------------------------------------
# Geometries for Testing
# ----------------------------------------------
# This section of the code is dedicated to creating a series of meshes
# to be used for testing purposes, especially convergence studies.
# ----------------------------------------------

from ngsolve import *
import netgen.csg as csg
import netgen.occ as occ
from netgen.meshing import MeshingStep
from ngsolve.webgui import Draw as DrawMesh
from netgen import stl

__all__ = [
    'generate_plane',
    'generate_circle',
    'generate_sphere',
    'generate_half_sphere',
    'generate_ball',
    'generate_cylinder',
    'generate_torus',
    'generate_half_torus', 
    'generate_box',
    'import_stl_mesh'
]

'''
Auxiliary function to allowing to mesh, switching between volume and surface mesh
'''
def Meshing(geo, maxh, order_g, vol_or_bnd):

    if vol_or_bnd == 'BND':
        mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE)
    elif vol_or_bnd == 'VOL':
        mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)

    return mesh

'''
Plane
NOTE: This is a plane embedded in 3D, so its number of elements is 0
'''
def generate_plane(maxh=0.1, order_g = 1, a =1.0, b=2.0, bnd_name = "boundary", geo_only = False):
    # order_g is maintained just for compatibility, but it does not improve the 
    # description of the plane. a,b are the side length. The boundary of the plane
    # has been named to facilitate handling of boundary conditions

    wp = occ.WorkPlane()
    wp.Rectangle(a,b)
    face = wp.Face()

    face.edges.name = bnd_name

    geo = occ.OCCGeometry(face)

    if geo_only:
        return geo
    else:
        mesh = Meshing(geo, maxh, order_g, 'VOL')	
        return mesh , geo

'''
2D circle
'''
# Later add-on, switch from 2D to 2D immersed in 3D
def generate_circle(maxh=0.1, order_g = 1, center=occ.Pnt(0,0,0), R = 1.0, bnd_name = "boundary", geo_only = False):

    wp = occ.WorkPlane()
    wp.Arc(R, 180)
    wp.Arc(R, 180)
    synapse = wp.Face()
    synapse.edges.name = bnd_name
    synapse.edges.col = (0, 0, 0)
    synapse = synapse.Move((0,-R,0))
    synapse = synapse.Move((center[0], center[1], center[2]))

    geo = occ.OCCGeometry(synapse, dim = 2)

    if geo_only:
        return geo
    else:
        mesh = Meshing(geo, maxh, order_g, 'VOL')
        return mesh, geo

'''
Sphere
NOTE: no vol_or_bnd option here, since explicitly requesting for a surface
'''
def generate_sphere(maxh=0.1, order_g = 1, center=csg.Pnt(0,0,0), R = 1.0, geo_only = False):
    
    geo          = csg.CSGeometry()
    sphere       = csg.Sphere(center, R)
    geo.Add(sphere)
    
    if geo_only:
        return geo
    else:
        mesh = Mesh(geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE))
        mesh.Curve(order_g)
        return mesh, geo

'''
Half sphere
NOTE: no vol_or_bnd option here, since explicitly requesting for a surface
'''
def generate_half_sphere(maxh=0.1, order_g = 1, center=csg.Pnt(0,0,0), R = 1.0,  bnd_name = "bottom", geo_only = False):
    
    geo          = csg.CSGeometry()
    sphere       = csg.Sphere(center, R)
    bot          = csg.Plane(center, csg.Vec(0,0,-1))
    finitesphere = sphere * bot

    geo.AddSurface(sphere, finitesphere)
    geo.NameEdge(sphere,bot, bnd_name)

    if geo_only:
        return geo
    else:
        mesh = Mesh(geo.GenerateMesh(maxh=maxh))
        mesh.Curve(order_g)
        return mesh, geo

'''
Ball
'''
def generate_ball(maxh = 0.1, order_g = 1, center=occ.Pnt(0,0,0), R = 1.0, vol_or_bnd = 'VOL', bnd_name = "boundary", geo_only = False):

    body = occ.Sphere(center, R)
    body.faces.name = bnd_name
    body = body.Move((center[0], center[1], center[2]))

    geo = occ.OCCGeometry(body)

    if geo_only:
        return geo
    else:
        mesh = Meshing(geo, maxh, order_g, vol_or_bnd)
        return mesh, geo


'''
Cylinder
NOTE: no vol_or_bnd option here, since explicitly requesting for a surface
'''
def generate_cylinder(maxh = 0.1, order_g=1, bnd_name = "boundary", geo_only = False):
    geo       = csg.CSGeometry()
    cyl       = csg.Cylinder(csg.Pnt(0,0,0), csg.Pnt(1,0,0), 1)
    right     = csg.Plane(csg.Pnt(2,0,0), csg.Vec(1,0,0))
    left      = csg.Plane(csg.Pnt(-2,0,0), csg.Vec(-1,0,0))
    finitecyl = cyl * left * right

    geo.AddSurface(cyl, finitecyl)
    geo.NameEdge(cyl, left, bnd_name)
    geo.NameEdge(cyl, right, bnd_name)

    if geo_only:
        return geo
    else:
        mesh = Meshing(geo, maxh, order_g, 'BND')
        return mesh, geo

'''
Torus
'''
# TODO: Other possibilities to generate
def generate_torus(maxh = 0.1, order_g = 1, center=occ.Pnt(0,0,0), R = 1.0, r = 0.4, vol_or_bnd = 'VOL', geo_only = False):

    spline = csg.SplineCurve2d() # create a 2d spline
    eps = r*1e-2

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

    rev = csg.Revolution ( csg.Pnt(0,0,-1), csg.Pnt(0,0,1), spline)
    geo = csg.CSGeometry()
    geo.Add (rev.col([0,0,1]))

    if geo_only:
        return geo
    else:
        mesh = Meshing(geo, maxh, order_g, vol_or_bnd)
        return mesh, geo

def generate_half_torus(maxh = 0.1, order_g = 1, R = 1.0, r = 0.4, vol_or_bnd = "VOL", bnd_name = "bottom", geo_only = False):

    pnt1 = occ.Pnt(R-r, 0, 0 )
    pnt2 = occ.Pnt(R, 0, r )
    pnt3 = occ.Pnt(R+r , 0, 0 )
    pnt4 = occ.Pnt(R , 0 , -r)

    arc1 = occ.ArcOfCircle(pnt1, pnt2, pnt3)
    arc2 = occ.ArcOfCircle(pnt3, pnt4, pnt1)

    w = occ.Wire([arc1, arc2])
    w.edges.name = "membrane"
    body = w.Revolve(occ.Axis((0,0,0),occ.Z), 180).Rotate(occ.Axis((0,0,0),occ.X), 90)

    body.edges[occ.Z<maxh/2].name = bnd_name

    geo = occ.OCCGeometry(body)

    if geo_only:
        return geo
    else:
        mesh = Meshing(geo, maxh, order_g, vol_or_bnd)
        return mesh, geo

def generate_box(maxh = 0.1, order_g = 1, center=occ.Pnt(0,0,0), a = 1, b = 1, c = 1, vol_or_bnd = 'VOL', bnd_name = "boundary", geo_only = False):

    body = occ.Box(occ.Pnt(-a/2,-b/2,-c/2), occ.Pnt(a/2, b/2, c/2))
    body = body.Move((center[0], center[1], center[2]))
    body.faces.name = bnd_name

    geo= occ.OCCGeometry(body)

    if geo_only:
        return geo
    else:
        mesh = Meshing(geo, maxh, order_g, vol_or_bnd)
        return mesh, geo

def import_stl_mesh(fname, maxh = 0.1, order_g = 1, geo_only = False):

    geo = stl.STLGeometry(fname)

    if geo_only:
        return geo
    else:
        ngmesh = geo.GenerateMesh(perfstepsend=MeshingStep.MESHSURFACE,
                            quad_dominated=False,
                            maxh=maxh,
                            grading=0.7,
                            optimize2d = "smsmsmSmSmSm",
                            yangle=50,
                            contyangle=50,
                            edgecornerangle=30)
        mesh = Mesh(ngmesh)
        mesh.Curve(order_g)
        return mesh, geo

# def generate_n_torus_mesh(maxh, order_g = 1, char_len=1, n=2):
#     # char_len is the thickness of the thorus and coincides with the inner radius of 
#     # the holes, the width is automatically deduced in order to fit the n holes 
#     # disposed tangential to the outer circumference in a way that they don't
#     # intersect eachother

#     char_len = 1
#     if n<2:
#         raise RuntimeError("At least 2 holes must be made, otherwise opt for the torus geometry")
#     R = 3*char_len/4
#     width = R/sin(pi/n) + R + char_len/2

#     pnt1 = occ.Pnt(0, 0, char_len/2)
#     pnt2 = occ.Pnt(width-char_len/2, 0, char_len/2)
#     pnt3 = occ.Pnt(width, 0, 0)
#     pnt4 = occ.Pnt(width-char_len/2, 0, -char_len/2)
#     pnt5 = occ.Pnt(0, 0, -char_len/2)

#     segm1 = occ.Segment(pnt1, pnt2)
#     arc= occ.ArcOfCircle(pnt2, pnt3, pnt4)
#     segm2 = occ.Segment(pnt4, pnt5)
#     segm3 = occ.Segment(pnt5, pnt1)

#     w = occ.Wire([segm1, arc, segm2, segm3])
#     f = occ.Face(w)
#     body = f.Revolve(occ.Axis((0,0,0),occ.Z), 360)

#     pnt1 = occ.Pnt(0, 0, char_len/2)
#     pnt2 = occ.Pnt(3*char_len/4, 0, char_len/2)
#     pnt3 = occ.Pnt(char_len/4, 0, 0)
#     pnt4 = occ.Pnt(3*char_len/4, 0, -char_len/2)
#     pnt5 = occ.Pnt(0, 0, -char_len/2)

#     segm1 = occ.Segment(pnt1, pnt2)
#     arc= occ.ArcOfCircle(pnt2, pnt3, pnt4)
#     segm2 = occ.Segment(pnt4, pnt5)
#     segm3 = occ.Segment(pnt5, pnt1)

#     w = occ.Wire([segm1, arc, segm2, segm3])
#     f = occ.Face(w)
#     body2 = f.Revolve(occ.Axis((0,0,0),occ.Z), 360)
#     body2 = body2.Move((R/sin(pi/n), 0, 0))

#     for i in range(n):
#         body = body-body2.Rotate(occ.Axis((0,0,0), occ.Z), 360/n*i)

#     geo = occ.OCCGeometry(body)
#     mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE)
#     mesh = Mesh(mesh)
#     mesh.Curve(order_g)	
#     return mesh


# %%
if __name__ == "__main__":

    for funct in __all__[:-1]:

        mesh, _ = globals()[funct](geo_only = False)
        print(funct, ', nel=', mesh.ne)
        DrawMesh(mesh)

    '''
    More advanced examples for stress testing the algorithms
    '''
    rel_path = "../../../data/geometries/"

    mesh, _ = import_stl_mesh(maxh = 2, fname = rel_path + 'bunny.stl', geo_only = False)
    DrawMesh(mesh)

    mesh, _ = import_stl_mesh(maxh = 0.05, fname = rel_path + 'spot_cow.stl', geo_only = False)
    DrawMesh(mesh)

    # mesh, _ = import_stl_mesh(maxh = 10, fname = rel_path + 'brain.stl', geo_only = False)
    # DrawMesh(mesh)


# %%
