# %%
# ----------------------------------------------
# Geometries for Testing
# ----------------------------------------------
# This section of the code is dedicated to creating a series of meshes
# to be used for testing purposes, especially convergence studies.
# ----------------------------------------------
#
# Library of NGSolve mesh-generation functions for the standard test geometries
# used throughout tests/ and reproduction/. Naming convention:
#   - generate_boundary_*: builds a pure surface mesh (mesh.ne == 0, no volume
#     elements) for boundary/surface PDE models.
#   - generate_volume_*: builds a bulk (volume) mesh.
# Common parameters across most functions: maxh = target mesh element size,
# order_g = curved-element (geometric) order, bnd_name/bbnd_name = name given to
# the boundary/edge (bboundary) region so PDE models can select it by name.

from ngsolve import *
import netgen.csg as csg
import netgen.occ as occ
from netgen.meshing import MeshingStep
from ngsolve.webgui import Draw as DrawMesh
from netgen import stl
from netgen.meshing import Element0D, Element1D, Element2D, MeshPoint, Pnt
from netgen.meshing import Mesh as NetGenMesh
import numpy as np

'''
Auxiliary function to allowing to mesh, switching between volume and surface mesh
'''
def Meshing(geo, maxh, order_g, vol_or_bnd):
    """Generate and curve a mesh from a netgen geometry `geo`.

    Args:
        geo: A netgen geometry object (csg.CSGeometry or occ.OCCGeometry) exposing
            `GenerateMesh`.
        maxh: Target maximum mesh element size.
        order_g: Curved-element order to apply via `mesh.Curve(order_g)`.
        vol_or_bnd: 'BND' to stop meshing at MeshingStep.MESHSURFACE (producing a
            pure surface mesh with no volume elements), 'VOL' to mesh the full volume.

    Returns:
        The curved ngsolve.Mesh.
    """

    if vol_or_bnd == 'BND':
        mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE)
    elif vol_or_bnd == 'VOL':
        mesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3)
    mesh = Mesh(mesh)
    mesh.Curve(order_g)

    return mesh

'''
Arc
'''
def generate_boundary_arc(r=1, N=20, bbnd_name = "bboundary"):
    """Build a 1D (curve) mesh of a semicircular arc of radius `r`, manually
    assembled from `N` linear segments (split into two "default" material
    regions, left/right half) with its two endpoints tagged `bbnd_name`.
    """

    ngmesh = NetGenMesh(dim=2)
    pids = []
    radius = r
    n_segments = N  # Number of segments for smoothness
    theta_vals = np.linspace(0, np.pi, n_segments + 1)  # Angles from 0 to π
    # Add points to the mesh.
    for theta in theta_vals:
        pids.append(ngmesh.Add(MeshPoint(Pnt(radius * np.cos(theta), radius * np.sin(theta), 0))))
    # Left half of the domain is material 1, right half material 2.

    idx_doml = ngmesh.AddRegion("default", dim=1)
    idx_domr = ngmesh.AddRegion("default", dim=1)
        
    for i in range(n_segments):
        ngmesh.Add(Element1D([pids[i], pids[i + 1]], index=idx_doml if i < 10 else idx_domr))

    # Add BC to the mesh.

    idx_l = ngmesh.AddRegion(bbnd_name, dim=0)
    idx_r = ngmesh.AddRegion(bbnd_name, dim=0)
    ngmesh.Add(Element0D(pids[0], index=idx_l))  
    ngmesh.Add(Element0D(pids[n_segments], index=idx_r))
    mesh = Mesh(ngmesh)
    
    return mesh

'''
Plane
NOTE: This is a plane embedded in 3D, so its number of elements is 0
'''
def generate_boundary_plane(maxh=0.1, order_g = 1, a =1.0, b=2.0, bbnd_name = "bboundary"):
    """Build a flat rectangular a x b surface mesh embedded in 3D (mesh.ne == 0).

    order_g is kept only for interface compatibility with the other
    generate_boundary_* functions; it has no effect on a flat plane.
    """
    # order_g is maintained just for compatibility, but it does not improve the
    # description of the plane. a,b are the side length. The boundary of the plane
    # has been named to facilitate handling of boundary conditions


    face = occ.WorkPlane(occ.Axes((0,0,0), n=occ.Z, h=occ.X)).Rectangle(a,b).Face()
    face.edges.name = bbnd_name
    geo = occ.OCCGeometry(face)
    mesh = Meshing(geo, maxh, order_g, 'VOL')
        
    return mesh

'''
Circle Embedded in 3D
NOTE: This is a circle embedded in 3D, so its number of elements is 0
'''
def generate_boundary_circle(maxh=0.1, order_g = 1, R = 1.0, bbnd_name = "bboundary"):
    """Build a flat circular disk boundary of radius `R` embedded in 3D
    (mesh.ne == 0); its edge is named `bbnd_name`.
    """
    # order_g is maintained just for compatibility, but it does not improve the
    # description of the plane. a,b are the side length. The boundary of the plane
    # has been named to facilitate handling of boundary conditions


    face = occ.WorkPlane(occ.Axes((0,0,0), n=occ.Z, h=occ.X)).Circle(0, 0, R).Face()
    face.edges.name = bbnd_name
    geo = occ.OCCGeometry(face)
    mesh = Meshing(geo, maxh, order_g, 'VOL')
        
    return mesh

'''
2D circle
'''
def generate_volume_circle(maxh=0.1, order_g = 1, center=occ.Pnt(0,0,0), R = 1.0, bnd_name = "boundary"):
    """Build a filled 2D disk (bulk) mesh of radius `R` centered at `center`;
    its boundary edge is named `bnd_name`.
    """

    face = occ.WorkPlane(occ.Axes((0,0,0), n=occ.Z, h=occ.X)).Circle(0, 0, R).Face()
    face.edges.name = bnd_name
    face = face.Move((center[0], center[1], center[2]))
    geo = occ.OCCGeometry(face, dim = 2)

    mesh = Meshing(geo, maxh, order_g, 'VOL')
    return mesh

'''
Sphere
'''
def generate_boundary_sphere(maxh=0.1, order_g = 1, center=csg.Pnt(0,0,0), R = 1.0):
    """Build a closed spherical surface mesh (mesh.ne == 0) of radius `R`,
    centered at `center`, using a CSG sphere primitive.
    """

    geo          = csg.CSGeometry()
    sphere       = csg.Sphere(center, R)
    geo.Add(sphere)

    mesh = Meshing(geo, maxh, order_g, 'BND')
    mesh.Curve(order_g)
    return mesh
    
'''
Ellipse
'''  
def generate_boundary_ellipse(maxh = 0.1, order_g = 1, a=1, b=1, c=1):
    """Build a closed ellipsoidal surface mesh (mesh.ne == 0) with semi-axes
    `a`, `b`, `c`.
    """

    ell = occ.Ellipsoid(occ.Axes((0,0,0),occ.X,occ.Y),a,b,c).faces[0]
    geo = occ.OCCGeometry(ell, dim = 2)
    mesh = Meshing(geo, maxh, order_g, 'BND')
    # Unused/dead code: random perturbation of interior mesh points, projected
    # back onto the ellipsoid, left commented out below (not executed).
    # fixpoint = []
    # for seg in mesh.ngmesh.Elements1D():
    #     fixpoint += seg.vertices
    # fixpoint = set(fixpoint)
    # par = 0.03*maxh if maxh > 0.2 else 0.08*maxh
    # for i,p in enumerate(mesh.ngmesh.Points()):
    #     if i + 1 in fixpoint: continue
    #     x1,x2,x3 = p[0],p[1],p[2]
    #     x1 += np.random.uniform(-par,par)
    #     x2 += np.random.uniform(-par,par)
    #     x3 += np.random.uniform(-par,par)
    #     radius = sqrt(x1**2/a**2+x2**2/b**2+x3**2/c**2)
    #     p[0] = x1/radius
    #     p[1] = x2/radius
    #     p[2] = x3/radius

    return mesh
    
'''
Sigar
'''
def generate_boundary_sigar(maxh=0.1, order_g = 1, center=csg.Pnt(0,0,0), r = 1.0, h= 2):
    """Build a capsule/stadium-shaped ("sigar") closed surface mesh (mesh.ne == 0):
    a cylinder of radius `r` and height `h`, capped by a hemisphere of the same
    radius at each end.
    """

    cyl = occ.Cylinder((0,-h/2,0), occ.Y, r=r, h=h)
    sphere1 = occ.Sphere( (0,h/2,0), r)
    sphere2 = occ.Sphere( (0,-h/2,0), r)
    fused = cyl + sphere1 + sphere2

    geo = occ.OCCGeometry(fused)
    mesh = Meshing(geo, maxh, order_g, 'BND')
    mesh.Curve(order_g)
    
    return mesh

'''
Half sphere
'''
def generate_boundary_half_sphere(maxh=0.1, order_g = 1, center=csg.Pnt(0,0,0), R = 1.0,  bbnd_name = "bboundary"):
    """Build a hemispherical surface mesh (mesh.ne == 0) of radius `R`, obtained
    by intersecting a sphere with the upper half-space; the equatorial rim
    (where the sphere meets the cutting plane) is named `bbnd_name`.
    """

    geo          = csg.CSGeometry()
    sphere       = csg.Sphere(center, R)
    bot          = csg.Plane(center, csg.Vec(0,0,-1))
    finitesphere = sphere * bot

    geo.AddSurface(sphere, finitesphere)
    geo.NameEdge(sphere,bot, bbnd_name)

    mesh = Meshing(geo, maxh, order_g, 'BND')
    mesh.Curve(order_g)
    
    return mesh

'''
Ball
'''
def generate_volume_ball(maxh = 0.1, order_g = 1, center=occ.Pnt(0,0,0), R = 1.0, bnd_name = "boundary"):
    """Build a filled 3D ball (bulk) mesh of radius `R`, centered at `center`;
    its bounding sphere surface is named `bnd_name`.
    """

    body = occ.Sphere(center, R)
    body.faces.name = bnd_name
    body = body.Move((center[0], center[1], center[2]))

    geo = occ.OCCGeometry(body)

    mesh = Meshing(geo, maxh, order_g, 'VOL')
    return mesh


'''
Cylinder
'''
def generate_boundary_cylinder(maxh = 0.1, R=1, order_g=1, bbnd_name = "bboundary"):
    """Build an open (capless) cylindrical surface mesh (mesh.ne == 0) of radius
    `R`, spanning z in [-1, 1]; both circular rims are named `bbnd_name`.
    """
    geo       = csg.CSGeometry()
    cyl       = csg.Cylinder(csg.Pnt(0,0,0), csg.Pnt(0,0,1), R)
    right     = csg.Plane(csg.Pnt(0,0,1), csg.Vec(0,0,1))
    left      = csg.Plane(csg.Pnt(0,0,-1), csg.Vec(0,0,-1))
    finitecyl = cyl * left * right

    geo.AddSurface(cyl, finitecyl)
    geo.NameEdge(cyl, left, bbnd_name)
    geo.NameEdge(cyl, right, bbnd_name)

    mesh = Meshing(geo, maxh, order_g, 'BND')
    return mesh

'''
Torus
'''
# TODO: Other possibilities to generate
def generate_boundary_torus(maxh = 0.1, order_g = 1, center=occ.Pnt(0,0,0), R = 1.0, r = 0.4):
    """Build a closed toroidal surface mesh (mesh.ne == 0) with major radius `R`
    and minor (tube) radius `r`, obtained by revolving a circular cross-section
    (built here as two OCC arcs) 360 degrees about the z-axis.
    """

    # Unused/dead code: an alternative torus construction via a revolved CSG
    # spline curve, left commented out below (not executed); the OCC-based
    # construction further down is the one actually used.
    # spline = csg.SplineCurve2d() # create a 2d spline
    # eps = r*1e-2

    # # define the control points
    # pnts = [ (0,R-r), (-r+eps,R-r+eps), (-r,R),
    #         (-r+eps,R+r-eps), (0,R+r), (r-eps,R+r-eps), (r,R), (r-eps,R-r+eps) ]
    # # define the splines using the control points
    # segs = [ (0,1,2), (2,3,4), (4,5,6), (6,7,0) ]

    # # add the points and segments to the spline
    # for pnt in pnts:
    #     spline.AddPoint (*pnt)

    # for seg in segs:
    #     spline.AddSegment (*seg)

    # rev = csg.Revolution ( csg.Pnt(0,0,-1), csg.Pnt(0,0,1), spline)
    # geo = csg.CSGeometry()
    # geo.Add (rev.col([0,0,1]))

    # if geo_only:
    #     return geo
    # else:
    #     mesh = Meshing(geo, maxh, order_g, vol_or_bnd)
    #     return mesh, geo
    pnt1 = occ.Pnt(R-r, 0, 0 )
    pnt2 = occ.Pnt(R, 0, r )
    pnt3 = occ.Pnt(R+r , 0, 0 )
    pnt4 = occ.Pnt(R , 0 , -r)

    arc1 = occ.ArcOfCircle(pnt1, pnt2, pnt3)
    arc2 = occ.ArcOfCircle(pnt3, pnt4, pnt1)

    w = occ.Wire([arc1, arc2])
    body = w.Revolve(occ.Axis((0,0,0),occ.Z), 360).Rotate(occ.Axis((0,0,0),occ.X), 90)

    geo = occ.OCCGeometry(body)

    mesh = Meshing(geo, maxh, order_g, 'BND')
    return mesh

def generate_boundary_half_torus(maxh = 0.1, order_g = 1, R = 1.0, r = 0.4, bbnd_name = "bboundary"):
    """Build an open half-toroidal surface mesh (mesh.ne == 0) with major radius
    `R` and minor (tube) radius `r`, obtained by revolving the same circular
    cross-section as generate_boundary_torus by only 180 degrees; the two open
    (cut) rims are named `bbnd_name`.
    """

    pnt1 = occ.Pnt(R-r, 0, 0 )
    pnt2 = occ.Pnt(R, 0, r )
    pnt3 = occ.Pnt(R+r , 0, 0 )
    pnt4 = occ.Pnt(R , 0 , -r)

    arc1 = occ.ArcOfCircle(pnt1, pnt2, pnt3)
    arc2 = occ.ArcOfCircle(pnt3, pnt4, pnt1)

    w = occ.Wire([arc1, arc2])
    body = w.Revolve(occ.Axis((0,0,0),occ.Z), 180).Rotate(occ.Axis((0,0,0),occ.X), 90)

    body.edges[occ.Z<maxh/2].name = bbnd_name

    geo = occ.OCCGeometry(body)
    mesh = Meshing(geo, maxh, order_g, 'BND')
    
    return mesh

def generate_boundary_box(maxh = 0.1, order_g = 1, center=occ.Pnt(0,0,0), a = 1, b = 1, c = 1):
    """Build a closed rectangular-box surface mesh (mesh.ne == 0) with side
    lengths `a`, `b`, `c`, centered at `center`.
    """

    body = occ.Box(occ.Pnt(-a/2,-b/2,-c/2), occ.Pnt(a/2, b/2, c/2))
    body = body.Move((center[0], center[1], center[2]))

    geo= occ.OCCGeometry(body)

    mesh = Meshing(geo, maxh, order_g, 'BND')
    return mesh

def generate_boundary_smoothed_box(maxh = 0.1, order_g = 1, center=occ.Pnt(0,0,0), a = 1, b = 1, c = 1):
    """Build a rectangular-box surface mesh (mesh.ne == 0) like
    generate_boundary_box, but with all edges rounded off (filleted) with a
    radius equal to a third of the smallest side length, avoiding sharp edges
    that are hard to mesh/curve accurately.
    """

    fillet = min(a, b, c)/3
    body = occ.Box(occ.Pnt(-a/2,-b/2,-c/2), occ.Pnt(a/2, b/2, c/2))
    body = body.Move((center[0], center[1], center[2]))
    body = body.MakeFillet (body.edges, fillet)

    geo= occ.OCCGeometry(body)

    mesh = Meshing(geo, maxh, order_g, 'BND')
    return mesh

def import_stl_mesh(fname, maxh = 0.1, order_g = 1):
    """Load and mesh an externally supplied .stl surface geometry file `fname`,
    producing a surface mesh (mesh.ne == 0) with the given target size `maxh`
    and curved-element order `order_g`.
    """

    geo = stl.STLGeometry(fname)

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
    return mesh

# Unused/dead code: a generalized "n-torus" (torus with n circular holes cut
# tangentially around its circumference) mesh generator, left commented out below
# in its entirety (not executed anywhere).
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
# Manual, visual smoke-test: generates each geometry above (via a presumed but
# undefined `__all__` list of the geometry functions) and draws it interactively,
# meant to be run/inspected by hand, not as part of the automated test suite.
if __name__ == "__main__":

    for funct in __all__[:-1]:

        mesh, _ = globals()[funct](geo_only = False)
        print(funct, ', nel=', mesh.ne)
        DrawMesh(mesh)

    '''
    More advanced examples for stress testing the algorithms
    '''
    rel_path = "../../../data/geometries/"

    mesh, _ = import_stl_mesh(maxh = 2, fname = rel_path + 'bunny.stl')
    DrawMesh(mesh)

    mesh, _ = import_stl_mesh(maxh = 0.05, fname = rel_path + 'spot_cow.stl')
    DrawMesh(mesh)

    # mesh, _ = import_stl_mesh(maxh = 10, fname = rel_path + 'brain.stl')
    # DrawMesh(mesh)


# %%
