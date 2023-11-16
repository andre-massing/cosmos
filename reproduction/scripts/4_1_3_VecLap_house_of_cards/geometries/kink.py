from ngsolve import *
from netgen.csg import *
from ngsolve.internal import visoptions
from netgen import meshing

from math import pi

def MakeKinkMesh(H,maxh=10):
    W = sqrt(1-H**2)
    
    geo = CSGeometry()
    bottom   = Plane (Pnt(0,0,0), Vec(0,0, 1) )
    
    surface = SplineSurface(bottom)
    pts = [(0,0,0),(0,1,0),(1,1,0),(1,0,0)]
    geopts = [surface.AddPoint(*p) for p in pts]
    for p1,p2,bc in [(0,1,"left"), (1, 2,"topleft"),(2,3,"middle"),(3,0,"bottomleft")]:
        surface.AddSegment(geopts[p1],geopts[p2],bc)
    geo.AddSplineSurface(surface)
    
    surface2 = SplineSurface(bottom)
    pts = [(1,0,0),(1,1,0),(2,1,0),(2,0,0)]
    geopts = [surface2.AddPoint(*p) for p in pts]
    for p1,p2,bc in [(0,1,"middle"), (1, 2,"topright"),(2,3,"right"),(3,0,"bottomright")]:
        surface2.AddSegment(geopts[p1],geopts[p2],bc)
    geo.AddSplineSurface(surface2)
        
    mesh = Mesh(geo.GenerateMesh(perfstepsend=meshing.MeshingStep.MESHSURFACE,maxh=maxh))
    
    # Vdeform = Compress(H1(mesh,order=1,dim=mesh.dim, definedon=mesh.Boundaries(".*")))
    Vdeform = H1(mesh,order=1,dim=mesh.dim, definedon=mesh.Boundaries(".*"))
    gfdeform = GridFunction(Vdeform)

    xi = IfPos(x-1,2-x,x)
    gfdeform.Set(((W-1)*x,0,H*xi), definedon=mesh.Boundaries(".*"))
    
    Draw(gfdeform,mesh,"deform")
    mesh.SetDeformation(gfdeform)
    surfacearea=Integrate(1,mesh,BND)
    assert(abs(surfacearea-2)<1e-12)
    assert(abs(Integrate(z,mesh,BND)-H)<1e-12)
    return mesh

def RefineKinkMesh(mesh,H):
    W = sqrt(1-H**2)
    mesh.UnsetDeformation()
    mesh.Refine()
    Vdeform = Compress(H1(mesh,order=1,dim=mesh.dim, definedon=mesh.Boundaries(".*")))
    gfdeform = GridFunction(Vdeform)
    xi = IfPos(x-1,2-x,x)
    gfdeform.Set(((W-1)*x,0,H*xi), definedon=mesh.Boundaries(".*"))
    Draw(gfdeform,mesh,"deform")
    mesh.SetDeformation(gfdeform)

if __name__ == "__main__":
    mesh = MakeKinkMesh(0,maxh=0.2)
    vtk = VTKOutput(ma=mesh,coefs=[],names=[],filename="mesh_flat",subdivision=0)
    vtk.Do(vb=BND)
    
    mesh = MakeKinkMesh(sqrt(3/4),maxh=0.2)
    vtk = VTKOutput(ma=mesh,coefs=[],names=[],filename="mesh_house_of_cards",subdivision=0)
    vtk.Do(vb=BND)
