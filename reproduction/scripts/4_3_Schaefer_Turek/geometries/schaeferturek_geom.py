from ngsolve import *
from netgen.csg import *
from ngsolve.internal import visoptions
from netgen import meshing

from math import pi
import re

def MakeSchaeferTurekGeomMesh(maxh=10,mapping=None,order=5):


    geo = CSGeometry()
    bottom   = Plane (Pnt(0,0,0), Vec(0,0, 1) )

    r = 0.05
    
    hole = Cylinder(Pnt(0.2,0.2,-1), Pnt(0.2,0.2,1),r)
        

    if (mapping in ["roled","kink","flat2"]):
        brick1 = Plane (Pnt(0,0,-1), Vec(-1,0, 0) ) * Plane (Pnt(1.1,0.41,1), Vec(1,0, 0) ) \
            * Plane (Pnt(0,0,-1), Vec(0,-1, 0) ) * Plane (Pnt(1.1,0.41,1), Vec(0,1, 0) ) \
            * Plane (Pnt(0,0,-1), Vec(0,0, -1) ) * Plane (Pnt(1.1,0.41,1), Vec(0,0,1) )
    
        brick2 = Plane (Pnt(1.1,0,-1), Vec(-1,0, 0) ) * Plane (Pnt(2.2,0.41,1), Vec(1,0, 0) ) \
            * Plane (Pnt(1.1,0,-1), Vec(0,-1, 0) ) * Plane (Pnt(2.2,0.41,1), Vec(0,1, 0) ) \
            * Plane (Pnt(1.1,0,-1), Vec(0,0, -1) ) * Plane (Pnt(2.2,0.41,1), Vec(0,0,1) )
        flowdom1 = bottom * (brick1-hole)
        flowdom2 = bottom * (brick2)
    
        geo.AddSurface(bottom,flowdom1)
        geo.AddSurface(bottom,flowdom2)
        ngmesh = geo.GenerateMesh(perfstepsend=meshing.MeshingStep.MESHSURFACE,maxh=maxh)
        ngmesh.SetCD2Name(1,"wall")
        ngmesh.SetCD2Name(2,"wall")
        ngmesh.SetCD2Name(3,"outflow")
        ngmesh.SetCD2Name(4,"wall")
        ngmesh.SetCD2Name(5,"wall")
        ngmesh.SetCD2Name(6,"middle")
        ngmesh.SetCD2Name(7,"inflow")
        ngmesh.SetCD2Name(8,"cyl")
    else:
        brick = Plane (Pnt(0,0,-1), Vec(-1,0, 0) ) * Plane (Pnt(2.2,0.41,1), Vec(1,0, 0) ) \
            * Plane (Pnt(0,0,-1), Vec(0,-1, 0) ) * Plane (Pnt(2.2,0.41,1), Vec(0,1, 0) ) \
            * Plane (Pnt(0,0,-1), Vec(0,0, -1) ) * Plane (Pnt(2.2,0.41,1), Vec(0,0,1) )
        
        flowdom = bottom * (brick-hole)
        
        geo.AddSurface(bottom,flowdom)
            
        ngmesh = geo.GenerateMesh(perfstepsend=meshing.MeshingStep.MESHSURFACE,maxh=maxh)
        ngmesh.SetCD2Name(1,"wall")
        ngmesh.SetCD2Name(2,"wall")
        ngmesh.SetCD2Name(3,"outflow")
        ngmesh.SetCD2Name(4,"inflow")
        ngmesh.SetCD2Name(5,"cyl")
    
    mesh = Mesh(ngmesh)
    mesh.Curve(order)
    Draw(mesh)
    
    Vdeform = Compress(H1(mesh,order=order,dim=mesh.dim, definedon=mesh.Boundaries(".*")))
    # Vdeform = H1(mesh,order=order,dim=mesh.dim, definedon=mesh.Boundaries(".*"))
    gfdeform = GridFunction(Vdeform)

    # no mapping:
    gfdeform.Set((0,0,0), definedon=mesh.Boundaries(".*"))
    
    # fries mapping:
    if mapping == "fries-A":
        xi = cos(pi*x/2.2)*(y+0.35)
        eta = sin(pi*x/2.2)*(y+0.35)
        zeta = 2+0.5*sqrt(xi**2+eta**2)-sin(3*sqrt(xi**2+eta**2))
        gfdeform.Set((xi-x,eta-y,zeta-z), definedon=mesh.Boundaries(".*"))

    # fries mapping:
    if mapping == "fries-B":
        def q(x):
            return -0.2/2.42 * x**2 + 0.44/2.42 * x
        r = x
        s = -(1+q(x))*(y-0.205)*cos(pi/6*(1-25/11*x))
        t = -(1+q(x))*(y-0.205)*sin(pi/6*(1-25/11*x))
        
        xi = cos(50/198*pi*r)*(s+6/5)
        eta = sin(50/198*pi*r)*(s+6/5)
        zeta = t + 0.2 * sin(3*r)
        gfdeform.Set((xi-x,eta-y,zeta-z), definedon=mesh.Boundaries(".*"))

    # kink mapping:
    if mapping == "kink":
        H = 0.5
        W = sqrt(1-H**2)
        xi = x*W
        eta = y
        zeta = H*IfPos(x-1.1,2.2-x,x)
        gfdeform.Set((xi-x,eta-y,zeta-z), definedon=mesh.Boundaries(".*"))
        
    # length preserving mapping:
    if mapping == "roled":
        R = 0.41/pi
        eta = -cos(pi*y/0.41)*R
        zeta = (1-sin(pi*y/0.41))*R
        gfdeform.Set((0,eta-y,zeta-z), definedon=mesh.Boundaries(".*"))

    tauhat1 = CoefficientFunction((1,0,0))
    tauhat2 = CoefficientFunction((0,1,0))
    normalhat = CoefficientFunction((0,0,1))

    F = CoefficientFunction(Id(3)  + Grad(gfdeform),dims=(3,3))

    n1_s = Inv(F).trans * normalhat
    n1 = n1_s / Norm(n1_s)
    
    tau1_s = F * tauhat1
    tau1 = tau1_s / Norm(tau1_s)
    
    tau2_s = F * tauhat2
    tau2 = tau2_s / Norm(tau2_s)

    gfn1 = GridFunction(Vdeform)
    gfn1.Set(n1, definedon=mesh.Boundaries(".*"))
    n1 = CoefficientFunction(gfn1 / Norm(gfn1),dims=(3,1))
    
    gftau1 = GridFunction(Vdeform)
    gftau1.Set(tau1, definedon=mesh.Boundaries(".*"))
    tau1 = CoefficientFunction(gftau1 / Norm(gftau1),dims=(3,1))

    gftau2 = GridFunction(Vdeform)
    gftau2.Set(tau2, definedon=mesh.Boundaries(".*"))
    tau2 = CoefficientFunction(gftau2 / Norm(gftau2),dims=(3,1))

    V = Compress(H1(mesh,order=order,definedon=mesh.Boundaries(".*")))
    
    gfxhat = GridFunction(V)
    gfxhat.Set(x, definedon=mesh.Boundaries(".*"))
    xhat = gfxhat

    gfyhat = GridFunction(V)
    gfyhat.Set(y, definedon=mesh.Boundaries(".*"))
    yhat = gfyhat
    
    Draw(gfdeform,mesh,"deform")
    mesh.SetDeformation(gfdeform)

    
    return mesh, gfdeform, (xhat, yhat), (n1,tau1,tau2)

if __name__ == "__main__":

    for mapping in ["None","fries-A","roled"]:
        mesh, gfdeform, (n1,tau1,tau2) = MakeSchaeferTurekGeomMesh(maxh=0.025,mapping=mapping, order=5)
        mesh.UnsetDeformation()
        vtk = VTKOutput(ma=mesh,coefs=[gfdeform],names=["gfdeform"],filename="st_geom_"+mapping+"_mesh",subdivision=0)
        vtk.Do(vb=BND)
        vtk = VTKOutput(ma=mesh,coefs=[n1, tau1,tau2, gfdeform],names=["normal", "tau1", "tau2", "gfdeform"],filename="st_geom_"+mapping,subdivision=3)
        vtk.Do(vb=BND)
        mesh.SetDeformation(gfdeform)
