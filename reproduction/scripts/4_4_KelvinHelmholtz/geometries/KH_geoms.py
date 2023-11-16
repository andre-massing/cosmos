from ngsolve import *
from netgen.csg import *
from ngsolve.internal import visoptions
from netgen import meshing

from math import pi

def MakeKHGeom_Cyl(R=1/(2*pi),H=1,with_cover=False,maxh=10,order=5):

    geo = CSGeometry()
    bottom   = Plane (Pnt(0,0,-H/2), Vec(0,0, -1) )
    top   = Plane (Pnt(0,0,H/2), Vec(0,0, 1) )
    hole = Cylinder(Pnt(0,0,-H/2), Pnt(0,0,H/2),R)
    flowdom = bottom * top * hole

    if with_cover:
        geo.Add(flowdom)
    else:
        geo.AddSurface(hole,flowdom)
    
    ngmesh = geo.GenerateMesh(perfstepsend=meshing.MeshingStep.MESHSURFACE,quad=False,maxh=maxh)
    
    ngmesh.SetCD2Name(1,"bottom")
    ngmesh.SetCD2Name(2,"top")
    
    mesh = Mesh(ngmesh)
    mesh.Curve(order)
    Draw(mesh)

    xi = (-atan2(x,y) / (2*pi) + 0.5)
    def ABS(q):
        return IfPos(q,q,-q)
    n = 1.0/R*CoefficientFunction((x,y,0))
    rz = sqrt(x**2+y**2)
    tau1 = 1.0/rz*CoefficientFunction((-y,x,0))
    tau2 = CoefficientFunction((0,0,1))
    if with_cover:
        eps=1e-16
        eta = IfPos(z,
                   IfPos(z-H/2+eps,R-sqrt(x**2+y**2)+H/2,z),
                   IfPos(-H/2-z+eps,-R+sqrt(x**2+y**2)-H/2,z))
        n = IfPos(z,
                   IfPos(z-H/2+eps,CoefficientFunction((0,0,1)),n),
                   IfPos(-H/2-z+eps,CoefficientFunction((0,0,-1)),n))
        tau2 = IfPos(z,
                   IfPos(z-H/2+eps,1.0/sqrt(x**2+y**2)*CoefficientFunction((-x,-y,0)),tau2),
                   IfPos(-H/2-z+eps,1.0/sqrt(x**2+y**2)*CoefficientFunction((x,y,0)),tau2))
    else:
        eta = z # eta = (asin(z) / pi) + 0.5
        
    return mesh, xi, eta, 2 * pi * rz, 1,  n, tau1, tau2

def MakeKHGeom_Torus(R=2,r=1,maxh=10,order=5):
    if order==9 and R==1-1/3 and r==1/3 and maxh==0.05:
        print("loading torus mesh from file")
        mesh = Mesh("geometries/torus.vol.gz")
    else:
        geo = CSGeometry()
        geo.Add(Torus(Pnt(0,0,0),Vec(0,0,1),R,r))
        ngmesh = geo.GenerateMesh(perfstepsend=meshing.MeshingStep.MESHSURFACE,quad=False,maxh=maxh)
        mesh = Mesh(ngmesh)
    mesh.Curve(order)
    Draw(mesh)
    xi = (-atan2(x,y) / (2*pi) + 0.5)

    qx = sqrt(x**2+y**2)-R
    qz = z
    eta = (atan2(qz,qx) / (2*pi))

    Q = R*CoefficientFunction((sin(2*pi*xi),-cos(2*pi*xi),0))
    normal = 1.0/r * (CoefficientFunction((x,y,z))-Q)

    rz = sqrt(x**2+y**2)
    tau1 = 1.0/rz*CoefficientFunction((-y,x,0))
    tau2 = Cross(normal,tau1)
    
    return mesh,xi,eta,2 * pi * rz, 2 * pi * r,normal,tau1,tau2

def MakeKHGeom_Sphere(R=1,maxh=10,order=5):
    geo = CSGeometry()
    geo.Add(Sphere(Pnt(0,0,0),R).bc("sphere"))
    ngmesh = geo.GenerateMesh(perfstepsend=meshing.MeshingStep.MESHSURFACE,quad=False,maxh=maxh)
    
    mesh = Mesh(ngmesh)
    mesh.Curve(order)
    Draw(mesh)

    xi = (-atan2(x,y) / (2*pi) + 0.5)
    eta = (asin(z) / pi)
    n = 1.0/R*CoefficientFunction((x,y,z))
    rz = sqrt(x**2+y**2)
    tau1 = 1.0/rz*CoefficientFunction((-y,x,0))
    tau2 = Cross(n,tau1)
    return mesh,xi,eta, 2 * pi * rz, 2 * pi * R, n,tau1,tau2

if __name__ == "__main__":
    mesh,xi,eta,dxi,deta,n,tau1,tau2 = MakeKHGeom_Cyl(R=1/(2*pi),H=1)
    Draw(mesh)
    Draw(xi,mesh,"xi")
    Draw(eta,mesh,"eta")
    Draw(dxi,mesh,"dxi")
    Draw(deta,mesh,"deta")
    Draw(n,mesh,"normal")
    Draw(tau1,mesh,"tau1")
    Draw(tau2,mesh,"tau2")
    input("")
    
    mesh,xi,eta,dxi,deta,n,tau1,tau2 = MakeKHGeom_Cyl(R=1/(2*pi),H=1,with_cover=True)
    Draw(mesh)
    Draw(xi,mesh,"xi")
    Draw(eta,mesh,"eta")
    Draw(dxi,mesh,"dxi")
    Draw(deta,mesh,"deta")
    Draw(n,mesh,"normal")
    Draw(tau1,mesh,"tau1")
    Draw(tau2,mesh,"tau2")
    input("")
    
    mesh,xi,eta,dxi,deta,n,tau1,tau2 = MakeKHGeom_Cyl(R=1/(2*pi),H=1-1/pi,with_cover=True)
    Draw(mesh)
    Draw(xi,mesh,"xi")
    Draw(eta,mesh,"eta")
    Draw(dxi,mesh,"dxi")
    Draw(deta,mesh,"deta")
    Draw(n,mesh,"normal")
    Draw(tau1,mesh,"tau1")
    Draw(tau2,mesh,"tau2")
    input("")
    
    mesh,xi,eta,dxi,deta,n,tau1,tau2 = MakeKHGeom_Sphere(R=1/(2*pi))
    Draw(mesh)
    Draw(xi,mesh,"xi")
    Draw(eta,mesh,"eta")
    Draw(dxi,mesh,"dxi")
    Draw(deta,mesh,"deta")
    Draw(n,mesh,"normal")
    Draw(tau1,mesh,"tau1")
    Draw(tau2,mesh,"tau2")
    input("")
    
    mesh,xi,eta,dxi,detan,tau1,tau2 = MakeKHGeom_Torus(R=40*(1/(2*pi)-0.06),r=40*(0.06))
    Draw(mesh)
    Draw(xi,mesh,"xi")
    Draw(eta,mesh,"eta")
    Draw(dxi,mesh,"dxi")
    Draw(deta,mesh,"deta")
    Draw(n,mesh,"normal")
    Draw(tau1,mesh,"tau1")
    Draw(tau2,mesh,"tau2")
    input("")
    
    # vtk = VTKOutput(ma=mesh,coefs=[gfdeform],names=["gfdeform"],filename="st_geom_"+mapping+"_mesh",subdivision=0)
    # vtk.Do(vb=BND)
