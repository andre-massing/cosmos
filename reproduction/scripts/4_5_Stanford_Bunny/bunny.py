from ngsolve import *
from netgen.csg import *
from ngsolve.internal import visoptions
from math import pi
from netgen import gui
from netgen import meshing
import re

from netgen import stl
SetNumThreads(7)

import socket
hostname = socket.gethostname()

preg = 1e-10
order = 4
bonus_intorder = order
alpha = 80
# dt = 0.0005
dt = 0.005
IMEXRK2 = True

hmax=4

#input("")
# from netgen import occ
# geo = occ.OCCGeometry("geometries/bunny.stp")

#geo = stl.STLGeometry("geometries/bunny-geo.stl")
geo = stl.STLGeometry("geometries/bunny.stl")
# geo = stl.STLGeometry("geometries/bunny-new.stl")
ngmesh = geo.GenerateMesh(perfstepsend=meshing.MeshingStep.MESHSURFACE,
                          quad=False,
                          maxh=hmax,
                          grading=0.7,
                          optimize2d = "smsmsmSmSmSm",
                          yangle=50,
                          contyangle=50,
                          edgecornerangle=30)
# mesh = Mesh(ngmesh)
# mesh = Mesh("geometries/bunny.vol.gz")
# ngmesh = netgen.meshing.Mesh()
# ngmesh.Load("geometries/bunny-mesh-003-p5-6978.vol.gz")
# ngmesh.SetGeometry(geo)
mesh = Mesh(ngmesh)
mesh.Curve(4)
Draw(mesh)

# normal = specialcf.normal(3)

# vtk = VTKOutput(ma=mesh,coefs=[normal],names=["normal"],filename="normal_vtk",subdivision=2)
# vtk.Do(vb=BND)

# asdf
# Draw(x,mesh,"x")
# Draw(y,mesh,"y")
# Draw(z,mesh,"z")
# input("")

# asdf

nu = 0.02

if IMEXRK2:
    gamma = 1 - sqrt(0.5)
    delta = 1 - 1/(2*gamma)
else:
    gamma = 1

uinf= 1
d0 = 1/28.0

tref = 1 #d0/uinf
tend = 500 * tref

vtk_dt = tref
postprocess_dt = tref*0.5

Draw(mesh)
# Draw(xi,mesh,"xi")
# Draw(eta,mesh,"eta")
# Draw(n1,mesh,"normal")
# Draw(xidir,mesh,"tau1")
# Draw(etadir,mesh,"tau2")

# tau = [xidir,etadir]

def MarkDofsOnBBND(bitarrays, bbnd, V, comp=None, clear=False):
  if type(bitarrays) != list:
    bitarrays = [bitarrays]
  offset = 0
  if comp != None:
    W = V.components[comp]
    if comp > 0:
      for i in range(comp):
        offset += V.components[i].ndof
  else:
    W = V
  for el in W.Elements(BBND):
    bbnd_name = mesh.GetBBoundaries()[el.index]
    match = re.search(bbnd, bbnd_name)
    if match != None:
      for dofs in el.dofs:
        for ba in bitarrays:
          if clear:
            ba.Clear(dofs+offset)
          else:
            ba.Set(dofs+offset)


VDivSurf = HDivSurface(mesh, order = order, dirichlet="top|bottom")
MarkDofsOnBBND(VDivSurf.FreeDofs(), "top|bottom", VDivSurf, clear=True)

VHat = HCurl(mesh, order = order, flags={"orderface": 0})
#VHat = HCurl(mesh, order = order, order_face = 0)
#Q = FESpace(type = l2surf, mesh=mesh, order = order-1)
Q = SurfaceL2(mesh, order = order-1)

V = FESpace([VDivSurf,VHat,Q])
MarkDofsOnBBND(V.FreeDofs(), "top|bottom", V, comp=0, clear=True)

u, uhat,p = V.TrialFunction()
v, vhat,q = V.TestFunction()

normal = specialcf.normal(3)
tangential = specialcf.tangential(3)
n = Cross(normal,tangential)
nmat = CoefficientFunction(normal, dims =(3,1))
proj = Id(3) - nmat*nmat.trans


qq = CoefficientFunction((y,-x,0))
qq = qq - (qq * normal) * normal
xidir = qq/Norm(qq)
etadir = Cross(normal,xidir)
Draw(xidir,mesh,"xidir")
Draw(etadir,mesh,"etadir")


def tang(vec):
    return vec - (vec*n)*n

h = specialcf.mesh_size

gradu = proj * CoefficientFunction ( (grad(u),), dims=(3,3) ) * proj
gradv = proj * CoefficientFunction ( (grad(v),), dims=(3,3) ) * proj

epsu = 0.5*(gradu + gradu.trans)
epsv = 0.5*(gradv + gradv.trans)

vhat = vhat.Trace()
uhat = uhat.Trace()

m = BilinearForm(V)
m += SymbolicBFI(u.Trace()*v.Trace(), BND)
m.Assemble()
print("finished assembling m")

a = BilinearForm(V)
a += SymbolicBFI( 2*nu*InnerProduct(epsu, epsv), BND)
a += SymbolicBFI ( 2*nu*InnerProduct ( epsu * n,  tang(vhat-v.Trace()) ), BND, element_boundary=True )
a += SymbolicBFI ( 2*nu*InnerProduct ( epsv * n,  tang(uhat-u.Trace()) ), BND, element_boundary=True )
a += SymbolicBFI ( nu*alpha*order*order/h * InnerProduct ( tang(vhat-v.Trace()),  tang(uhat-u.Trace()) ) ,BND, element_boundary=True )
a += SymbolicBFI ( div(u.Trace()) *q.Trace(), BND)
a += SymbolicBFI ( div(v.Trace()) *p.Trace(), BND)

if preg > 0 :
    a += SymbolicBFI (  preg*p.Trace() *q.Trace(), BND)
#a += SymbolicBFI ( 1e6 * div(u.Trace()) * div(v.Trace()), BND, element_boundary=True )
a.Assemble()
print("finished assembling a")

conv = BilinearForm(V, nonassemble=True)
conv += (-InnerProduct(gradv*u.Trace(), u.Trace()) * ds(bonus_intorder = bonus_intorder)).Compile(True,True)
u_Other = (u.Trace()*n)*n + tang(uhat)
conv += (IfPos(u.Trace() * n, u.Trace()*n*u.Trace()*v.Trace(), u.Trace()*n*u_Other*v.Trace()) * ds(bonus_intorder = bonus_intorder, element_boundary = True)).Compile(True,True)

f = LinearForm(V)
f.Assemble()
print("finished assembling f")

# U0 = 1.5
# uin = CoefficientFunction(U0*4*y*(0.41-y)/(0.41*0.41) * IfPos(1e-2-x,1,0))
# dirvals = CoefficientFunction(( 0,0,uin) )
# Draw(cf =dirvals, mesh = mesh, name = "dirvals")

gfu = GridFunction(V)
# gfu.components[0].Set(dirvals, definedon = mesh.Boundaries("dir"))
# gfu.components[1].Set(dirvals, definedon = mesh.Boundaries("dir"))

velocity = CoefficientFunction(gfu.components[0])
pressure = CoefficientFunction(gfu.components[2])
Draw(pressure, mesh = mesh, name = "pressure")
Draw(velocity, mesh = mesh, name = "velocity")
divu = Trace(proj * CoefficientFunction(gfu.components[0].Operator("grad", BND), dims=(3,3)).trans * proj)
Draw(divu, mesh = mesh, name = "divvelocity")
gradgfu= proj * CoefficientFunction((gfu.components[0].Operator("grad",BND),),dims=(3,3)) * proj
vort = InnerProduct(xidir,gradgfu*etadir) - InnerProduct(etadir,gradgfu*xidir)
Draw(vort,mesh,"vort")

visoptions.scalfunction = "velocity:0"

gfvort = GridFunction(H1(mesh,order=order))
dvort= proj * grad(gfvort)
gfvort.Set(vort,definedon=mesh.Boundaries(".*"))
Draw(gfvort,mesh,"gfvort")


import os
if not os.path.exists(hostname+"_data"):
    os.makedirs(hostname+"_data")

outnamebase = hostname+"_data/P_"+str(order)+"_bio_"+str(bonus_intorder)+"_h_"+"dt_"+str(dt)

outfilename = outnamebase+".txt"
outfile = open(outfilename,"w")
outfile.write("#dofs:  "+str(V.ndof)+str("\n"))
outfile.write("#cdofs: "+str(sum(V.FreeDofs(True)))+str("\n"))
outfile.write("#nface: "+str(mesh.nface)+str("\n"))
outfile.write("#nedge: "+str(mesh.nedge)+str("\n"))
outfile.write("#nv:    "+str(mesh.nv)+str("\n"))
outfile.write("#preg:  "+str(preg)+str("\n"))
outfile.write("#alpha: "+str(alpha)+str("\n"))
outfile.write("#dt:    "+str(dt)+str("\n"))
# outfile.write("#hmax:  "+str(hmax)+str("\n"))
outfile.close()

outfilename = outnamebase+".dat"
outfile = open(outfilename,"w")
outfile.write("#time\ttref\tdiverr\tenergy\tenstrophy\tpalinstrophy\n")
outfile.close()

diverr = (divu**2).Compile(True,True)
energy = (InnerProduct(gfu.components[0],gfu.components[0])).Compile(True,True)
enstrophy = (vort**2).Compile(True,True)
palinstrophy = (InnerProduct(dvort,dvort)).Compile(True,True)

def PostProcess(t,tref):
    outfile = open(outfilename,"a")
    total_diverr = sqrt(Integrate(diverr,mesh,BND,order=2*order))
    total_energy = 0.5*Integrate(energy,mesh,BND,order=2*order)
    total_enstrophy = 0.5*Integrate(enstrophy,mesh,BND,order=2*order)
    gfvort.Set(vort,definedon=mesh.Boundaries(".*"))
    total_palinstrophy = 0.5*Integrate(palinstrophy,mesh,BND,order=2*order)
    outfile.write("{:12.6e}\t{:12.6e}\t{:12.6e}\t{:12.6e}\t{:12.6e}\t{:12.6e}".format(t,t/tref,total_diverr,total_energy,total_enstrophy,total_palinstrophy)+"\n")
    outfile.close()

bt = V.FreeDofs()
res = gfu.vec.CreateVector()

mstar = m.mat.CreateMatrix()
mstar.AsVector().data = m.mat.AsVector() + gamma * dt * a.mat.AsVector()

if preg > 0:
    inv = mstar.Inverse(bt, inverse="sparsecholesky")
else:
    inv = mstar.Inverse(bt, inverse="pardiso")

def tanh(x):
    return 1 - 2/(exp(2*x)+1)

# uinit = 0.03*tanh(0.25*(z-42.7)) * CoefficientFunction((y,-x,0))





# asdf

psiinit = 0

xyzs = []

import random
n = 0
ntry = 0
while(n<200 and ntry < mesh.nv-1):
    ntry += 1
    vertnr = random.randint(0,mesh.nv-1)
    if n%2 == 0:
        sign = 1
    else:
        sign = -1
        # print()
    vert = mesh[NodeId(VERTEX,vertnr)]
    x0,y0,z0 = vert.point
    print(f"candidate {vertnr} (n = {n}) (ntry = {ntry})")
    notaccept = False
    for xyz in xyzs:
        if sqrt((xyz[0]-x0)**2+(xyz[1]-y0)**2+(xyz[2]-z0)**2) < 3.5**2:
            notaccept = True
    if notaccept:
        continue
    n += 1
    xyzs.append((x0,y0,z0))
    print(f"add ({x0},{y0},{z0})")
    psiinit = psiinit + sign*20*exp(-0.05*((x-x0)**2+(y-y0)**2+(z-z0)**2))
# Draw(psiinit,mesh,"psiinit"+str(n))

# x0 = -41.869 
# y0 = -10.2367
# z0 = 54.744  
# x1 = -23.8674
# y1 = -21.4355
# z1 = 55.6725 

# psiinit = - 25*exp(-0.0125*((x-x0)**2+(y-y0)**2+(z-z0)**2)) - 25*exp(-0.0125*((x-x1)**2+(y-y1)**2+(z-z1)**2))
# Draw(psiinit,mesh,"psi")
gfpsiinit = GridFunction(H1(mesh,order=order+2))
gfpsiinit.Set(psiinit,definedon=mesh.Boundaries(".*"))
# Draw(gfpsiinit,mesh,"gfpsi")
# dpsi = gfpsiinit.Operator("grad", BND)
dpsi = grad(gfpsiinit)#.Operator("grad", BND)
# Draw(dpsi,mesh,"dpsi")
dpsidxi = InnerProduct(xidir,dpsi)
# Draw(dpsidxi,mesh,"dpsidxi")
dpsideta = InnerProduct(etadir,dpsi)
# Draw(dpsideta,mesh,"dpsideta")
uinit = dpsideta*xidir-dpsidxi*etadir
gfvort.Set(vort,definedon=mesh.Boundaries(".*"))
# Draw(uinit,mesh,"uinit")
# uinit = exp(-((x-0.5)**2*(y-0.5)**2+(z-0.5)**2)) * etadir

minit = BilinearForm(V)
minit += SymbolicBFI(u.Trace()*v.Trace() + div(u.Trace()) * q.Trace() + p.Trace() * div(v.Trace()), BND)
# minit += SymbolicBFI(u.Trace()*v.Trace(), BND)
if preg > 0 :
    minit += SymbolicBFI (preg*p.Trace() *q.Trace(), BND)
minit += SymbolicBFI (1.0e-6/h*tang(uhat-u.Trace()) * tang(vhat-v.Trace()), BND, element_boundary=True)
minit.Assemble()
print("finished assembling minit")

finit = LinearForm(V)
# finit += SymbolicLFI( uinit*v.Trace(), BND, simd_evaluate=False)
finit += SymbolicLFI( uinit*v.Trace(), BND, simd_evaluate=False)
finit.Assemble()
print("finished assembling finit")

if preg > 0:
    invm = minit.mat.Inverse(bt, inverse="sparsecholesky")
else:
    invm = minit.mat.Inverse(bt, inverse="pardiso")

for i in range(5):
    res.data = finit.vec -  minit.mat * gfu.vec
    gfu.vec.data += invm * res

# input("")

t = 0
k=0
convvec = gfu.vec.CreateVector()
convvec[:] = 0.0
convvec1 = gfu.vec.CreateVector()
convvec1[:] = 0.0
convvec2 = gfu.vec.CreateVector()
convvec2[:] = 0.0
diffvec1 = gfu.vec.CreateVector()
diffvec1[:] = 0.0
diffvec2 = gfu.vec.CreateVector()
diffvec2[:] = 0.0
u1 = gfu.vec.CreateVector()
u1[:] = 0.0

vtk = VTKOutput(ma=mesh,coefs=[gfu.components[0],gfu.components[2],gfvort],names=["velocity","pressure","vorticity"],filename=outnamebase+"_vtk",subdivision=2)
vtk.Do(vb=BND)
vtk.Do(vb=BND)

# input("A")
PostProcess(0.0,tref)
Redraw()

import time

start = time.time()
estimated_time_to_arrival = float("nan")
with TaskManager():
    while t < tend:
        linebreak = False
        if not IMEXRK2: # IMEX1
          convvec.data = conv.mat * gfu.vec 
          res.data = a.mat * gfu.vec + convvec
          gfu.vec.data -= dt * inv * res
        else:     # IMEXRK2
          convvec1.data = conv.mat * gfu.vec 
          diffvec1.data = a.mat * gfu.vec
          res.data = convvec1 + diffvec1
          
          u1.data = gfu.vec.data - gamma * dt * inv * res
          
          convvec2.data = conv.mat * u1
          diffvec2.data = a.mat * u1

          res.data = delta*convvec1 + (1-delta)*convvec2 + gamma*diffvec1 + (1-gamma)*diffvec2
          gfu.vec.data -= dt * inv * res
        
        t = t + dt
        k = k + 1
        #if(k%10==0):
        Redraw()
        #Redraw()
        # input("next step")
        time_taken = time.time() - start
        estimated_total_time = time_taken*tend/t
        estimated_time_to_arrival = estimated_total_time-time_taken
        print ("\rt = {:12.9f} / tbar={:12.9f} / , remain. comp. time: {:8.1f} (seconds)".format(t,t/tref,estimated_time_to_arrival),end="")
        
        if k%(postprocess_dt//dt)==0:
            print(" -postpr.:",end="")
            PostProcess(t,tref)
            print(" done.",end="")
            linebreak = True            
        if k%(vtk_dt//dt)==0:
            print(" -vtkout:",end="")
            vtk.Do(vb=BND)
            print(" done.",end="")
            linebreak = True
        if linebreak:
            print("")

        
