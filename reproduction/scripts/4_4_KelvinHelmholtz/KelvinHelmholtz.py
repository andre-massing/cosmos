from ngsolve import *
from netgen.csg import *
from ngsolve.internal import visoptions
from math import pi
from geometries.KH_geoms import *
from netgen import gui
import re

#SetNumThreads(10)

import socket
hostname = socket.gethostname()

preg = 1e-10
order = 8
bonus_intorder = 0
alpha = 10
# dt = 0.0005
dt = 0.001/28
IMEXRK2 = True

#dt = 0.0025
#tend = 20

case = 0

hmax = 0.05

Re = 1000

R = 1/(2*pi)
if case == 0:
    mesh,xi,eta,dxi,deta,n1,xidir,etadir = MakeKHGeom_Cyl(R=R,H=1,maxh=hmax,order=order+1)
elif case == 1:
    mesh,xi,eta,dxi,deta,n1,xidir,etadir = MakeKHGeom_Cyl(R=R,H=1,with_cover=True,maxh=hmax,order=order+1)
elif case == 2:
    mesh,xi,eta,dxi,deta,n1,xidir,etadir = MakeKHGeom_Cyl(R=R,H=1-1/pi,with_cover=True,maxh=hmax,order=order+1)
elif case == 3:
    mesh,xi,eta,dxi,deta,n1,xidir,etadir = MakeKHGeom_Cyl(R=R,H=0.5,with_cover=True,maxh=hmax,order=order+1)
elif case == 4:
    R = 1
    mesh,xi,eta,dxi,deta,n1,xidir,etadir = MakeKHGeom_Sphere(R=R,maxh=hmax,order=order+1)
elif case == 5:
    R = 1
    r = 1/3
    mesh,xi,eta,dxi,deta,n1,xidir,etadir = MakeKHGeom_Torus(R=R-r,r=r,maxh=hmax,order=order+1)
    
Draw(mesh)

nu = 1.0/28/Re

if IMEXRK2:
    gamma = 1 - sqrt(0.5)
    delta = 1 - 1/(2*gamma)
else:
    gamma = 1

uinf= 1
d0 = 1/28.0

tref = d0/uinf
tend = 600 * tref

vtk_dt = tref
postprocess_dt = tref*0.2

Draw(mesh)
Draw(xi,mesh,"xi")
Draw(eta,mesh,"eta")
Draw(n1,mesh,"normal")
Draw(xidir,mesh,"tau1")
Draw(etadir,mesh,"tau2")

tau = [xidir,etadir]

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

import os
if not os.path.exists(hostname+"_data"):
    os.makedirs(hostname+"_data")

outnamebase = hostname+"_data/Re_"+str(Re)+"_P_"+str(order)+"_bio_"+str(bonus_intorder)+"_h_"+str(hmax)+"dt_"+str(dt)

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
outfile.write("#hmax:  "+str(hmax)+str("\n"))
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

if case == 4 or case == 5: 
    cn=2*pi*1e-3
    mode_a = 16
    mode_b = 20
    mode_a_factor = 1
    mode_b_factor = 0.1
else:
    cn=1e-3
    mode_a = 8
    mode_b = 20
    mode_a_factor = 1
    mode_b_factor = 1

psi = cn*uinf*exp(-eta**2/(d0**2))*(mode_a_factor*cos(mode_a*pi*xi)+mode_b_factor*cos(mode_b*pi*xi))
dpsideta = - (2*eta)/(d0**2) * psi
mdpsidxi = cn *uinf*exp(-eta**2/(d0**2))*(mode_a_factor*mode_a*pi*sin(mode_a*pi*xi)+mode_b_factor*mode_b*pi*sin(mode_b*pi*xi))

rz = sqrt(x*x+y*y)/R
xidir0 = rz * xidir # CoefficientFunction((-y,x,0))

smearjump = tanh(2*eta/d0)

if case==5:
    smearjump = tanh(2*eta/d0) - tanh(2*(eta-0.5)/d0) - tanh(2*(eta+0.5)/d0)

uinit = rz * (uinf * smearjump * xidir) + 1.0/deta * dpsideta * xidir + 1.0/dxi *mdpsidxi * etadir

minit = BilinearForm(V)
minit += SymbolicBFI(u.Trace()*v.Trace() + div(u.Trace()) * q.Trace() + p.Trace() * div(v.Trace()), BND)
if preg > 0 :
    minit += SymbolicBFI (preg*p.Trace() *q.Trace(), BND)
minit += SymbolicBFI (1.0/h*tang(uhat-u.Trace()) * tang(vhat-v.Trace()), BND, element_boundary=True)
minit.Assemble()
print("finished assembling minit")

finit = LinearForm(V)
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

Redraw()
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

vtk = VTKOutput(ma=mesh,coefs=[gfu.components[0],gfu.components[2],vort],names=["velocity","pressure","vorticity"],filename=outnamebase+"_vtk",subdivision=3)
vtk.Do(vb=BND)
vtk.Do(vb=BND)

# input("A")
PostProcess(0.0,tref)

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
        #input("next step")
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

        
