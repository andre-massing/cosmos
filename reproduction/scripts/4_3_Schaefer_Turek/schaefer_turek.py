from ngsolve import *
from ngsolve.internal import visoptions
from math import pi
import re

from netgen import gui

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ku", help="order", type=int)
parser.add_argument("--kg", help="geom_order", type=int)
parser.add_argument("--maxh", help="maxh", type=float)
parser.add_argument("--dt", help="dt", type=float)
parser.add_argument("--mapping", help="mapping", type=str)
parser.add_argument("--nthreads", help="number of threads", type=int)
parser.add_argument("--vtk_start_time", help="sampling start time", type=float)
parser.add_argument("--vtk_interval", help="sampling interval for vtk (only after vtk_start_time)", type=float)
args = parser.parse_args()

if args.ku == None:
    args.ku = 4
    print("Setting ku to", args.ku)

if args.kg == None:
    args.kg = args.ku + 1
    print("Setting kg to", args.kg)

if args.dt == None:
    args.dt = 0.001
    print("Setting dt to", args.dt)

if args.maxh == None:
    args.maxh = 0.05
    print("Setting maxh to", args.maxh)

if args.mapping == None:
    args.mapping = "flat"
    print("Setting mapping to", args.mapping)

if args.nthreads != None:
    SetNumThreads(args.nthreads)    

if args.vtk_start_time == None:
    args.vtk_start_time = 28
    print("Setting vtk_start_time to", args.vtk_start_time)
    
if args.vtk_interval == None:
    args.vtk_interval = 0.02
    print("Setting vtk_interval to", args.vtk_interval)
    
order = args.ku
# maxh = 0.025
# dt = 0.0005
maxh = args.maxh
dt = args.dt

mapping = args.mapping
# mapping = "flat"

from geometries.schaeferturek_geom import MakeSchaeferTurekGeomMesh
mesh, deform, (xhat,yhat), (n1, tau1, tau2) = MakeSchaeferTurekGeomMesh(maxh=maxh, mapping=mapping, order = args.kg)
tau = [tau1,tau2]

alpha = 10
nu = 0.001

gamma = 1 - sqrt(0.5)
delta = 1 - 1/(2*gamma)

VDivSurf = HDivSurface(mesh, order = order)
VHat = HCurl(mesh, order = order, flags={"orderface": 0})
Q = SurfaceL2(mesh, order = order-1)

V = FESpace([VDivSurf,VHat,Q])

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

MarkDofsOnBBND([V.FreeDofs(False),V.FreeDofs(True)], "inflow|wall|cyl", V, clear=True)

u, uhat,p = V.TrialFunction()
v, vhat,q = V.TestFunction()

def Cross(u,v):
    return CoefficientFunction((u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2] ,u[0]*v[1]-u[1]*v[0]))

normal = specialcf.normal(3)
tangential = specialcf.tangential(3)
n = Cross(normal,tangential)

nmat = CoefficientFunction(normal, dims =(3,1))
proj = Id(3) - nmat*nmat.trans

def tang(vec):
    #return (vec * tangential)*tangential
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
a += SymbolicBFI ( 2*nu*InnerProduct(epsu, epsv), BND)
a += SymbolicBFI ( 2*nu*InnerProduct ( epsu * n,  tang(vhat-v.Trace()) ), BND, element_boundary=True )
a += SymbolicBFI ( 2*nu*InnerProduct ( epsv * n,  tang(uhat-u.Trace()) ), BND, element_boundary=True )
a += SymbolicBFI ( nu*alpha*order*order/h * InnerProduct ( tang(vhat-v.Trace()),  tang(uhat-u.Trace()) ) ,BND, element_boundary=True )
a += SymbolicBFI ( div(u.Trace()) *q.Trace(), BND)
a += SymbolicBFI ( div(v.Trace()) *p.Trace(), BND)
a += SymbolicBFI ( 1e-8*p.Trace() *q.Trace(), BND)
#a += SymbolicBFI ( 1e6 * div(u.Trace()) * div(v.Trace()), BND, element_boundary=True )
a.Assemble()

astress = BilinearForm(V, symmetric=False, check_unused=False)
astress += SymbolicBFI ( 2*nu*InnerProduct ( -epsu * n,  tang(vhat) ), BND, element_boundary=True )
astress += SymbolicBFI ( 2*nu*InnerProduct ( -epsu * n,  (v.Trace()*n)*n ), BND, element_boundary=True )
astress += SymbolicBFI ( -div(v.Trace()) *p.Trace(), BND)
#astress += SymbolicBFI ( nu*alpha*order*order/h * InnerProduct ( tang(vhat),  tang(uhat-u.Trace()) ) ,BND, element_boundary=True )
astress.Assemble()

print("finished assembling a")

conv = BilinearForm(V, nonassemble=True)
conv += (-InnerProduct(gradv*u.Trace(), u.Trace()) * ds).Compile(True,True)
u_Other = (u.Trace()*n)*n + tang(uhat)
conv += (IfPos(u.Trace() * n, u.Trace()*n*u.Trace()*v.Trace(), u.Trace()*n*u_Other*v.Trace()) * ds(element_boundary = True)).Compile(True,True)


f = LinearForm(V)
f.Assemble()
print("finished assembling f")

print("preparing force comp. test functions")

# right = mesh(0.25+1e-8,0.2) # undeformed coords
# left = mesh(0.15-1e-8,0.2)  # undeformed coords
scale = 20.0

cyldofs = [BitArray(V.components[0].FreeDofs()),BitArray(V.components[1].FreeDofs())]
for i in range(2):
  cyldofs[i].Clear()
  MarkDofsOnBBND(cyldofs[i], "cyl", V.components[i], clear=False)

gfdlxyz = [GridFunction(V) for i in range(6)]

force_dir = [tau[0], tau[1], n1,
             CoefficientFunction((1,0,0)),
             CoefficientFunction((0,1,0)),
             CoefficientFunction((0,0,1))]

for i,direction in [(0,n),
                    (1,tangential)]: # hdiv-set / hcurl-set
  ms = BilinearForm(V.components[i])
  ms += SymbolicBFI((V.components[i].TestFunction().Trace()*direction) * (V.components[i].TrialFunction().Trace()*direction),BND, element_boundary=True)
  ms.Assemble()
  msinv = ms.mat.Inverse(cyldofs[i])
  for j in range(6): # drag/lift
    mf = LinearForm(V.components[i])
    mf += SymbolicLFI(InnerProduct(V.components[i].TestFunction().Trace(),direction)*InnerProduct(direction,force_dir[j]),BND, element_boundary=True)
    mf.Assemble()
    gfdlxyz[j].components[i].vec[:] = 0
    gfdlxyz[j].components[i].vec.data = msinv * mf.vec
del ms, msinv, mf

gfdrag, gflift, gfnormal, gfforcex, gfforcey, gfforcez = gfdlxyz

fvec = gfdrag.vec.CreateVector()
def CalcForces(gfu):
  fvec.data = -a.mat * gfu.vec
  fvec.data += -conv.mat * gfu.vec 
  # fvec.data = astress.mat * gfu.vec
  # p = gfu.components[2]
  drag = InnerProduct(fvec, gfdrag.vec)*scale
  lift = InnerProduct(fvec, gflift.vec)*scale
  perp = InnerProduct(fvec, gfnormal.vec)*scale
  forcex = InnerProduct(fvec, gfforcex.vec)
  forcey = InnerProduct(fvec, gfforcey.vec)
  forcez = InnerProduct(fvec, gfforcez.vec)
  # forces = sqrt(drag**2+lift**2) / scale
  forces = sqrt(forcex**2+forcey**2+forcez**2)
  
  eps = 1e-12
  dp = -gfu.components[2](0.15-eps,0.2,VOL_or_BND=BND)+gfu.components[2](0.25+eps,0.2,VOL_or_BND=BND)
  
  return drag,lift,perp,forcex,forcey,forcez,forces,dp


print("finished force comp. preparations")

U0 = 1.5
uin_n = U0*4*yhat*(0.41-yhat)/(0.41*0.41)

if (re.search("fries",mapping) != None):
  Ffes = SurfaceL2(mesh, order=order, dim=9)
  gfF = GridFunction(Ffes)
  mesh.UnsetDeformation()
  gfF.Set(CoefficientFunction(Id(3)  + Grad(deform),dims=(3,3)),definedon=mesh.Boundaries(".*"))
  mesh.SetDeformation(deform)
  F = CoefficientFunction(gfF, dims=(3,3))
  uin_hat = CoefficientFunction((uin_n,0,0))
  uin = F * uin_hat
else:
  uin = uin_n * tau1

# Draw(xhat,mesh,"xhat")
# Draw(yhat,mesh,"yhat")

gfu = GridFunction(V)

if False:
  inflowdofs = BitArray(V.components[1].FreeDofs())
  inflowdofs.Clear()
  MarkDofsOnBBND(inflowdofs, "inflow", V.components[1], clear=False)

  minit = BilinearForm(V.components[1])
  minit += SymbolicBFI((V.components[1].TestFunction().Trace()*n) * (V.components[1].TrialFunction().Trace()*n),BND, element_boundary=True)
  minit.Assemble()
  mfinit = LinearForm(V.components[1])
  mfinit += SymbolicLFI(InnerProduct(V.components[1].TestFunction().Trace(),n)*InnerProduct(uin,n),BND, element_boundary=True)
  mfinit.Assemble()
  gfu.components[1].vec.data = minit.mat.Inverse(inflowdofs) * mfinit.vec
  

# gfu.components[0].Set(proj*exu, definedon = mesh.Boundaries(".*"))
inflowdofs = BitArray(V.components[0].FreeDofs())
inflowdofs.Clear()
MarkDofsOnBBND(inflowdofs, "inflow", V.components[0], clear=False)

minit = BilinearForm(V.components[0])
minit += SymbolicBFI((V.components[0].TestFunction().Trace()*n) * (V.components[0].TrialFunction().Trace()*n),BND, element_boundary=True)
minit.Assemble()
mfinit = LinearForm(V.components[0])
mfinit += SymbolicLFI(InnerProduct(V.components[0].TestFunction().Trace(),n)*InnerProduct(uin,n),BND, element_boundary=True)
mfinit.Assemble()

mfinit2 = LinearForm(V.components[0])
mfinit2 += SymbolicLFI(InnerProduct(V.components[0].TestFunction().Trace(),n),BND, element_boundary=True)
mfinit2.Assemble()


one_at_inflow = mfinit.vec.CreateVector()
mfacet_times_one = mfinit.vec.CreateVector()

one_at_inflow.data = minit.mat.Inverse(inflowdofs) * mfinit2.vec
mfacet_times_one.data = minit.mat * one_at_inflow
  

total_length = InnerProduct(one_at_inflow,mfacet_times_one)
print(f"total length of inflow segment: {total_length}")
gfu.components[0].vec.data = minit.mat.Inverse(inflowdofs) * mfinit.vec
total_inflow = -InnerProduct(gfu.components[0].vec,mfacet_times_one)
avg_inflow = total_inflow / total_length
print(f"total inflow on inflow segment: {total_inflow}")
if False:
  print(f"total inflow correction: {1.0/avg_inflow}")
  gfu.components[0].vec.data = 1.0/avg_inflow * gfu.components[0].vec
  total_inflow = -InnerProduct(gfu.components[0].vec,mfacet_times_one)
  print(f"total inflow after correction: {total_inflow}")

velocity = gfu.components[0]
Draw(velocity, mesh = mesh, name = "velocity")

pressure = -CoefficientFunction(gfu.components[2])
Draw(pressure, mesh = mesh, name = "pressure")
Draw(velocity, mesh = mesh, name = "velocity")

gradgfu= CoefficientFunction((gfu.components[0].Operator("grad",BND),),dims=(3,3))
vort = InnerProduct(tau1,gradgfu*tau2) - InnerProduct(tau2,gradgfu*tau1)
Draw(vort,mesh,"vort")

Draw(vort, mesh = mesh, name = "vorticity")
visoptions.scalfunction = "velocity:0"

bt = V.FreeDofs()
# SetFreeDofs(V,bt)

invstokes = a.mat.Inverse(bt, inverse ="sparsecholesky")
res = gfu.vec.CreateVector()
res.data = f.vec - a.mat*gfu.vec
gfu.vec.data += invstokes * res


drag,lift,perp,forcex,forcey,forcez,forces,dp = CalcForces(gfu)
print (f"c_d = {drag:20}, c_l = {lift:20}, |dp| = {dp:20}, |f| = {forces:20}")

#velocity = CoefficientFunction(gfu.components[0])
#pressure = CoefficientFunction(gfu.components[2])
#Draw(pressure, mesh = mesh, name = "pressure")
#Draw(cf =velocity, mesh = mesh, name = "velocity")
draw_fes = SurfaceL2(mesh,order=order)
draw_vec_fes = SurfaceL2(mesh,order=order,dim=3)
draw_velocity = GridFunction(draw_vec_fes)
draw_vorticity = GridFunction(draw_fes)

draw_vorticity.Set(vort, definedon = mesh.Boundaries(".*"))
draw_velocity.Set(velocity, definedon = mesh.Boundaries(".*"))

Draw(draw_velocity, mesh = mesh, name = "draw_velocity")
Draw(draw_vorticity, mesh = mesh, name = "draw_vorticity")

Redraw()
vtk = VTKOutput(ma=mesh,coefs=[draw_velocity, pressure, draw_vorticity, deform],names=["velocity", "pressure", "vorticity", "deform"],filename="schaefer_turek_"+mapping,subdivision=2)

def VTKOutput():
  draw_vorticity.Set(vort, definedon = mesh.Boundaries(".*"))
  draw_velocity.Set(velocity, definedon = mesh.Boundaries(".*"))
  mesh.UnsetDeformation()
  vtk.Do(vb=BND)
  mesh.SetDeformation(deform)
  

VTKOutput()
# input("")

mstar = m.mat.CreateMatrix()
mstar.AsVector().data = m.mat.AsVector() + gamma * dt * a.mat.AsVector()

inv = mstar.Inverse(bt, inverse="sparsecholesky")

tend = 30
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

outfile = open("data/schaefer_turek_"+mapping+".dat","w")
outfile.write("#time drag lift perp forcex forcey forcez forces dp\n")

with TaskManager():
    while t < tend:
        if False: # IMEX1
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

        Redraw()
        
        if(t>args.vtk_start_time and k%(round(args.vtk_interval/dt))==1):
          VTKOutput()
        drag,lift,perp,forcex,forcey,forcez,forces,dp = CalcForces(gfu)
        forcesdiff = abs(sqrt(drag**2+lift**2+perp**2)/20 - sqrt(forcex**2+forcey**2+forcez**2))
        print (f"t = {t:20}, c_d = {drag:20}, c_l = {lift:20}, |dp| = {dp:20}, |f| = {forces:20}")
        outfile.write(f"{t} {drag} {lift} {perp} {forcex} {forcey} {forcez} {forces} {dp}\n")
outfile.close()
