from netgen.csg import *
from ngsolve.internal import visoptions
from netgen.meshing import *
from ngsolve import *

from math import pi
import re

def SolveHdivHDG(mesh, exu, ex_du, force, order=1, alpha=40, condense=True, dirichlet="", label= "HDG"):
  normal = specialcf.normal(3)
  tangential = specialcf.tangential(3)
  n = Cross(normal,tangential)
  
  VDivSurf = HDivSurface(mesh, order = order)
  # print(VDivSurf.ndof)
  VDivSurf = Compress(VDivSurf)
  # print(VDivSurf.ndof)
  VHat1 = HCurl(mesh, order = order, orderface = 0)
  # print(VHat1.ndof)
  VHat1 = Compress(VHat1)
  # print(VHat1.ndof)
  # VHat2 = HCurl(mesh, order = order, flags={"orderface": 0})
  
  # print(f"{VDivSurf.ndof} {VHat1.ndof} -- ")
  
  V = FESpace([VDivSurf,VHat1])
  
  def SetFreeDofs(V):
    for el in V.components[0].Elements(BBND):
      bbnd_name = mesh.GetBBoundaries()[el.index]
      match = re.search(dirichlet, bbnd_name)
      if match != None:
        for dofs in el.dofs:            
          V.FreeDofs().Clear(dofs)
          V.FreeDofs(True).Clear(dofs)
          # V.components[0].FreeDofs().Clear(dofs)
          # V.components[0].FreeDofs(True).Clear(dofs)
        
    for el in V.components[1].Elements(BBND):
      bbnd_name = mesh.GetBBoundaries()[el.index]
      match = re.search(dirichlet, bbnd_name)
      if match != None:
        for dofs in el.dofs:            
          V.FreeDofs().Clear(dofs+V.components[0].ndof)
          V.FreeDofs(True).Clear(dofs+V.components[0].ndof)
          # V.components[1].FreeDofs().Clear(dofs)
          # V.components[1].FreeDofs(True).Clear(dofs)
          
  SetFreeDofs(V)

  u, uhat1 = V.TrialFunction()
  v, vhat1 = V.TestFunction()
  
  def tang(vec):
      return vec - (vec*n)*n
  
  uhat = uhat1.Trace()
  vhat = vhat1.Trace()
  
  h = specialcf.mesh_size
  
  gradu = CoefficientFunction ( (grad(u),), dims=(3,3) ).trans
  gradv = CoefficientFunction ( (grad(v),), dims=(3,3) ).trans
  
  nmat = CoefficientFunction(normal, dims =(3,1))
  proj = Id(3) - nmat*nmat.trans
  
  #gradu is really gradu!!!! there is a transpose above
  gradu = 1/2 * (proj * gradu + gradu.trans * proj)
  gradv = 1/2 * (proj * gradv + gradv.trans * proj)

  # for i in range(V.ndof):
  #   print(V.CouplingType(i))
    
  a = BilinearForm(V, condense=condense)
  a += SymbolicBFI(InnerProduct(gradu, gradv), BND)
  a += SymbolicBFI( u.Trace() * v.Trace() , BND)
  a += SymbolicBFI ( InnerProduct ( gradu * n,  tang(vhat-v.Trace()) ), BND, element_boundary=True )
  a += SymbolicBFI ( InnerProduct ( gradv * n,  tang(uhat-u.Trace()) ), BND, element_boundary=True )
  a += SymbolicBFI ( alpha*(order+1)**2/h * InnerProduct ( tang(vhat-v.Trace()),  tang(uhat-u.Trace()) ) ,BND, element_boundary=True )
  
  f = LinearForm(V)
  f += SymbolicLFI( proj * force * v.Trace(),BND)
  
  gfu = GridFunction(V)
  
  with TaskManager():
    print("#############################")
    f.Assemble()
    print("finished assembling f")
    # pre = Preconditioner(a,type="direct", inverse ="sparsecholesky")
    a.Assemble()
    print("finished assembling a")
    print("#############################")
    print("solve system")
    # bt = V.FreeDofs()
    # print(bt)
    # bt = 
    # print(bt)
    # print(a.mat)
    # input("")
    # BVP(a,f,gfu,pre)
    
    gfu.vec[:] = 0

    print(f"Set dirichlet on {dirichlet}")

    gfu.components[0].Set(proj*exu, definedon = mesh.Boundaries(".*"))


    m = BilinearForm(V.components[1])
    m += SymbolicBFI((V.components[1].TestFunction().Trace()*tangential) * (V.components[1].TrialFunction().Trace()*tangential),BND, element_boundary=True)
    m.Assemble()
    mf = LinearForm(V.components[1])
    mf += SymbolicLFI(InnerProduct(V.components[1].TestFunction().Trace(),tangential)*InnerProduct(proj*exu,tangential),BND, element_boundary=True)
    mf.Assemble()
    gfu.components[1].vec.data = m.mat.Inverse() * mf.vec
    
    # gfu.components[1].Set(proj*exu, definedon = mesh.Boundaries(".*"))

    # Set interior bubbles to zero:
    for i in range(V.ndof):
      if V.FreeDofs()[i]:
        gfu.vec[i] = 0
    
    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * gfu.vec

    if a.condense:
      r.data += a.harmonic_extension_trans * r
    gfu.vec.data += a.mat.Inverse(V.FreeDofs(a.condense), inverse ="pardiso") * r
    if a.condense:
      gfu.vec.data += a.harmonic_extension * gfu.vec
      gfu.vec.data += a.inner_solve * r
    # BVP(a,f,gfu,pre)
    print("#############################")
    
  
  du = proj * CoefficientFunction(gfu.components[0].Operator("grad", BND), dims=(3,3)).trans * proj
  u = gfu.components[0]
  
  Draw(cf = u, mesh = mesh, name = label + "_u")
  Draw(cf = proj*u-proj*exu, mesh = mesh, name = label + "_diffu")
  visoptions.scalfunction = label + "_u:0"

  L2Error_tang = sqrt(Integrate(InnerProduct(proj*(exu-u),exu-u),mesh,BND,order=2*order))
  L2Error_normal = sqrt(Integrate((u*normal)*(u*normal),mesh,BND,order=2*order))
  H1Error = sqrt(Integrate(InnerProduct(ex_du - du,ex_du - du),mesh,BND,order=2*order))

  return V.ndof, sum(V.FreeDofs(True)), a.mat.nze, L2Error_tang, L2Error_normal, H1Error
  
