from netgen.csg import *
from ngsolve.internal import visoptions
from netgen.meshing import *
from ngsolve import *

from math import pi
import re

def SolveHDG(mesh, exu, ex_du, force, order=1, alpha=40, condense=True, dirichlet="", label= "HDG"):
  normal = specialcf.normal(3)
  tangential = specialcf.tangential(3)
  n = Cross(normal,tangential)
  
  VDivSurf = Discontinuous(HDivSurface(mesh, order = order), BND=True)
  VDivSurf = Compress(VDivSurf)
  
  # VHat1 = TangentialFacetFESpace(mesh, order = order, dirichlet=dirichlet)
  # VHat1 = Compress(HCurl(mesh, order = order, orderface = 0, dirichlet=dirichlet))
  VHat1 = Compress(HCurl(mesh, order = order, orderface = 0, dirichlet=dirichlet))
  # VHat1 = HCurl(mesh, order = order, orderface = 0, dirichlet=dirichlet)
  # VHat2 = HCurl(mesh, order = order, flags={"orderface": 0})
  # VHat2b = HDivSurface(mesh, order = order)
  VHat2 = HDivSurface(mesh, order = order, orderinner = 0, dirichlet=dirichlet)
  VHat2 = Compress(VHat2)
  # print(VHat2.ndof,VHat2b.ndof)
  # input("")

  # print(f"VHat1.FreeDofs() = {VHat1.FreeDofs()}")
  # print(f"VHat2.FreeDofs() = {VHat2.FreeDofs()}")
  
  # VHat2 = HCurl(mesh, order = order, flags={"orderface": 0})
  
  print(f"{VDivSurf.ndof} {VHat1.ndof} {VHat2.ndof}")
  
  V = FESpace([VDivSurf,VHat1,VHat2])

  def SetFreeDofs(V, bt):
    for el in V.components[1].Elements(BBND):
      bbnd_name = mesh.GetBBoundaries()[el.index]
      match = re.search(dirichlet, bbnd_name)
      if match != None:
        for dofs in el.dofs:            
          bt.Clear(dofs+V.components[0].ndof)
        
    for el in V.components[2].Elements(BBND):
      bbnd_name = mesh.GetBBoundaries()[el.index]
      match = re.search(dirichlet, bbnd_name)
      if match != None:
        for dofs in el.dofs:            
          bt.Clear(dofs+V.components[1].ndof)
          
  SetFreeDofs(V,V.FreeDofs())
  SetFreeDofs(V,V.FreeDofs(True))
  
  u, uhat1, uhat2 = V.TrialFunction()
  v, vhat1, vhat2 = V.TestFunction()
  
  def tang(vec):
      return vec - (vec*n)*n
  
  # uhat = tang(uhat1.Trace())+Cross(normal,tang(uhat2.Trace()))
  # vhat = tang(vhat1.Trace())+Cross(normal,tang(vhat2.Trace()))
  
  uhat = tang(uhat1.Trace())+(uhat2.Trace()*n)*n
  vhat = tang(vhat1.Trace())+(vhat2.Trace()*n)*n
  
  h = specialcf.mesh_size
  
  gradu = CoefficientFunction ( (grad(u),), dims=(3,3) ).trans
  gradv = CoefficientFunction ( (grad(v),), dims=(3,3) ).trans
  
  nmat = CoefficientFunction(normal, dims =(3,1))
  proj = Id(3) - nmat*nmat.trans
  
  #gradu is really gradu!!!! there is a transpose above
  gradu = 1/2 * (proj * gradu + gradu.trans * proj)
  gradv = 1/2 * (proj * gradv + gradv.trans * proj)
  
  
  a = BilinearForm(V, condense=condense)
  a += SymbolicBFI(InnerProduct(gradu, gradv), BND)
  a += SymbolicBFI( u.Trace() * v.Trace() , BND)
  a += SymbolicBFI ( InnerProduct ( gradu * n,  vhat-v.Trace() ), BND, element_boundary=True )
  a += SymbolicBFI ( InnerProduct ( gradv * n,  uhat-u.Trace() ), BND, element_boundary=True )
  a += SymbolicBFI ( alpha*(order+1)**2/h * InnerProduct ( vhat-v.Trace(),  uhat-u.Trace() ) ,BND, element_boundary=True )
  
  
  f = LinearForm(V)
  f += SymbolicLFI( proj * force * v.Trace(),BND)
  
  gfu = GridFunction(V)
  
  with TaskManager():
    print("#############################")
    print("finished assembling a")
    f.Assemble()
    print("finished assembling f")
    # pre = Preconditioner(a,type="direct", inverse ="sparsecholesky")
    a.Assemble()
    
    print("#############################")
    print("solve system")
    bt = V.FreeDofs(a.condense)
    
    # gfu.vec.data = a.mat.Inverse(bt, inverse ="sparsecholesky") * f.vec
    gfu.vec[:] = 0

    print(f"Set dirichlet on {dirichlet}")
    
    # gfu.components[1].Set(CoefficientFunction((1,1,1)), definedon = mesh.Boundaries(".*"))
    # gfu.components[2].Set(CoefficientFunction((1,1,1)), definedon = mesh.Boundaries(".*"))
    # gfu.components[1].Set(proj*exu, definedon = mesh.Boundaries(dirichlet))
    # gfu.components[2].Set(proj*exu, definedon = mesh.Boundaries(dirichlet))

    m = BilinearForm(V.components[1])
    m += SymbolicBFI((V.components[1].TestFunction().Trace()*tangential) * (V.components[1].TrialFunction().Trace()*tangential),BND, element_boundary=True)
    m.Assemble()
    mf = LinearForm(V.components[1])
    mf += SymbolicLFI(InnerProduct(V.components[1].TestFunction().Trace(),tangential)*InnerProduct(proj*exu,tangential),BND, element_boundary=True)
    mf.Assemble()
    gfu.components[1].vec.data = m.mat.Inverse() * mf.vec
    
    # print(gfu.components[1].vec)
    # # gfu.components[1].Set(proj*exu, definedon = mesh.Boundaries(".*"))
    # # print(gfu.components[1].vec)
    # Draw(gfu.components[1],mesh,"gfu1")
    # input("")
    # exit()
    # m = BilinearForm(V.components[2])
    # m += V.components[2].TestFunction().Trace() * V.components[2].TrialFunction().Trace() * ds
    # m.Assemble()
    # mf = LinearForm(V.components[2])
    # mf += InnerProduct(V.components[2].TestFunction().Trace(),proj*exu) * ds
    # mf.Assemble()
    # gfu.components[2].vec.data = m.mat.Inverse() * mf.vec
    
    # print(gfu.components[2].vec)
    gfu.components[2].Set(proj*exu, definedon = mesh.Boundaries(".*"))
    # print(gfu.components[2].vec)

    # input("")
    # exit()

    # Draw(gfu.components[1],mesh,"curli")
    # input("")
    # exit()

    # print(gfu.components[1].vec)
    # Set interior bubbles to zero:
    for i in range(V.ndof):
      if V.FreeDofs()[i]:
        gfu.vec[i] = 0
    
    # print(gfu.components[1].vec)
    # print(gfu.components[2].vec)
    # Draw(cf = gfu.components[1], mesh = mesh, name = label + "UCURL")
    # Draw(cf = gfu.components[2], mesh = mesh, name = label + "UDIV")
    # input("")
    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * gfu.vec

    if a.condense:
      r.data += a.harmonic_extension_trans * r
    gfu.vec.data += a.mat.Inverse(bt, inverse ="sparsecholesky") * r
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
  
