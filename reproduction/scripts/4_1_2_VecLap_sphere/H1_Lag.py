from netgen.csg import *
from netgen.meshing import * 
from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions
from ngsolve import *
import re

def SolveH1Lagrange(mesh, exu, ex_du, force, order=1, condense=True, dirichlet="", label= "H1Lag"):
  
  V1 = Compress(H1(mesh,order = order, definedon=mesh.Boundaries(".*")))
  # V2 = SurfaceL2(mesh,order = order-1)
  V2 = Compress(H1(mesh,order = order, definedon=mesh.Boundaries(".*")))
  V = FESpace([V1,V1,V1, V2])

  def SetFreeDofs(V, bt):
    offset = 0
    for i in range(4):
      for el in V.components[i].Elements(BBND):
        bbnd_name = mesh.GetBBoundaries()[el.index]
        match = re.search(dirichlet, bbnd_name)
        if match != None:
          for dofs in el.dofs:            
            bt.Clear(dofs+offset)
      offset += V.components[i].ndof
          
  SetFreeDofs(V,V.FreeDofs())
  SetFreeDofs(V,V.FreeDofs(True))
 
  u1,u2,u3, p = V.TrialFunction()
  v1,v2,v3, q = V.TestFunction()
  
  u = CoefficientFunction((u1.Trace(),u2.Trace(),u3.Trace()))
  v = CoefficientFunction((v1.Trace(),v2.Trace(),v3.Trace()))
  
  normal = specialcf.normal(3)
  nmat = CoefficientFunction(normal, dims =(3,1))
  proj = Id(3) - nmat*nmat.trans
  
  grad_u = CoefficientFunction((grad(u1).Trace(),grad(u2).Trace(),grad(u3).Trace()), dims=(3,3))*proj
  grad_v = CoefficientFunction((grad(v1).Trace(),grad(v2).Trace(),grad(v3).Trace()), dims=(3,3))*proj
  eps_u = 0.5 * proj * (grad_u + grad_u.trans) * proj
  eps_v = 0.5 * proj * (grad_v + grad_v.trans) * proj
  
  a = BilinearForm(V,condense=condense)
  a += SymbolicBFI( (proj*u) * v + InnerProduct( eps_u, eps_v), BND)
  a += SymbolicBFI( u * normal * q + v * normal * p, BND)
  
  f = LinearForm(V)
  f += SymbolicLFI(proj * force * v, BND)
  
  gfu = GridFunction(V)
  with TaskManager():
      f.Assemble()
      # a.Assemble()
      # pre = Preconditioner(a,type="direct", inverse ="pardiso")
      a.Assemble()
      bt = V.FreeDofs(a.condense)
      gfu.vec[:] = 0
      print(f"Set dirichlet on {dirichlet}")
      for i in range(3):
        gfu.components[i].Set((proj*exu)[i], definedon = mesh.Boundaries(".*"))

      for i in range(V.ndof):
        if V.FreeDofs()[i]:
          gfu.vec[i] = 0
        
      r = f.vec.CreateVector()
      r.data = f.vec - a.mat * gfu.vec
  
      if a.condense:
        r.data += a.harmonic_extension_trans * r

      inv = a.mat.Inverse(bt, inverse ="pardiso")
      gfu.vec.data += inv * r
        
      if a.condense:
        gfu.vec.data += a.harmonic_extension * gfu.vec
        gfu.vec.data += a.inner_solve * r
      # BVP(a,f,gfu,pre)
      print("#############################")
  
      
  
  # res = gfu.vec.CreateVector()
  # res.data = f.vec - a.mat * gfu.vec

  # gfu.vec.data += inv * res
  # print(V.FreeDofs())
  # print(V.ndof)
  # print(gfu.vec)

  u = CoefficientFunction((gfu.components[0],gfu.components[1],gfu.components[2]))

  du = proj * CoefficientFunction((grad(gfu.components[0])[0],grad(gfu.components[0])[1],grad(gfu.components[0])[2],
                                   grad(gfu.components[1])[0],grad(gfu.components[1])[1],grad(gfu.components[1])[2],
                                   grad(gfu.components[2])[0],grad(gfu.components[2])[1],grad(gfu.components[2])[2]),
                                  dims=(3,3)) * proj
  
  Draw(cf = u, mesh = mesh, name = label + "_u")
  Draw(cf = proj*u-proj*exu, mesh = mesh, name = label + "_diffu")
  visoptions.scalfunction = label + "_u:0"

  L2Error_tang = sqrt(Integrate(InnerProduct(proj*(exu-u),exu-u),mesh,BND,order=2*order))
  L2Error_normal = sqrt(Integrate((u*normal)*(u*normal),mesh,BND,order=2*order))
  H1Error = sqrt(Integrate(InnerProduct(ex_du - du,ex_du - du),mesh,BND,order=2*order))

  nze = 0
  for e in a.mat.AsVector():
    if e != 0.0:
      nze += 1
  
  return V.ndof, sum(V.FreeDofs(True)), nze, L2Error_tang, L2Error_normal, H1Error
  
