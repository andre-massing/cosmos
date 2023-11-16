from ngsolve import *
from netgen.csg import *
from ngsolve.internal import visoptions

from math import pi

def Cross(u,v):
    return CoefficientFunction((u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2] ,u[0]*v[1]-u[1]*v[0]))

def SolveHDG_stokes(mesh, exu, ex_du, exp, force, nu, order=2, alpha = 10, condense = False, inverse = "umfpack"):
    normal = specialcf.normal(3)
    tangential = specialcf.tangential(3)
    n = Cross(normal,tangential)

    def tang(vec):        
        return vec - (vec*n)*n
    
    h = specialcf.mesh_size

    nmat = CoefficientFunction(normal, dims =(3,1))
    proj = Id(3) - nmat*nmat.trans

    nu = nu
    
    alpha = alpha

    VDivSurf = Discontinuous(HDivSurface(mesh, order = order), BND=True)
    VDivSurf = Compress(VDivSurf)
    VHat1 = Compress(HCurl(mesh, order = order, orderface = 0))
    VHat2 = HDivSurface(mesh, order = order, orderinner = 0)
    VHat2 = Compress(VHat2)

    Q = SurfaceL2(mesh, order = order-1)

    N = NumberSpace(mesh)

    V = FESpace([VDivSurf,VHat1,VHat2,Q,N])

    if condense:
        off = VDivSurf.ndof + VHat1.ndof+ VHat2.ndof
        for el in V.components[3].Elements(BND):
            for i in range(1,len(el.dofs)):           
                V.SetCouplingType(off+el.dofs[i], COUPLING_TYPE.LOCAL_DOF)

        V.Update()

    u, uhat1, uhat2,p, lam = V.TrialFunction()
    v, vhat1, vhat2,q, mu = V.TestFunction()

    uhat = tang(uhat1.Trace())+(uhat2.Trace()*n)*n
    vhat = tang(vhat1.Trace())+(vhat2.Trace()*n)*n

    gradu = proj * CoefficientFunction ( (grad(u),), dims=(3,3) ).trans * proj 
    gradv = proj * CoefficientFunction ( (grad(v),), dims=(3,3) ).trans * proj

    gradu = 1/2 * (gradu + gradu.trans)
    gradv = 1/2 * (gradv + gradv.trans)
    
    a = BilinearForm(V, condense = condense, symmetric = True)
    a += SymbolicBFI(nu * InnerProduct(gradu, gradv), BND)
    a += SymbolicBFI (nu *  InnerProduct ( gradu * n,  (vhat-v.Trace()) ), BND, element_boundary=True )
    a += SymbolicBFI (nu *  InnerProduct ( gradv * n,  (uhat-u.Trace()) ), BND, element_boundary=True )
    a += SymbolicBFI (nu *  4*(order+1)**2/h * InnerProduct ( (vhat-v.Trace()),  (uhat-u.Trace()) ) ,BND, element_boundary=True )
    a += SymbolicBFI (  -div(u.Trace()) *q.Trace(), BND)
    a += SymbolicBFI (  -div(v.Trace()) *p.Trace(), BND)
    a += SymbolicBFI (  (u.Trace() - uhat)*n * q.Trace(), BND, element_boundary=True)
    a += SymbolicBFI (  (v.Trace() - vhat)*n * p.Trace(), BND, element_boundary=True)

    a += SymbolicBFI (  1*(mu.Trace()*p.Trace() + lam.Trace()*q.Trace()), BND)

    a.Assemble()

    f = LinearForm(V)
    f += SymbolicLFI( force * v.Trace(),BND, bonus_intorder = 2*order)
    f.Assemble()
    
    def SetFreeDofs(bt):                     
        for el in V.components[1].Elements(BBND):
            for dofs in el.dofs:            
                V.FreeDofs().Clear(dofs+V.components[0].ndof)
                V.FreeDofs(True).Clear(dofs+V.components[0].ndof)
        for el in V.components[2].Elements(BBND):
            for dofs in el.dofs:            
                V.FreeDofs().Clear(dofs+V.components[0].ndof+V.components[1].ndof)
                V.FreeDofs(True).Clear(dofs+V.components[0].ndof+V.components[1].ndof)
    
    SetFreeDofs(V)
    gfu = GridFunction(V)
   
    if condense:        
        inv_ii = a.mat.Inverse(V.FreeDofs(True), inverse =inverse)
        ext = IdentityMatrix(V.ndof) + a.harmonic_extension
        extT = IdentityMatrix(V.ndof) + a.harmonic_extension_trans
        inv = ext @ inv_ii @ extT + a.inner_solve
    else:
        inv = a.mat.Inverse(V.FreeDofs(), inverse =inverse)
        
    gfu.vec.data =  inv * f.vec

    velocity = CoefficientFunction(gfu.components[0])
    pressure = CoefficientFunction(gfu.components[3])
    #Draw(cf = pressure, mesh = mesh, name = "pressure")
    #Draw(cf =velocity, mesh = mesh, name = "velocity")
    #visoptions.scalfunction = "velocity:0"

    D_u = proj * CoefficientFunction(gfu.components[0].Operator("grad", BND), dims=(3,3)).trans * proj
    eps_u = 1/2 * (D_u + D_u.trans)

    diff_D = eps_u - ex_du 
    diff_u = gfu.components[0] - exu
    err_D = sqrt(Integrate(InnerProduct(diff_D,diff_D), mesh, BND,order=2*order))
    err_u = sqrt(Integrate(InnerProduct(diff_u,diff_u), mesh, BND,order=2*order))
    err_p = sqrt(Integrate((pressure - exp)*(pressure - exp), mesh, BND,order=2*order))
    err_divu = sqrt(Integrate(div(gfu.components[0])*div(gfu.components[0]), mesh, BND,order=2*order))

    return V.ndof, err_D, err_u, err_divu, err_p
