from ngsolve import *
from netgen.csg import *
from ngsolve.internal import visoptions

from math import pi

def Cross(u,v):
    return CoefficientFunction((u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2] ,u[0]*v[1]-u[1]*v[0]))

def SolveHdivHDG_stokes(mesh, exu, ex_du, exp, force, nu, order=2, alpha = 10, condense = False,inverse = "umfpack"):
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

    VDivSurf = HDivSurface(mesh, order = order)
    VDivSurf = Compress(VDivSurf)
    VHat = HCurl(mesh, order = order, orderface = 0)
    VHat = Compress(VHat)
    Q = SurfaceL2(mesh, order = order-1)
    N = NumberSpace(mesh)

    V = FESpace([VDivSurf,VHat,Q, N])
        
    if condense:
        off = VDivSurf.ndof + VHat.ndof
        for el in V.components[2].Elements(BND):
            for i in range(1,len(el.dofs)):           
                V.SetCouplingType(off+el.dofs[i], COUPLING_TYPE.LOCAL_DOF)

        V.Update()
        
    u, uhat,p, lam = V.TrialFunction()
    v, vhat,q, mu = V.TestFunction()


    gradu = proj * CoefficientFunction ( (grad(u),), dims=(3,3) ).trans * proj 
    gradv = proj * CoefficientFunction ( (grad(v),), dims=(3,3) ).trans * proj

    gradu = 1/2 * (gradu + gradu.trans)
    gradv = 1/2 * (gradv + gradv.trans)

    vhat = vhat.Trace()
    uhat = uhat.Trace()
    
    a = BilinearForm(V, condense = condense, symmetric = True)
    a += SymbolicBFI(nu * InnerProduct(gradu, gradv), BND)
    a += SymbolicBFI (nu *  InnerProduct ( gradu * n,  tang(vhat-v.Trace()) ), BND, element_boundary=True )
    a += SymbolicBFI (nu *  InnerProduct ( gradv * n,  tang(uhat-u.Trace()) ), BND, element_boundary=True )
    a += SymbolicBFI (nu *  4*(order+1)**2/h * InnerProduct ( tang(vhat-v.Trace()),  tang(uhat-u.Trace()) ) ,BND, element_boundary=True )
    a += SymbolicBFI (  -div(u.Trace()) *q.Trace(), BND)
    a += SymbolicBFI (  -div(v.Trace()) *p.Trace(), BND)
    a += SymbolicBFI (  1*(mu.Trace()*p.Trace() + lam.Trace()*q.Trace()), BND)

    a.Assemble()
    
    f = LinearForm(V)
    f += SymbolicLFI( force * v.Trace(),BND, bonus_intorder = order)
    f.Assemble()

    def SetFreeDofs(V):
        for el in V.components[0].Elements(BBND):
            for dofs in el.dofs:            
                V.FreeDofs().Clear(dofs)
                V.FreeDofs(True).Clear(dofs)
        for el in V.components[1].Elements(BBND):
            for dofs in el.dofs:   
                V.FreeDofs().Clear(dofs+V.components[0].ndof)
                V.FreeDofs(True).Clear(dofs+V.components[0].ndof)
    
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
    pressure = CoefficientFunction(gfu.components[2])
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
