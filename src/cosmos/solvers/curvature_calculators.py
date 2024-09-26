
from ngsolve import VectorH1, NormalFacetSurface, VectorFacetSurface, GridFunction, CF, BilinearForm, LinearForm, InnerProduct, grad, ds, x, y, z, specialcf, Id, Cross, OuterProduct

__all__ = ["compute_mean_curvature_vector", "compute_stabilized_mean_curvature_vector"]

def compute_mean_curvature_vector(mesh):
    print("Computing mean curvature vector\n")
    # Get mesh curvature order to define proper function space
    order_g = mesh.GetCurveOrder()
    V = VectorH1(mesh, order=order_g)
    kappa, eta = V.TnT()
    
    Idh = GridFunction(V)
    Idh.Set(CF((x,y,z)), definedon=mesh.Boundaries(".*"))
    
    m = BilinearForm(V)
    m += InnerProduct(kappa, eta)*ds
    m.Assemble()
    
    l = LinearForm(V)
    # TODO: Determine right sign for normal vector
    l += InnerProduct(grad(Idh).Trace(), grad(eta).Trace())*ds
    l.Assemble()
    
    # Solve system
    Minv = m.mat.Inverse(V.FreeDofs(), inverse="umfpack")
    
    kappah = GridFunction(V)
    kappah.vec.data = Minv * l.vec
    
    return kappah

def compute_stabilized_mean_curvature_vector(mesh, gamma_E=0.01):
    print("Computing stabilized mean curvature vector\n")
    # Get mesh curvature order to define proper function space
    order_g = mesh.GetCurveOrder()
    V = VectorH1(mesh, order=order_g)
    dV = VectorFacetSurface(mesh, order=order_g)
    W = V*dV
    (kappa, dkappa), (eta, deta) = W.TnT()
    
    Idh = GridFunction(V)
    Idh.Set(CF((x,y,z)), definedon=mesh.Boundaries(".*"))
    
    # Compute mesh size, normal and tangential vectors 
    h = specialcf.mesh_size
    ns = specialcf.normal(mesh.dim)
    tE = specialcf.tangential(mesh.dim)
    nE = Cross(ns, tE)
    
    # Define bilinear form 
    m = BilinearForm(W)
    m += InnerProduct(kappa.Trace(), eta.Trace())*ds
    jump_dkappadn = (kappa.Trace().Deriv()*nE-dkappa.Trace())
    jump_detadn = (eta.Trace().Deriv()*nE-deta.Trace())
    # jump_dkappadn = (kappa.Trace().Deriv().trans*nE-dkappa.Trace())
    # jump_detadn = (eta.Trace().Deriv().trans*nE-deta.Trace())
    m += gamma_E*h*InnerProduct(jump_dkappadn,jump_detadn)*ds(element_boundary=True)
    
    l = LinearForm(W)
    # TODO: Determine right sign for normal vector
    l += InnerProduct(grad(Idh).Trace(), grad(eta).Trace())*ds
    
    # Assemble and solve system
    m.Assemble()
    print(f"Norm(m) = {m.mat.AsVector().Norm()}")
    l.Assemble()
    Minv = m.mat.Inverse(W.FreeDofs(), inverse="umfpack")
    
    wh = GridFunction(W)
    wh.vec.data = Minv * l.vec
    kappah, _ = wh.components
    
    return kappah