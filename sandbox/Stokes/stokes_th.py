# %% 
# Importing the necessary libraries
from ngsolve import *
from ngsolve.webgui import Draw

# %% Define Stokes CIP solver
def StokesTH(mesh, *, order_u, nu, f, g, filename="results/stokes_th_sol"):
    # Function spaces
    order_p = order_u - 1
    V = VectorH1(mesh, order=int(order_u))
    Q = H1(mesh, order=int(order_p), dgjumps=True)
    N = NumberSpace(mesh)
    
    # TODO: Only needed if no boundary is included
    W = V*Q*N
    (u, p, lam), (v, q, mu) = W.TnT()

    x_h = GridFunction(W)
    u_h = x_h.components[0]
    p_h = x_h.components[1]

    # Setting up output
    vtkout = VTKOutput(mesh,
                       coefs=[u_h, p_h],
                       names=["u_h", "p_h"],
                       filename=filename, subdivision=2)

    # Mesh size
    h = specialcf.mesh_size

    # Normal vector
    ns = specialcf.normal(mesh.dim)
    ts = specialcf.tangential(mesh.dim)
    nn = Cross(ns,ts)
    # normal and tangential  projection
    Qs = OuterProduct(ns, ns)
    Ps = Id(mesh.dim) - Qs
    # Weingarten map
    Bs = Grad(ns)

    # Covariant Derivative
    def Cov_h(u):
        return Ps*Grad(u.Trace())*Ps

    # Corrected Covariant Derivative
    def Cov_th(u):
        return Cov_h(u) - InnerProduct(u, ns)*Bs
    
    # Strain tensor
    def E_h(u):
        return Sym(Ps*Grad(u).Trace()*Ps)

    # Corrected strain rate tensor for non-tangential 
    def E_th(u):
        return E_h(u) - InnerProduct(u, ns)*Bs
    
    A = BilinearForm(W)

    A += nu*InnerProduct(E_th(u), E_th(v))*ds
    # Mass term
    A += InnerProduct(Ps*u, Ps*v)*ds
    # Add b_h contributions 
    A += InnerProduct(grad(q).Trace(), u)*ds
    A += InnerProduct(grad(p).Trace(), v)*ds
    
    # TODO: Only needed if no boundary is included
    # Constraint to enforce zero presure mean for boundary less surfaces
    A += (p*mu + q*lam)*ds
    
    # Penalization of normal component through bilinear form  
    # beta = Penalty parameter to weakly enforce tangential condition
    beta = 1.0
    # A += nu*beta*h**(-2.0)*InnerProduct(ns*u, ns*v)*ds
    A += nu*beta/(h*h)*InnerProduct(ns*u, ns*v)*ds

    A.Assemble()

    # Imposing manufactured solutions
    l = LinearForm(W)
    l += ( InnerProduct(Ps*f, Ps*v) - g*q)* ds 
    # l += InnerProduct(q, u_ex*nn)*ds(definedon=mesh.BBoundaries("bottom"))

    res = x_h.vec.CreateVector()

    # matrix for actual system inversion
    with TaskManager():
        l.Assemble()
        res.data = l.vec

        # u_h.Set(u_ex, definedon=mesh.BBoundaries("bottom"))
        res.data -= A.mat*x_h.vec

        x_h.vec.data += A.mat.Inverse(freedofs = W.FreeDofs()) * res.data
        
    vtkout.Do(vb=BND)
    
    return u_h, p_h