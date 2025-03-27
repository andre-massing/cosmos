from ngsolve import *
from ngsolve.solvers import *

def elastic_motion(mesh, dX, domain):

    E, nu = 210, 0.2
    # Lamé constants:
    mu  = E / 2 / (1+nu)
    lam = E * nu / ((1+nu)*(1-2*nu))

    fes = VectorH1(mesh, order=1, dirichlet='.*')
    u  = fes.TrialFunction()

    gfu = GridFunction(fes)

    gfu.Set(dX, definedon = mesh.Boundaries(domain))

    def Pow(a, b):
        return exp (log(a)*b)

    def NeoHook (C):
        return 0.5 * mu * (Trace(C-I) + 2*mu/lam * Pow(Det(C), -lam/2/mu) - 1)

    I = Id(mesh.dim)
    F = I + Grad(u)
    C = F.trans * F

    a = BilinearForm(fes, symmetric=True)
    a += Variation(  NeoHook (C).Compile() * dx)

    Newton(a, gfu, maxit = 20, printing = False)

    return gfu