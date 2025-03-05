from ngsolve import *
from ngsolve.solvers import *
from cosmos.solvers.tools import params_update

def elastic_motion(mesh, dX, dXold, dt, V, motions):

    if mesh.ne != 0:

        dirichlet = ''
        for params in motions:
            dirichlet = dirichlet + params['domain'] + '|'

        fes = VectorH1(mesh, order=1, dirichlet=dirichlet)
        u  = fes.TrialFunction()

        gfu = GridFunction(fes)
        gfu_m = GridFunction(fes)

        for params in motions:
            up_p = params_update(params)
            gfu_m.Set(up_p['function'], definedon = mesh.Boundaries(up_p['domain']))
            gfu.vec.data += gfu_m.vec.data

        def C(u):
            F = Grad(u) + Grad(u).trans
            return F.trans * F

        def NeoHooke (C):
            return Trace(C)

        a = BilinearForm(fes)
        a += Variation(NeoHooke(C(u)).Compile()*dx)

        Newton(a, gfu, maxit = 20)

        dX.Set(gfu)

        V.Set((dX - dXold)/dt)