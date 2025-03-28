from ngsolve import *
from ngsolve.solvers import Newton

class Scheme():

    def __init__(self):
        pass

    def Solve(self, data, pde):

        raise Exception('Base class method is called! Empty method!')

class SteadyExplicit(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde):

        A = BilinearForm(pde.fes)
        F = LinearForm(pde.fes)

        lhs = pde.GetLHS(data, pde.trial, pde.test)
        rhs = pde.GetRHS(data, pde.test)

        A += lhs
        F += rhs

        if hasattr(pde, 'nl'):
            if pde.nl:
                F += -1*pde.GetNL(data, pde.gfu, pde.test)

        A.Assemble()
        F.Assemble()

        pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*F.vec

class SteadyImplicit(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde):

        A = BilinearForm(pde.fes)

        lhs = pde.GetLHS(data, pde.trial, pde.test)
        rhs = pde.GetRHS(data, pde.test)

        A += lhs - rhs

        if hasattr(pde, 'nl'):
            if pde.nl:
                A += pde.GetNL(data, pde.trial, pde.test)

        Newton(A, pde.gfu, maxit=100, printing = False)

class BackwardEuler(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde):

        A = BilinearForm(pde.fes)
        F = LinearForm(pde.fes)

        if hasattr(pde, 'nl'):
            if pde.nl:
                F += -1*pde.GetNL(data, pde.gfu, pde.test)

        data.t.Set(data.t.Get() + data.dt.Get())

        lhs = pde.GetLHS(data, pde.trial, pde.test)
        rhs = pde.GetRHS(data, pde.test)
        mass = pde.GetMass(data, pde.trial, pde.test)
        mass_gfu = pde.GetMass(data, pde.gfu, pde.test)

        A += lhs + mass
        F += rhs + mass_gfu

        A.Assemble()
        F.Assemble()

        pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*F.vec

        data.t.Set(data.t.Get() - data.dt.Get())
