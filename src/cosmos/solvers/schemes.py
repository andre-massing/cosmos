from ngsolve import *
from ngsolve.solvers import Newton
from ngsolve.webgui import Draw

class Scheme():

    def __init__(self):
        pass

    def Solve(self, data, pde):

        raise Exception('Base class method is called! Empty method!')

class SteadyExplicit(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde):

        pde.PreProcess(data)

        A = BilinearForm(pde.fes)
        F = LinearForm(pde.fes)

        lhs = pde.GetLHS(data, pde.trial, pde.test)
        rhs = pde.GetRHS(data, pde.test)

        A += lhs
        F += rhs

        if hasattr(pde, 'nl'):
            if pde.nl:
                F += -1*pde.GetNL(data, pde.gfu_old_comp, pde.test)

        A.Assemble()
        F.Assemble()

        pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*F.vec

        pde.PostProcess(data)

class SteadyImplicit(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde):

        pde.PreProcess(data)

        A = BilinearForm(pde.fes)

        lhs = pde.GetLHS(data, pde.trial, pde.test)
        rhs = pde.GetRHS(data, pde.test)

        A += lhs - rhs

        if hasattr(pde, 'nl'):
            if pde.nl:
                A += pde.GetNL(data, pde.trial, pde.test)

        Newton(A, pde.gfu, maxit=100, printing = False)

        pde.PostProcess(data)

class BackwardEuler(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde):

        if hasattr(pde, 'nl'):

            pde.PreProcess(data)

            A = BilinearForm(pde.fes)

            data.t.Set(data.t.Get() + data.dt.Get())

            if pde.nl:
                A += pde.GetNL(data, pde.trial, pde.test)

            lhs = pde.GetLHS(data, pde.trial, pde.test)
            rhs = pde.GetRHS(data, pde.test)
            mass = pde.GetMass(data, pde.trial, pde.test)
            mass_gfu = pde.GetMassOld(data, pde.gfu_old_comp, pde.test)

            A += lhs + mass - rhs - mass_gfu

            Newton(A, pde.gfu, maxit=100, printing = False)

            pde.PostProcess(data)

            data.t.Set(data.t.Get() - data.dt.Get())

        else:

            pde.PreProcess(data)

            A = BilinearForm(pde.fes)
            F = LinearForm(pde.fes)

            data.t.Set(data.t.Get() + data.dt.Get())

            lhs = pde.GetLHS(data, pde.trial, pde.test)
            rhs = pde.GetRHS(data, pde.test)
            mass = pde.GetMass(data, pde.trial, pde.test)
            mass_gfu = pde.GetMassOld(data, pde.gfu_old_comp, pde.test)

            A += lhs + mass
            F += rhs + mass_gfu

            A.Assemble()
            F.Assemble()

            pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*F.vec

            pde.PostProcess(data)

            data.t.Set(data.t.Get() - data.dt.Get())

class BE_IMEX(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde):

        pde.PreProcess(data)
            
        A = BilinearForm(pde.fes)
        F = LinearForm(pde.fes)

        if hasattr(pde, 'nl'):
            if pde.nl:
                F += -1*pde.GetNL(data, pde.gfu_old_comp, pde.test)

        data.t.Set(data.t.Get() + data.dt.Get())

        lhs = pde.GetLHS(data, pde.trial, pde.test)
        rhs = pde.GetRHS(data, pde.test)
        mass = pde.GetMass(data, pde.trial, pde.test)
        mass_gfu = pde.GetMassOld(data, pde.gfu_old_comp, pde.test)

        A += lhs + mass
        F += rhs + mass_gfu

        A.Assemble()
        F.Assemble()

        pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*F.vec

        pde.PostProcess(data)

        data.t.Set(data.t.Get() - data.dt.Get())

class SteadyExplicitDispl(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde, dX_data):

        data.mesh.SetDeformation(dX_data.dX)

        pde.PreProcess(data)

        A = BilinearForm(pde.fes)
        F = LinearForm(pde.fes)

        lhs = pde.GetLHS(data, pde.trial, pde.test)
        rhs = pde.GetRHS(data, pde.test)

        A += lhs
        F += rhs

        if hasattr(pde, 'nl'):
            if pde.nl:
                F += -1*pde.GetNL(data, pde.gfu_old_comp, pde.test)

        A.Assemble()
        F.Assemble()

        pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*F.vec

        pde.PostProcess(data)

        data.mesh.UnsetDeformation()

class SteadyImplicitDispl(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde, dX_data):

        data.mesh.SetDeformation(dX_data.dX)

        pde.PreProcess(data)

        A = BilinearForm(pde.fes)

        lhs = pde.GetLHS(data, pde.trial, pde.test)
        rhs = pde.GetRHS(data, pde.test)

        A += lhs - rhs

        if hasattr(pde, 'nl'):
            if pde.nl:
                A += pde.GetNL(data, pde.trial, pde.test)

        Newton(A, pde.gfu, maxit=20, printing = False)

        pde.PostProcess(data)

        data.mesh.UnsetDeformation()


class BackwardEulerDispl(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde, dX_data):

        if hasattr(pde, 'nl'):

            pde.PreProcess(data)

            A = BilinearForm(pde.fes)
            M_old = LinearForm(pde.fes)
            F = LinearForm(pde.fes)

            M_old += pde.GetMassOld(data, pde.gfu_old_comp, pde.test)
            M_old.Assemble()
            res = M_old.vec

            data.mesh.SetDeformation(dX_data.dX)
            data.t.Set(data.t.Get() + data.dt.Get())

            F += pde.GetRHS(data, pde.test)
            F.Assemble()
            res += F.vec

            A += pde.GetLHS(data, pde.trial, pde.test)
            A += pde.GetMass(data, pde.trial, pde.test)
            if hasattr(pde, 'nl'):
                if pde.nl:
                    A += pde.GetNL(data, pde.trial, pde.test)
            
            SimpleNewtonSolve(A, res, pde.gfu)

            pde.PostProcess(data)

            data.t.Set(data.t.Get() - data.dt.Get())
            data.mesh.SetDeformation(dX_data.dX_old)

        else:

            pde.PreProcess(data)

            A = BilinearForm(pde.fes)
            M_old = LinearForm(pde.fes)
            F = LinearForm(pde.fes)

            M_old += pde.GetMassOld(data, pde.gfu_old_comp, pde.test)
            M_old.Assemble()
            res = M_old.vec

            data.mesh.SetDeformation(dX_data.dX)
            data.t.Set(data.t.Get() + data.dt.Get())

            A += pde.GetLHS(data, pde.trial, pde.test)
            A += pde.GetMass(data, pde.trial, pde.test)
            A.Assemble()
            F += pde.GetRHS(data, pde.test)
            F.Assemble()
            res += F.vec

            pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

            pde.PostProcess(data)

            data.t.Set(data.t.Get() - data.dt.Get())
            data.mesh.SetDeformation(dX_data.dX_old)

def SimpleNewtonSolve(a, f, gfu, tol=1e-13,maxits=20):
    res = gfu.vec.CreateVector()
    du = gfu.vec.CreateVector()
    fes = gfu.space
    for it in range(maxits):
        a.Apply(gfu.vec, res)
        res += -1*f
        a.AssembleLinearization(gfu.vec)
        du.data = a.mat.Inverse(fes.FreeDofs()) * res
        gfu.vec.data -= du
        #stopping criteria
        stopcritval = sqrt(abs(InnerProduct(du,res)))
        if stopcritval < tol:
            break

class BE_IMEXDispl(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde, dX_data):

        pde.PreProcess(data)

        A = BilinearForm(pde.fes)
        M_old = LinearForm(pde.fes)
        F = LinearForm(pde.fes)

        M_old += pde.GetMassOld(data, pde.gfu_old_comp, pde.test)
        if hasattr(pde, 'nl'):
            if pde.nl:
                M_old += -1*pde.GetNL(data, pde.gfu_old_comp, pde.test)
        M_old.Assemble()
        res = M_old.vec

        data.mesh.SetDeformation(dX_data.dX)
        data.t.Set(data.t.Get() + data.dt.Get())

        A += pde.GetLHS(data, pde.trial, pde.test)
        A += pde.GetMass(data, pde.trial, pde.test)
        A.Assemble()
        F += pde.GetRHS(data, pde.test)
        F.Assemble()
        res += F.vec

        pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

        pde.PostProcess(data)

        data.t.Set(data.t.Get() - data.dt.Get())
        data.mesh.SetDeformation(dX_data.dX_old)
