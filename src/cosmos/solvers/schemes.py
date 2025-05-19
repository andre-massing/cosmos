from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.solvers.solver_base import SolverALE

class Scheme():

    def __init__(self):
        pass

    def Solve(self, data, pde):

        raise Exception('Base class method is called! Empty method!')
    
def SimpleNewtonSolve(data, a, f, gfu, tol=1e-6, maxits=20):
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

    if it == maxits-1:
        print(stopcritval)
        raise Exception('Newton solver has not converged!')

class Steady(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde, ale):

        pde.PreProcess(data, ale)

        A = BilinearForm(pde.fes)
        A += pde.GetLHS(data, pde.get_trial(), pde.get_test(), ale)
        
        F = LinearForm(pde.fes)
        F += pde.GetRHS(data, pde.get_test(), ale)
        F.Assemble()
        res = F.vec

        if pde.nonlinear:

            A += pde.GetNL(data, pde.get_trial(), pde.get_test(), ale)
            SimpleNewtonSolve(data, A, res, pde.gfu)

        else:

            A.Assemble()
            pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

        pde.PostProcess(data, ale)

class BDF1(Scheme):

    def __init__(self, conservative = True):
        
        super().__init__()
        self.conservative = conservative

    def Solve(self, data, pde, ale):

        ale1 = SolverALE(data.mesh)

        if self.conservative:

            ale1.deformation.vec.data = ale.prev_d[-1].data
            ale1.velocity.vec.data = ale.prev_v[-1].data

            pde.PreProcess(data, ale1)

            M1 = BilinearForm(pde.fes)
            M1 += pde.GetMass(data, pde.get_trial(), pde.get_test(), ale1)
            M1.Assemble()
            mass_old_vec = M1.mat*pde.prev_gfu[-1]

            data.t.Set(data.t.Get() + data.dt.Get())

        else:

            data.t.Set(data.t.Get() + data.dt.Get())

            pde.PreProcess(data, ale)

            M1 = BilinearForm(pde.fes)
            M1 += pde.GetMass(data, pde.get_trial(), pde.get_test(), ale)
            M1.Assemble()
            mass_old_vec = M1.mat*pde.prev_gfu[-1]

        F = LinearForm(pde.fes)
        F += pde.GetRHS(data, pde.get_test(), ale)
        F.Assemble()
        res = F.vec
        res += mass_old_vec

        A = BilinearForm(pde.fes)
        lhs = pde.GetLHS(data, pde.get_trial(), pde.get_test(), ale)
        mass = pde.GetMass(data, pde.get_trial(), pde.get_test(), ale)
        A += lhs + mass

        if pde.nonlinear:

            A += pde.GetNL(data, pde.get_trial(), pde.get_test(), ale)
            SimpleNewtonSolve(data, A, res, pde.gfu)

        else:

            A.Assemble()
            pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

        data.t.Set(data.t.Get() - data.dt.Get())

        pde.PostProcess(data, ale)

class BDF2(Scheme):

    def __init__(self, conservative = True):
        
        super().__init__()
        self.conservative = conservative

    def Solve(self, data, pde, ale):

        if len(pde.prev_gfu) < 1:

            scheme = BDF1(self.conservative)
            scheme.Solve(data, pde, ale)

        else:

            pde.PreProcess(data, ale)

            if self.conservative:

                data.t.Set(data.t.Get() - data.prev_dt[-2])

                ale1 = SolverALE(data.mesh)
                ale1.deformation.vec.data = ale.prev_d[-2].data
                ale1.velocity.vec.data = ale.prev_v[-2].data
                M2 = BilinearForm(pde.fes)
                M2 += pde.GetMass(data, pde.get_trial(), pde.get_test(), ale1)
                M2.Assemble()
                mass_old_vec = -0.5*M2.mat*pde.prev_gfu[-2]

                data.t.Set(data.t.Get() + data.prev_dt[-2])

                ale1.deformation.vec.data = ale.prev_d[-1].data
                ale1.velocity.vec.data = ale.prev_v[-1].data
                M1 = BilinearForm(pde.fes)
                M1 += pde.GetMass(data, pde.get_trial(), pde.get_test(), ale1)
                M1.Assemble()
                mass_old_vec += 2*M1.mat*pde.prev_gfu[-1]

                data.t.Set(data.t.Get() + data.dt.Get())

                mass = 1.5*pde.GetMass(data, pde.get_trial(), pde.get_test(), ale)

            else:

                data.t.Set(data.t.Get() + data.dt.Get())
                M = BilinearForm(pde.fes)
                M += pde.GetMass(data, pde.get_trial(), pde.get_test(), ale)
                M.Assemble()

                mass_old_vec = 2*M.mat*pde.prev_gfu[-1]
                mass_old_vec += -0.5*M.mat*pde.prev_gfu[-2]

                mass = 1.5*pde.GetMass(data, pde.get_trial(), pde.get_test(), ale)

            A = BilinearForm(pde.fes)
            lhs = pde.GetLHS(data, pde.get_trial(), pde.get_test(), ale)
            A += lhs + mass

            F = LinearForm(pde.fes)
            F += pde.GetRHS(data, pde.get_test(), ale)
            F.Assemble()
            res = F.vec
            res += mass_old_vec

            if pde.nonlinear:
                
                A += pde.GetNL(data, pde.get_trial(), pde.get_test(),ale)
                SimpleNewtonSolve(data, A, res, pde.gfu)

            else:

                A.Assemble()
                pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

            data.t.Set(data.t.Get() - data.dt.Get())

            pde.PostProcess(data, ale)



class CN(Scheme):

    def __init__(self, conservative = True):
        
        super().__init__()
        self.conservative = conservative

    def Solve(self, data, pde, ale):

        pde.PreProcess(data, ale)

        ale1 = SolverALE(data.mesh)
        ale1.deformation.vec.data = ale.prev_d[-1].data
        ale1.velocity.vec.data = ale.prev_v[-1].data

        ale_half = SolverALE(data.mesh)
        ale_half.deformation.vec.data = 0.5*(ale.prev_d[-1].data + ale.deformation.vec.data)
        ale_half.velocity.vec.data = 0.5*(ale.prev_v[-1].data + ale.velocity.vec.data)

        if self.conservative:

            res1 = pde.gfu.vec.CreateVector()
            A1 = BilinearForm(pde.fes)
            A1 += pde.GetLHS(data, pde.get_trial(), pde.get_test(), ale1)
            if pde.nonlinear:
                A1 += pde.GetNL(data, pde.get_trial(), pde.get_test(), ale1)
            A1.Apply(pde.prev_gfu[-1], res1)
            res = -0.5*res1

            F1 = LinearForm(pde.fes)
            F1 += pde.GetRHS(data, pde.get_test(), ale1)
            F1.Assemble()
            res += 0.5*F1.vec

            M = BilinearForm(pde.fes)
            M += pde.GetMass(data, pde.get_trial(), pde.get_test(), ale1)
            M.Assemble()
            res += M.mat*pde.prev_gfu[-1]

        else:

            res1 = pde.gfu.vec.CreateVector()
            A1 = BilinearForm(pde.fes)
            A1 += pde.GetLHS(data, pde.get_trial(), pde.get_test(), ale)
            if pde.nonlinear:
                A1 += pde.GetNL(data, pde.get_trial(), pde.get_test(), ale)
            A1.Apply(pde.prev_gfu[-1], res1)
            res = -0.5*res1

            F1 = LinearForm(pde.fes)
            F1 += pde.GetRHS(data, pde.get_test(), ale)
            F1.Assemble()
            res += 0.5*F1.vec

            M = BilinearForm(pde.fes)
            M += pde.GetMass(data, pde.get_trial(), pde.get_test(), ale_half)
            M.Assemble()
            res += M.mat*pde.prev_gfu[-1]

        data.t.Set(data.t.Get() + data.dt.Get())

        A = BilinearForm(pde.fes)
        lhs = pde.GetLHS(data, pde.get_trial(), pde.get_test(), ale)

        if self.conservative:
            mass = pde.GetMass(data, pde.get_trial(), pde.get_test(), ale)
        else:
            mass = pde.GetMass(data, pde.get_trial(), pde.get_test(), ale_half)
        A += 0.5*lhs + mass

        F = LinearForm(pde.fes)
        rhs = pde.GetRHS(data, pde.get_test(), ale)
        F += rhs
        F.Assemble()
        res += 0.5*F.vec

        if pde.nonlinear:

            A += 0.5*pde.GetNL(data, pde.get_trial(), pde.get_test(), ale)
            SimpleNewtonSolve(data, A, res, pde.gfu)

        else:

            A.Assemble()
            pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

        data.t.Set(data.t.Get() - data.dt.Get())

        pde.PostProcess(data, ale)