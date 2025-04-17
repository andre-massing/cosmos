from ngsolve import *
from ngsolve.solvers import Newton
from ngsolve.webgui import Draw

class Scheme():

    def __init__(self):
        pass

    def Solve(self, data, pde):

        raise Exception('Base class method is called! Empty method!')
    
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

class Steady(Scheme):

    def __init__(self):
        super().__init__()

    def Solve(self, data, pde):

        pde.PreProcess(data)

        A = BilinearForm(pde.fes)
        A += pde.GetLHS(data, pde.get_trial(), pde.get_test(), dX = data.dX)
        
        F = LinearForm(pde.fes)
        F += pde.GetRHS(data, pde.get_test(), dX = data.dX)
        F.Assemble()
        res = F.vec

        if pde.nonlinear:

            A += pde.GetNL(data, pde.get_trial(), pde.get_test(), dX = data.dX)
            SimpleNewtonSolve(A, res, pde.gfu)

        else:

            A.Assemble()
            pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

        pde.PostProcess(data)

class BDF1(Scheme):

    def __init__(self, conservative = True):
        
        super().__init__()
        self.conservative = conservative

    def Solve(self, data, pde):

        if self.conservative:

            gfu = GridFunction(data.dX.space)
            gfu.vec.data = data.prev_dX[-1].data

            pde.PreProcess(data, dX = gfu)

            M1 = BilinearForm(pde.fes)
            M1 += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = gfu)
            M1.Assemble()
            mass_old_vec = M1.mat*pde.prev_gfu[-1]

            data.t.Set(data.t.Get() + data.dt.Get())

        else:

            data.t.Set(data.t.Get() + data.dt.Get())

            pde.PreProcess(data, dX = data.dX)

            M1 = BilinearForm(pde.fes)
            M1 += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = data.dX)
            M1.Assemble()
            mass_old_vec = M1.mat*pde.prev_gfu[-1]

        F = LinearForm(pde.fes)
        F += pde.GetRHS(data, pde.get_test(), dX = data.dX)
        F.Assemble()
        res = F.vec
        res += mass_old_vec

        A = BilinearForm(pde.fes)
        lhs = pde.GetLHS(data, pde.get_trial(), pde.get_test(), dX = data.dX)
        mass = pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = data.dX)
        A += lhs + mass

        if pde.nonlinear:

            A += pde.GetNL(data, pde.get_trial(), pde.get_test(), dX = data.dX)
            SimpleNewtonSolve(A, res, pde.gfu)

        else:

            A.Assemble()
            pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

        data.t.Set(data.t.Get() - data.dt.Get())

        pde.PostProcess(data, dX = data.dX)

class BDF1imex(Scheme):

    def __init__(self, conservative = True):
        
        super().__init__()
        self.conservative = conservative

    def Solve(self, data, pde):

        gfu = GridFunction(data.dX.space)
        gfu.vec.data = data.prev_dX[-1].data

        pde.PreProcess(data, dX = gfu)

        if self.conservative:

            M1 = BilinearForm(pde.fes)
            M1 += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = gfu)
            M1.Assemble()
            mass_old_vec = M1.mat*pde.prev_gfu[-1]

            data.t.Set(data.t.Get() + data.dt.Get())

        else:

            data.t.Set(data.t.Get() + data.dt.Get())

            M1 = BilinearForm(pde.fes)
            M1 += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = gfu)
            M1.Assemble()
            mass_old_vec = M1.mat*pde.prev_gfu[-1]

        F = LinearForm(pde.fes)
        F += pde.GetRHS(data, pde.get_test(), dX = gfu)
        F.Assemble()
        res = F.vec

        if pde.nonlinear:

            N1 = BilinearForm(pde.fes)
            N1 += pde.GetNL(data, pde.get_trial(), pde.get_test(), dX = gfu)
            nonlin = pde.gfu.vec.CreateVector()
            N1.Apply(pde.prev_gfu[-1], nonlin)
            res += -1*nonlin

        res += mass_old_vec

        A = BilinearForm(pde.fes)
        lhs = pde.GetLHS(data, pde.get_trial(), pde.get_test(), dX = gfu)
        mass = pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = gfu)
        A += lhs + mass

        A.Assemble()
        pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

        data.t.Set(data.t.Get() - data.dt.Get()) 

        pde.PostProcess(data, dX = gfu)

class BDF2(Scheme):

    def __init__(self, conservative = True):
        
        super().__init__()
        self.conservative = conservative

    def Solve(self, data, pde):

        if len(pde.prev_gfu) < 1:

            scheme = BDF1(self.conservative)
            scheme.Solve(data, pde)

        else:

            pde.PreProcess(data, dX = data.dX)

            if self.conservative:

                data.t.Set(data.t.Get() - data.prev_dt[-2])

                gfu2 = GridFunction(data.dX.space)
                gfu2.vec.data = data.prev_dX[-2].data
                M2 = BilinearForm(pde.fes)
                M2 += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = gfu2)
                M2.Assemble()
                mass_old_vec = -0.5*M2.mat*pde.prev_gfu[-2]

                data.t.Set(data.t.Get() + data.prev_dt[-2])

                gfu1 = GridFunction(data.dX.space)
                gfu1.vec.data = data.prev_dX[-1].data
                M1 = BilinearForm(pde.fes)
                M1 += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = gfu1)
                M1.Assemble()
                mass_old_vec += 2*M1.mat*pde.prev_gfu[-1]

                data.t.Set(data.t.Get() + data.dt.Get())

                mass = 1.5*pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = data.dX)

            else:

                data.t.Set(data.t.Get() + data.dt.Get())
                M = BilinearForm(pde.fes)
                M += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = data.dX)
                M.Assemble()

                mass_old_vec = 2*M.mat*pde.prev_gfu[-1]
                mass_old_vec += -0.5*M.mat*pde.prev_gfu[-2]

                mass = 1.5*pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = data.dX)

            A = BilinearForm(pde.fes)
            lhs = pde.GetLHS(data, pde.get_trial(), pde.get_test(), dX = data.dX)
            A += lhs + mass

            F = LinearForm(pde.fes)
            F += pde.GetRHS(data, pde.get_test(), dX = data.dX)
            F.Assemble()
            res = F.vec
            res += mass_old_vec

            if pde.nonlinear:
                
                A += pde.GetNL(data, pde.get_trial(), pde.get_test(), dX = data.dX)
                SimpleNewtonSolve(A, res, pde.gfu)

            else:

                A.Assemble()
                pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

            data.t.Set(data.t.Get() - data.dt.Get())

            pde.PostProcess(data, dX = data.dX)

class BDF2imex(Scheme):

    def __init__(self, conservative = True):
        
        super().__init__()
        self.conservative = conservative

    def Solve(self, data, pde):

        if len(pde.prev_gfu) < 1:

            scheme = BDF1imex(self.conservative)
            scheme.Solve(data, pde)

        else:

            dX_extr = GridFunction(data.dX.space)
            dX_extr.vec.data = self.Extrapolate([data.prev_dX[-2], data.prev_dX[-1]]).data

            pde.PreProcess(data, dX = dX_extr)

            if self.conservative:

                data.t.Set(data.t.Get() - data.prev_dt[-2])

                gfu2 = GridFunction(data.dX.space)
                gfu2.vec.data = data.prev_dX[-2].data
                M2 = BilinearForm(pde.fes)
                M2 += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = gfu2)
                M2.Assemble()
                mass_old_vec = -0.5*M2.mat*pde.prev_gfu[-2]

                data.t.Set(data.t.Get() + data.prev_dt[-2])

                gfu1 = GridFunction(data.dX.space)
                gfu1.vec.data = data.prev_dX[-1].data
                M1 = BilinearForm(pde.fes)
                M1 += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = gfu1)
                M1.Assemble()
                mass_old_vec += 2*M1.mat*pde.prev_gfu[-1]

                data.t.Set(data.t.Get() + data.dt.Get())

                mass = 1.5*pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = dX_extr)

            else:

                data.t.Set(data.t.Get() + data.dt.Get())
                M = BilinearForm(pde.fes)
                M += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = dX_extr)
                M.Assemble()

                mass_old_vec = -0.5*M.mat*pde.prev_gfu[-2]
                mass_old_vec += 2*M.mat*pde.prev_gfu[-1]

                mass = 1.5*pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = dX_extr)

            A = BilinearForm(pde.fes)
            lhs = pde.GetLHS(data, pde.get_trial(), pde.get_test(), dX = dX_extr)
            A += lhs + mass

            F = LinearForm(pde.fes)
            F += pde.GetRHS(data, pde.get_test(), dX = dX_extr)
            F.Assemble()
            res = F.vec

            if pde.nonlinear:

                gfu_extr = GridFunction(pde.gfu.space)
                gfu_extr.vec.data = self.Extrapolate([pde.prev_gfu[-2], pde.prev_gfu[-1]]).data

                N1 = BilinearForm(pde.fes)
                nonlin = pde.gfu.vec.CreateVector()
                N1 += pde.GetNL(data, pde.get_trial(), pde.get_test(), dX = dX_extr)
                N1.Apply(gfu_extr.vec, nonlin)
                res += -1*nonlin

            res += mass_old_vec

            A.Assemble()
            pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

            data.t.Set(data.t.Get() - data.dt.Get())

            pde.PostProcess(data, dX = dX_extr)

    def Extrapolate(self, vec):

        v_ext = vec[-1].CreateVector()
        v_ext.data[:] = 0
        v_ext += 2*vec[-1] 
        v_ext += -1*vec[-2] 

        return v_ext

class CN(Scheme):

    def __init__(self, conservative = True):
        
        super().__init__()
        self.conservative = conservative

    def Solve(self, data, pde):

        pde.PreProcess(data, dX = data.dX)

        gfu = GridFunction(data.dX.space)
        gfu.vec.data = data.prev_dX[-1].data

        res1 = pde.gfu.vec.CreateVector()
        A1 = BilinearForm(pde.fes)
        A1 += pde.GetLHS(data, pde.get_trial(), pde.get_test(), dX = gfu)
        if pde.nonlinear:
            A1 += pde.GetNL(data, pde.get_trial(), pde.get_test(), dX = gfu)
        A1.Apply(pde.prev_gfu[-1], res1)
        res = -0.5*res1

        F1 = LinearForm(pde.fes)
        F1 += pde.GetRHS(data, pde.get_test(), dX = gfu)
        F1.Assemble()
        res += 0.5*F1.vec

        if self.conservative:

            M = BilinearForm(pde.fes)
            M += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = gfu)
            M.Assemble()
            res += M.mat*pde.prev_gfu[-1]

        else:

            gfu_half = GridFunction(data.dX.space)
            gfu_half.vec.data = 0.5*(data.prev_dX[-1].data + data.dX.vec.data)

            M = BilinearForm(pde.fes)
            M += pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = gfu_half)
            M.Assemble()
            res += M.mat*pde.prev_gfu[-1]

        data.t.Set(data.t.Get() + data.dt.Get())

        A = BilinearForm(pde.fes)
        lhs = pde.GetLHS(data, pde.get_trial(), pde.get_test(), dX = data.dX)

        if self.conservative:
            mass = pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = data.dX)
        else:
            mass = pde.GetMass(data, pde.get_trial(), pde.get_test(), dX = gfu_half)
        A += 0.5*lhs + mass

        F = LinearForm(pde.fes)
        rhs = pde.GetRHS(data, pde.get_test(), dX = data.dX)
        F += rhs
        F.Assemble()
        res += 0.5*F.vec

        if pde.nonlinear:

            A += 0.5*pde.GetNL(data, pde.get_trial(), pde.get_test(), dX = data.dX)
            SimpleNewtonSolve(A, res, pde.gfu)

        else:

            A.Assemble()
            pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*res

        data.t.Set(data.t.Get() - data.dt.Get())

        pde.PostProcess(data, dX = data.dX)