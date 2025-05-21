# Import necessary libraries
from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.pdes.ale import ale
from ngsolve.webgui import Draw
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady

def SimpleNewtonSolve(a, f, gfu, tol=1e-10, maxits=20):
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

class StrongCoupling(BasePDE):
    
    def __init__(self, time_scheme = BDF1()):

        super().__init__()
        self.time_scheme = time_scheme
        
        self.PDEs = []
        self.pntrs = [0]
        self.cpl = []

    def AddPDEs(self, *args):

        for pde in args:
            if not isinstance(pde, BasePDE):
                raise TypeError("The added PDE must inherit from BasePDE")
            self.PDEs.append(pde)

    def AddCoupling(self, f = None):

        self.nonlinear = True
        self.cpl.append({'f': f})

    def Initialize(self, solverdata):

        if self.initialized:
            return
        else:
            self.initialized = True

        if len(self.PDEs)<2:
            raise Exception('Coupling class is meant for more than only 1 PDE!')

        self.PDEs[0].Initialize(solverdata)
        self.fes = self.PDEs[0].fes
        if self.PDEs[0].nonlinear:
            self.nonlinear = True
        
        self.pntrs.append(self.PDEs[0].nfields)
        i = self.PDEs[0].nfields
        for pde in self.PDEs[1:]:
            i += pde.nfields
            self.pntrs.append(i)
            pde.Initialize(solverdata)
            self.fes = self.fes*pde.fes
            if pde.nonlinear:
                self.nonlinear = True
            
        self.gfu = GridFunction(self.fes)

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

    def GetLHS(self, solverdata, trial, test):

        lhs = self.PDEs[0].GetLHS(solverdata, trial[self.pntrs[0]:self.pntrs[1]],
                                test[self.pntrs[0]:self.pntrs[1]])
        for i, pde in enumerate(self.PDEs[1:]):
            lhs += pde.GetLHS(solverdata, trial[self.pntrs[i+1]:self.pntrs[i+2]],
                            test[self.pntrs[i+1]:self.pntrs[i+2]])       
            
        return lhs
    
    def GetRHS(self, solverdata, test):

        rhs = self.PDEs[0].GetRHS(solverdata, test[self.pntrs[0]:self.pntrs[1]])
        for i, pde in enumerate(self.PDEs[1:]):
            rhs += pde.GetRHS(solverdata, test[self.pntrs[i+1]:self.pntrs[i+2]])
            
        return rhs
    
    def GetNL(self, solverdata, trial, test):

        if self.cpl:
            cpl = self.cpl[0]['f'](solverdata, trial, test)
            for cpl_i in self.cpl[1:]:
                cpl += cpl_i['f'](solverdata, trial, test)
        else:
            cpl = CF(0)*ds
        for i, pde in enumerate(self.PDEs):
            if pde.nonlinear:
                cpl += pde.GetNL(solverdata, trial[self.pntrs[i]:self.pntrs[i+1]],
                            test[self.pntrs[i]:self.pntrs[i+1]])

        return cpl
    
    def GetMass(self, solverdata, trial, test):

        mass = self.PDEs[0].GetMass(solverdata, trial[self.pntrs[0]:self.pntrs[1]],
                                    test[self.pntrs[0]:self.pntrs[1]])

        for i, pde in enumerate(self.PDEs[1:]):
            mass_i  = pde.GetMass(solverdata, trial[self.pntrs[i+1]:self.pntrs[i+2]],
                            test[self.pntrs[i+1]:self.pntrs[i+2]])
            mass += mass_i
            
        return mass
    
    def PreProcess(self, solverdata):

        for i, pde in enumerate(self.PDEs):
            pde.PreProcess(solverdata)
            for j, k in enumerate(range(self.pntrs[i], self.pntrs[i+1])):
                if pde.nfields == 1:
                    self.gfu.components[k].vec.data = pde.gfu.vec.data
                else:
                    self.gfu.components[k].vec.data = pde.gfu.components[j].vec.data

        super().PreProcess(solverdata)
    
    def PostProcess(self, solverdata):

        super().PostProcess(solverdata)

        for i, pde in enumerate(self.PDEs):
            for j, k in enumerate(range(self.pntrs[i], self.pntrs[i+1])):
                if pde.nfields == 1:
                    pde.gfu.vec.data = self.gfu.components[k].vec.data
                else:
                    pde.gfu.components[j].vec.data = self.gfu.components[k].vec.data
            pde.PostProcess(solverdata)

    def Solve(self, solverdata):

        if isinstance(self.time_scheme, CN):
            raise Exception('Strong Coupling not implemented for Crank-Nicholson scheme')

        self.PreProcess(solverdata)

        if not isinstance(self.time_scheme, Steady):

            ale_curr = ale(solverdata)
            ale_curr.deformation.vec.data = solverdata.ale.deformation.vec.data
            ale_curr.velocity.vec.data = solverdata.ale.velocity.vec.data

            if isinstance(self.time_scheme, BDF2) and len(self.prev_gfu)>1:

                solverdata.ale.deformation.vec.data = solverdata.prev_def[-2].data
                solverdata.ale.velocity.vec.data = solverdata.prev_vel[-2].data
                Moldold = BilinearForm(self.fes)
                Moldold += self.GetMass(solverdata, self.get_trial(), self.get_test())
                Moldold.Assemble()
                resMoldold = Moldold.mat*self.prev_gfu[-2]

            solverdata.ale.deformation.vec.data = solverdata.prev_def[-1].data
            solverdata.ale.velocity.vec.data = solverdata.prev_vel[-1].data
            
            Mold = BilinearForm(self.fes)
            Mold += self.GetMass(solverdata, self.get_trial(), self.get_test())
            Mold.Assemble()
            resMold = Mold.mat*self.prev_gfu[-1]

        solverdata.t.Set(solverdata.t.Get() + solverdata.dt.Get())
        solverdata.ale.deformation.vec.data = ale_curr.deformation.vec.data
        solverdata.ale.velocity.vec.data = ale_curr.velocity.vec.data

        A = BilinearForm(self.fes)
        F = LinearForm(self.fes)
        F += self.GetRHS(solverdata, self.get_test())
        F.Assemble()

        A += self.GetLHS(solverdata, self.get_trial(), self.get_test())
        A += self.GetNL(solverdata, self.get_trial(), self.get_test())

        if isinstance(self.time_scheme, BDF2) and len(self.prev_gfu)>1:
            res = F.vec
            res += -0.5*resMoldold
            res += 2*resMold
            A += 1.5*self.GetMass(solverdata, self.get_trial(), self.get_test())
        elif isinstance(self.time_scheme, BDF2) and len(self.prev_gfu)==1 or \
            isinstance(self.time_scheme, BDF1):
            res = F.vec
            res += resMold
            A += self.GetMass(solverdata, self.get_trial(), self.get_test())

        SimpleNewtonSolve(A, res, self.gfu)

        solverdata.t.Set(solverdata.t.Get() - solverdata.dt.Get())

        self.PostProcess(solverdata)

    def Update(self, solverdata):

        for pde in self.PDEs:
            pde.Update(solverdata)

    def get_error(self, data, ex_sol, norm):

        raise Exception('Coupling class doesn''t have get_error method, use the single PDE one' )
    
    def set_solution(self, value):

        raise Exception('Coupling class doesn''t have set_solution method, use the single PDE one' )

    def get_solution(self):

        raise Exception('Coupling class doesn''t have get_solution method, use the single PDE one' )
    
    def print_info(self):

        for pde in self.PDEs:
            pde.print_info()