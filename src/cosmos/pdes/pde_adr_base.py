from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady
from cosmos.pdes.ale import ale
from ngsolve.webgui import Draw

class BaseADR(BasePDE):

    def Solve(self, solverdata):

        self.PreProcess(solverdata)
        self.SystemSolve(solverdata)
        self.PostProcess(solverdata)

    def SystemSolve(self, solverdata):

        if not isinstance(self.time_scheme, Steady):

            ale_curr = ale(solverdata)
            ale_curr.deformation.vec.data = solverdata.ale.deformation.vec.data

            if isinstance(self.time_scheme, BDF2) and len(self.prev_gfu)>1:

                solverdata.ale.deformation.vec.data = solverdata.prev_def[-2].data
                solverdata.ale.velocity.vec.data = solverdata.prev_vel[-2].data
                Moldold = BilinearForm(self.fes)
                Moldold += self.GetMass(solverdata, self.get_trial(), self.get_test())
                Moldold.Assemble()
                resMoldold = Moldold.mat*self.prev_gfu[-2]

            solverdata.ale.deformation.vec.data = ale_curr.deformation.vec.data

            if isinstance(self.time_scheme, CN):

                Aold = BilinearForm(self.fes)
                Aold += self.GetLHS(solverdata, self.get_trial(), self.get_test())
                Aold.Assemble()
                resAold = Aold.mat*self.prev_gfu[-1]
                Fold = LinearForm(self.fes)
                Fold += self.GetRHS(solverdata, self.get_test())
                Fold.Assemble()
                resFold = Fold.vec
            
            Mold = BilinearForm(self.fes)
            Mold += self.GetMass(solverdata, self.get_trial(), self.get_test())
            Mold.Assemble()
            resMold = Mold.mat*self.prev_gfu[-1]

        solverdata.t.Set(solverdata.t.Get() + solverdata.dt.Get())
        solverdata.ale.deformation.vec.data = solverdata.ale.deformation_new.vec.data

        A = BilinearForm(self.fes)
        F = LinearForm(self.fes)
        F += self.GetRHS(solverdata, self.get_test())
        F.Assemble()

        if isinstance(self.time_scheme, BDF2) and len(self.prev_gfu)>1:
            A += self.GetLHS(solverdata, self.get_trial(), self.get_test())
            res = F.vec
            res += -0.5*resMoldold
            res += 2*resMold
            A += 1.5*self.GetMass(solverdata, self.get_trial(), self.get_test())
        elif isinstance(self.time_scheme, BDF2) and len(self.prev_gfu)==1 or \
            isinstance(self.time_scheme, BDF1):
            A += self.GetLHS(solverdata, self.get_trial(), self.get_test())
            res = F.vec
            res += resMold
            A += self.GetMass(solverdata, self.get_trial(), self.get_test())
        elif isinstance(self.time_scheme, CN):
            A += 0.5*self.GetLHS(solverdata, self.get_trial(), self.get_test())
            res = 0.5*F.vec
            res += resMold
            res += 0.5*resFold
            res += -0.5*resAold
            A += self.GetMass(solverdata, self.get_trial(), self.get_test())

        A.Assemble()
        self.gfu.vec.data = A.mat.Inverse(freedofs = self.fes.FreeDofs())*res

        solverdata.t.Set(solverdata.t.Get() - solverdata.dt.Get())
        solverdata.ale.deformation.vec.data = ale_curr.deformation.vec.data

    def Update(self, solverdata):

        solverdata.mesh.SetDeformation(solverdata.ale.deformation)

        for save in self.save_error:
            save.Save(solverdata, self)

        for save in self.save_solution:
            save.Save(solverdata, self)

        solverdata.mesh.UnsetDeformation()

    def set_solution(self, field):

        self.solute.Set(field(), definedon=self.domain)

    def get_solution(self):

        return self.solute

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              Diffusion-dominant problems are supposed to be simulated.\n \
              It uses conforming H-1 elements of order 1.')

        # print('Its parameters are')
        # for key, field in self.params.items():
        #     print('  -', key, '- with field ', str(field))

        # print(60*'-', '\n')