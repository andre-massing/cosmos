from ngsolve import *
from cosmos.pdes.pde_mc_base import BaseMC
from cosmos.pdes.pde_tools import compute_error
import numpy as np
from cosmos.solvers.fields import Field
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady

class MCBGN(BaseMC):

    def __init__(self, rhs = None, domain:str = '.*', name = ['displacement', 'mean_curvature'],
                 mc0 = None, time_scheme = BDF1()):

        super().__init__()

        self.rhs = Field(rhs)
        self.domain = domain
        self.name = name
        self.mc0 = Field(mc0)
        self.time_scheme = time_scheme

        self.nfields = 2
        self.displacement = self.gfu

    def Initialize(self, solverdata):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = solverdata.mesh.Boundaries(self.domain)
        
        V1 = Compress(VectorH1(solverdata.mesh, order=self.fes_order,
                    definedon=self.domain))
            
        V2 = Compress(VectorH1(solverdata.mesh, order=self.fes_order,
                    definedon=self.domain))
            
        self.fes = V1*V2

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)
        self.displacement, self.mean_curvature = self.gfu.components 
        if self.mc0():
            self.mean_curvature.Set(self.mc0(), definedon = self.domain)
            
        V_vol = VectorH1(solverdata.mesh, order = self.fes_order)
        self.gfu_save = list(GridFunction(CompressCompound(V_vol*V_vol)).components)
        self.dX_save, self.kappa_save = self.gfu_save[0], self.gfu_save[1]
        self.dX_save.Set(self.displacement, definedon = self.domain)
        self.kappa_save.Set(self.mean_curvature, definedon = self.domain)

        self.X0 = GridFunction(V1)
        if solverdata.mesh.dim == 2:
            self.X0.Set(CF((x,y)), definedon=self.domain)
        elif solverdata.mesh.dim == 3:
            self.X0.Set(CF((x, y, z)), definedon=self.domain)

        for save in self.save_error:
            save.Initialize(solverdata, self)

        for save in self.save_solution:
            save.Initialize(solverdata, self)
            
    def GetLHS(self, solverdata, trial, test):

        if solverdata.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = solverdata.ale.deformation)
        elif solverdata.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = solverdata.ale.deformation)

        lhs = -InnerProduct(trial[1], test[0])*ds_lumped
        lhs += InnerProduct(trial[1], test[1])*ds_lumped
        lhs += (InnerProduct(grad(trial[0]).Trace(), grad(test[1]).Trace()))*ds(deformation = solverdata.ale.deformation)
        
        return lhs
        
    def GetRHS(self, solverdata, test):

        rhs = -InnerProduct(grad(self.X0).Trace(), grad(test[1]).Trace())*ds(deformation = solverdata.ale.deformation)
        if self.rhs():
            rhs += InnerProduct(self.rhs(), test[0])*ds(deformation = solverdata.ale.deformation)

        return rhs

    def GetMass(self, solverdata, trial, test):

        if solverdata.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = solverdata.ale.deformation)
        elif solverdata.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = solverdata.ale.deformation)

        mass = InnerProduct(trial[0], test[0])/solverdata.dt*ds_lumped

        return mass
        
    def Update(self, solverdata):

        solverdata.mesh.SetDeformation(solverdata.ale.deformation)

        self.dX_save.Set(self.displacement, definedon = self.domain)
        self.kappa_save.Set(self.mean_curvature, definedon = self.domain)

        for save in self.save_error:
            save.Save(solverdata, self)

        for save in self.save_solution:
            save.Save(solverdata, self)

        solverdata.mesh.UnsetDeformation()

    def get_error(self, solverdata, ex_sol, norm):

        err0 = compute_error(data=solverdata, gfu = self.displacement, u_ex=ex_sol[0],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        err1 = compute_error(data=solverdata, gfu = self.mean_curvature, u_ex=ex_sol[1],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err0, err1]

    
    def set_solution(self, value):

        self.displacement.Set(value[0], definedon=self.domain)
        self.mean_curvature.Set(value[1], definedon=self.domain)

    def get_solution(self):

        return [self.displacement, self.mean_curvature]

    def print_info(self):

        print(60*'-')

        print('This is a solver for the Willmore flow')
        print('It computes one step of it given the mesh')
        print('The algorithm is described in the article:')
        print('Stabilization for the mean curvature is implemented as in the article:')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')