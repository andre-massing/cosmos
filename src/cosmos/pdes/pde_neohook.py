from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error
import numpy as np
from ngsolve.webgui import Draw

class NeoHook(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs
        self.nfields = 2
        self.nonlinear = True

        accepted_keys = ['rhs', 'mu', 'lam', 'd0',
                         'neu', 'dir',
                         'domain', 'name']
        defaults = [None, None, None, None,
                    None, None,
                    '.*', ['displacement', 'velocity']]
        
        if kwargs:
            params_check(kwargs, accepted_keys, defaults)
            self.params = kwargs
        else:
            self.params = {}
            params_check(self.params, accepted_keys, defaults)

    def Initialize(self, data):

        if self.initialized:
            return
        else:
            self.initialized = True
        
        if data.mesh.ne == 0:
            raise Exception('The mesh has no volume elements! The PDE ' 
                            + self.name + ' cannot be initialized')
        
        self.domain = data.mesh.Materials(self.params['domain'])
        self.name = self.params['name']
        
        V1 = VectorH1(data.mesh, order = 1, definedon = self.domain)
        V2 = VectorH1(data.mesh, order = 1, definedon = self.domain)
            
        self.fes = V1*V2

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)
        self.d_h, self.V_h = self.gfu.components
        if self.params['d0']:
            self.d_h.Set(self.params['d0'], definedon = self.domain)
        
        self.gfu_save = list(self.gfu.components)

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)
            
    def GetLHS(self, data, trial, test, dX = None):
        
        lhs += -1*trial[1]*test[0]*dx
        
        return lhs
        
    def GetRHS(self, data, test, dX = None):

        if self.params['rhs']:
            rhs = self.params['rhs']*test[0]*dx
        else:
            rhs = CF((0,)*data.mesh.dim)*test[0]*dx
        
        return rhs

    def GetMass(self, data, trial, test, dX = None):

        mass = trial[0]*test[0]/data.dt*dx
        mass += trial[1]*test[1]/data.dt*dx

        return mass
    
    def GetNL(self, data, trial, test, dX = None):

        I = Id(data.mesh.dim)
        def CalcStresses(A):
            F = A + I
            C = F.trans * F
            E = 0.5 * (C - I)
            J = Det(F)
            Finv = Inv(F)
            return (F, C, E, J, Finv)
        F, C, E, J, Finv = CalcStresses(Grad(trial[0]))

        def NeoHooke(C, mu=1, lam=1):
            return 0.5 * mu * (Trace(C - I) + 2 * mu / lam * Det(C) ** (-lam / 2 / mu) - 1)

        nonlin = Variation( NeoHooke(C, self.params['mu'], self.params['lam']))*dx

        return nonlin
    
    def PreProcess(self, data, dX = None):

        self.prev_gfu.append(self.gfu.vec.Copy())      
        if len(self.prev_gfu)>6:
            self.prev_gfu.pop(0)
    
    def PostProcess(self, data, dX = None):

        super().PostProcess(data)

    def Update(self, data):

        data.mesh.SetDeformation(data.dX)

        for save in self.save_error:
            save.Save(data, self)

        for save in self.save_solution:
            save.Save(data, self)

        data.mesh.UnsetDeformation()

    def get_error(self, data, ex_sol, norm):

        err0 = compute_error(data=data, gfu = self.d_h, u_ex=ex_sol[0],
                            norm = norm, domain = self.domain, 
                            VorB = VOL)
        
        err1 = compute_error(data=data, gfu = self.V_h, u_ex=ex_sol[1],
                            norm = norm, domain = self.domain, 
                            VorB = VOL)
        
        return [err0, err1]

    
    def set_solution(self, value):

        self.d_h.Set(value[0], definedon=self.domain)
        self.V_h.Set(value[1], definedon=self.domain)

    def get_solution(self):

        return [self.d_h, self.V_h]

    def print_info(self):

        print(60*'-')

        print('This is a solver for the Stokes flow')
        print('It computes one step of it given the mesh')
        print('The algorithm is described in the article:')
        print('Stabilization for the mean curvature is implemented as in the article:')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')