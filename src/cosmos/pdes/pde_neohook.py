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
                        'factor', 'rho', 'dir',
                         'domain', 'name', 'dirichlet', 'steady']
        defaults = [None, None, None, None,
                    None, CF(1), None,
                    '.*', ['displacement', 'velocity'], None, False]
        
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
        
        if self.params['dirichlet']:
            V1 = Compress(VectorH1(data.mesh, order = 1, definedon = self.domain,
                          dirichlet = self.params['dirichlet']))
        else:
            V1 = Compress(VectorH1(data.mesh, order = 1, definedon = self.domain))
        self.fes = V1*V1

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
            
    def GetLHS(self, data, trial, test, ale):
        
        lhs = -1*trial[1]*test[1]*dx
        
        return lhs
        
    def GetRHS(self, data, test, ale):

        if self.params['rhs']:
            rhs = self.params['rhs']*test[0]*dx
        else:
            rhs = CF((0,)*data.mesh.dim)*test[0]*dx

        if self.params['dir']:
            gamma = 5 * self.fes_order * (self.fes_order+1)
            n = specialcf.normal(data.mesh.dim)
            h = specialcf.mesh_size
            for key, value in self.params['dir'].items():
                rhs += gamma/h*value*test[0]*ds(definedon = key, skeleton = True, deformation = ale.deformation)
        
        return rhs

    def GetMass(self, data, trial, test, ale):

        if self.params['steady']:
            mass = CF(0)*trial[0]*test[1]/data.dt*dx
            mass += CF(0)*self.params['rho']*trial[1]*test[0]/data.dt*dx
        else:
            mass = trial[0]*test[1]/data.dt*dx
            mass += self.params['rho']*trial[1]*test[0]/data.dt*dx

        return mass
    
    def GetNL(self, data, trial, test, ale):

        I = Id(data.mesh.dim)
        def CalcStresses(A):
            F = A + I
            C = F.trans * F
            B = F * F.trans
            E = 0.5 * (C - I)
            J = Det(F)
            Finv = Inv(F)
            return (F, C, B, E, J, Finv)
        F, C, B, E, J, Finv = CalcStresses(Grad(trial[0]))

        power = - self.params['lam']/2/self.params['mu']
        stress = self.params['mu']*(I - Det(C)**power*Inv(C).trans)

        nonlin = (InnerProduct(F * stress, Grad(test[0])))*dx

        if self.params['dir']:
            gamma = 5 * self.fes_order * (self.fes_order+1)
            n = specialcf.normal(data.mesh.dim)
            h = specialcf.mesh_size
            for key, value in self.params['dir'].items():
                nonlin += - InnerProduct((F * stress)*n, test[0])*ds(definedon = key, skeleton=True, deformation = ale.deformation)\
                    + gamma/h*trial[0]*test[0]*ds(definedon = key, skeleton = True, deformation = ale.deformation)

        return nonlin
    
    def PreProcess(self, data, ale):

        self.prev_gfu.append(self.gfu.vec.Copy())      
        if len(self.prev_gfu)>6:
            self.prev_gfu.pop(0)
    
    def PostProcess(self, data, ale):

        super().PostProcess(data, ale)

    def Update(self, data, ale):

        data.mesh.SetDeformation(ale.deformation)

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