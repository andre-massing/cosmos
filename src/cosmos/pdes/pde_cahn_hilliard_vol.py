from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error, MandBP
import numpy as np
import scipy.sparse as sp
from ngsolve.webgui import Draw

class CahnHilliardVol(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs

        self.fields = 1

        # Initialize the parameters
        accepted_keys = ['D', 'gamma', 'u0', 'rhs',
                         'neu_c', 'neu_mu', 'dir_c', 'dir_mu',
                         'domain', 'name', 'periodic',
                         'MP', 'BP']
        defaults = [None, None, None, None, 
                    {}, {}, {}, {},
                    '.*', ['c', 'mu'], False,
                    False, None]

        if kwargs:
            params_check(kwargs, accepted_keys, defaults)
            self.params = kwargs
            for i, (key, value) in enumerate(self.params.items()):
                if i<5 and isinstance(self.params[key], (int, float)):
                    self.params[key] = CF(self.params[key])
        else:
            self.params = {}
            params_check(self.params, accepted_keys, defaults) 

        self.gfu = [self.params['u0'][0], self.params['u0'][1]]
        self.nl = []

    def Initialize(self, data):

        self.dim = data.mesh.dim
        self.domain = data.mesh.Materials(self.params['domain'])
        self.name = self.params['name']

        if data.mesh.ne == 0:
            raise Exception('The mesh has no volume elements! The PDE ' 
                            + self.name + ' cannot be initialized')
        
        if self.params['periodic']:
            V = Compress(Periodic(H1(data.mesh, order = self.fes_order, 
                                            definedon = self.params['domain'])))
        else:
            V = Compress(H1(data.mesh, order = self.fes_order, 
                                   definedon = self.params['domain']))
        self.fes = V*V
            
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        self.gfu_comp = self.gfu.components
        self.gfu_old_comp = self.gfu_old.components
        self.gfu_save = list(self.gfu.components)

        if self.params['u0']:
            self.gfu_comp[0].Set(self.params['u0'][0])
            self.gfu_comp[1].Set(self.params['u0'][1])
        self.gfu_old.vec.data = self.gfu.vec.data

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)

        if self.params['MP']:

            if data.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir })
            elif data.mesh.dim == 3:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir })
            A = BilinearForm(self.gfu_comp[0].space, symmetric = True)
            u, v = self.gfu_comp[0].space.TnT()
            A += u*v*dx_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu_comp[0].vec.Copy().FV().NumPy()
            self.mass0 = np.sum(weights*gfu_vec)

    def GetLHS(self, data, trial, test):

        n = specialcf.normal(data.mesh.dim)
        h = specialcf.mesh_size

        if self.params['D']:
            lhs = self.params['D']*grad(trial[1])*grad(test[0])*dx
                    
            if self.params['dir_mu']:
                alpha = 5 * self.fes_order * (self.fes_order+1)
                for key, value in self.params['dir_mu'].items():
                   lhs +=  - self.params['gamma']*InnerProduct(n, grad(trial[1]))*test[0]*ds(definedon = key, skeleton=True) \
                        - self.params['gamma']*InnerProduct(n, grad(test[0]))*trial[1]*ds(definedon = key, skeleton=True)\
                        + self.params['gamma']*alpha/h*trial[1]*test[1]*ds(definedon = key, skeleton = True)
        else:
            lhs =  CF(0)*grad(trial[1])*grad(test[0])*dx

        lhs += trial[1]*test[1]*dx

        if self.params['gamma']:
            lhs += -1*self.params['gamma']*grad(trial[0])*grad(test[1])*dx

            if self.params['dir_c']:
                alpha = 5 * self.fes_order * (self.fes_order+1)
                for key, value in self.params['dir_c'].items():
                    lhs +=  self.params['D']*InnerProduct(n, grad(trial[0]))*test[1]*ds(definedon = key, skeleton=True) \
                        + self.params['D']*InnerProduct(n, grad(test[1]))*trial[0]*ds(definedon = key, skeleton=True)\
                        + self.params['D']*alpha/h*trial[0]*test[0]*ds(definedon = key, skeleton = True)
            
        return lhs

    def GetRHS(self, data, test):

        n = specialcf.normal(data.mesh.dim)
        h = specialcf.mesh_size

        if self.params['rhs']:
            rhs = self.params['rhs'][0]*test[0]*dx
            rhs += self.params['rhs'][1]*test[1]*dx
        else:
            rhs = CF(0)*test[0]*dx
            rhs += CF(0)*test[1]*dx
        
        if self.params['dir_c']:
            alpha = 5 * self.fes_order * (self.fes_order+1)
            for key, value in self.params['dir_c'].items():
                rhs +=  self.params['D']*InnerProduct(n, grad(test[1]))*value*ds(definedon = key, skeleton=True)\
                    + self.params['D']*alpha/h*value*test[0]*ds(definedon = key, skeleton = True)
        if self.params['neu_c']:
            for key, value in self.params['neu_c'].items():
                rhs += value*n*test[1]*ds(definedon = key)

        if self.params['dir_mu']:
            alpha = 5 * self.fes_order * (self.fes_order+1)
            for key, value in self.params['dir_mu'].items():
                rhs +=  - self.params['gamma']*InnerProduct(n, grad(test[0]))*value*ds(definedon = key, skeleton=True)\
                    + self.params['gamma']*alpha/h*value*test[1]*ds(definedon = key, skeleton = True)
        if self.params['neu_mu']:
            for key, value in self.params['neu_mu'].items():
                rhs += -value*n*test[0]*ds(definedon = key)
            
            
        return rhs

    def GetMass(self, data, trial, test):
        mass = 1/data.dt*trial[0]*test[0]*dx
        return mass
    
    def GetMassOld(self, data, trial, test):
        return self.GetMass(data, trial, test)
    
    def AddNonlinearity(self, **kwargs):

        params = kwargs
        # Initialize the parameters
        accepted_keys = ['marker0', 'markers', 'f', 'VorB', 'domain', 'grad']
        defaults = [0, 0, {}, VOL, '.*', False]  
        params_check(params, accepted_keys, defaults)
        self.nl.append(params)

    def GetNL(self, data, trial, test):

        nl_i = self.nl[0]
        trials_i = [trial[i] for i in nl_i['markers']]
        test_i = test[nl_i['marker0']]

        if nl_i['VorB'] == BND:
            if nl_i['grad']:
                nonlin = nl_i['f'](trials_i)*grad(test_i)*ds(definedon=nl_i['domain'])
            else:
                nonlin = nl_i['f'](trials_i)*test_i*ds(definedon=nl_i['domain'])
        else:
            if nl_i['grad']:
                nonlin = nl_i['f'](trials_i)*grad(test_i)*dx(definedon=nl_i['domain'])
            else:
                nonlin = nl_i['f'](trials_i)*test_i*dx(definedon=nl_i['domain']) 

        for nl_i in self.nl[1:]:

            trials_i = [trial[i] for i in nl_i['markers']]
            test_i = test[nl_i['marker0']]

            if nl_i['VorB'] == BND:
                if nl_i['grad']:
                    nonlin += nl_i['f'](trials_i)*grad(test_i)*ds(definedon=nl_i['domain'])
                else:
                    nonlin += nl_i['f'](trials_i)*test_i*ds(definedon=nl_i['domain'])
            else:
                if nl_i['grad']:
                    nonlin += nl_i['f'](trials_i)*grad(test_i)*dx(definedon=nl_i['domain'])
                else:
                    nonlin += nl_i['f'](trials_i)*test_i*dx(definedon=nl_i['domain'])
                    
        return nonlin
    
    def PreProcess(self, data):

        pass
    
    def PostProcess(self, data):

        if self.params['BP'] and not self.params['MP']:

            gfu_vec = self.gfu_comp[0].vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.params['BP'])
            self.gfu_comp[0].vec.data = gfu_new

        elif self.params['MP']:

            if hasattr(data, 'dt'):
                dt = data.dt.Get()
            else:
                raise Exception('A time-dependent simulation is needed to impose conservative mass!')

            if data.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir })
            elif data.mesh.dim == 3:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir })
            A = BilinearForm(self.gfu_comp[0].space, symmetric = True)
            u, v = self.gfu_comp[0].space.TnT()
            A += u*v*dx_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu_comp[0].vec.Copy().FV().NumPy()

            if self.params['BP']:
                BP = self.params['BP']
            else:
                BP = [-np.inf, np.inf]

            gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                MP=self.params['MP'], mass0 =self.mass0, dt = dt)

            self.gfu_comp[0].vec.data = gfu_new

    def Update(self, data):

        self.gfu_old.vec.data = self.gfu.vec.data

        for save in self.save_error:
            save.Save(data, self)

        for save in self.save_solution:
            save.Save(data, self)

    def get_error(self, data, ex_sol, norm):

        err1 = compute_error(data=data, gfu = self.gfu_comp[0], u_ex=ex_sol[0],
                            norm = norm, domain = self.domain, 
                            VorB = VOL)
        
        err2 = compute_error(data=data, gfu = self.gfu_comp[1], u_ex=ex_sol[1],
                            norm = norm, domain = self.domain, 
                            VorB = VOL)
        
        return [err1, err2]

    def set_solution(self, value):

        self.gfu.Set(value, definedon=self.domain)
        self.gfu_old.Set(value, definedon=self.domain)

    def get_solution(self):

        return self.gfu

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              Diffusion-dominant problems are supposed to be simulated.\n \
              It uses conforming H-1 elements of order 1.')

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')