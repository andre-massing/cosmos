from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.pdes.ale import ALE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error, MandBP
import numpy as np
import scipy.sparse as sp
from ngsolve.webgui import Draw

class CahnHilliardBnd(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.fields = 2
        self.nonlinear = True

        # Initialize the parameters
        accepted_keys = ['D', 'gamma', 'epsilon', 'u0', 'rhs',
                         'neu_c', 'neu_mu', 'dir_c', 'dir_mu',
                         'domain', 'name', 'periodic',
                         'MP', 'BP', 'ALE']
        defaults = [None, None, None, None, None, 
                    {}, {}, {}, {},
                    '.*', ['c', 'mu'], False,
                    False, None, None]

        if kwargs:
            params_check(kwargs, accepted_keys, defaults)
            self.params = kwargs
            for i, (key, value) in enumerate(self.params.items()):
                if i<5 and isinstance(self.params[key], (int, float)):
                    self.params[key] = CF(self.params[key])
        else:
            self.params = {}
            params_check(self.params, accepted_keys, defaults)

    def Initialize(self, data):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.dim = data.mesh.dim
        self.domain = data.mesh.Boundaries(self.params['domain'])
        self.name = self.params['name']
        
        if self.params['periodic']:
            V = Compress(Periodic(H1(data.mesh, order = self.fes_order, 
                                            definedon = self.domain)))
        else:
            V = Compress(H1(data.mesh, order = self.fes_order, 
                                   definedon = self.domain))
        self.fes = V*V
            
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        
        self.gfu = GridFunction(self.fes)
        self.gfu_save = list(GridFunction(Compress(H1(data.mesh, order = self.fes_order))**2).components)

        if self.params['u0']:
            self.gfu.components[0].Set(self.params['u0'][0], definedon = self.domain)
            self.gfu.components[1].Set(self.params['u0'][1], definedon = self.domain)

        self.gfu_save[0].Set(self.gfu.components[0], definedon = self.domain)
        self.gfu_save[1].Set(self.gfu.components[1], definedon = self.domain)

        if self.params['MP']:

            if data.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
                ds_lumped = ds(intrules = { SEGM : ir })
            elif data.mesh.dim == 3:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                ds_lumped = ds(intrules = { TRIG : ir })
            A = BilinearForm(self.gfu.components[0].space, symmetric = True)
            u, v = self.gfu.components[0].space.TnT()
            A += u*v*ds_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu.components[0].vec.Copy().FV().NumPy()
            self.mass0 = np.sum(weights*gfu_vec)

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)

    def GetLHS(self, data, trial, test, ale):

        ns = specialcf.normal(data.mesh.dim)
        Ps = Id(self.dim) - OuterProduct(ns, ns) 
        tE = specialcf.tangential(data.mesh.dim)
        h = specialcf.mesh_size
        if data.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        if data.mesh.dim == 2:
            facet_space = self.fes
        else:
            facet_space = FacetSurface(data.mesh, order = 0)

        if self.params['D']:
            lhs = self.params['D']*grad(trial[1]).Trace()*grad(test[0]).Trace()*ds(deformation = ale.deformation)
                   
            if self.params['dir_mu']:
                dir_mu = {}
                alpha = 5 * self.fes_order * (self.fes_order+1)
                for i, (key, value) in enumerate(self.params['dir_mu'].items()):
                    dir_mu[str(i)] = GridFunction(facet_space)
                    dir_mu[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                    lhs += - self.params['gamma']*InnerProduct(nE, grad(trial[1]).Trace())*dir_mu[str(i)]*test[0]*ds(element_boundary=True, deformation = ale.deformation) \
                            - self.params['gamma']*InnerProduct(nE, grad(test[0]).Trace())*dir_mu[str(i)]*trial[1]*ds(element_boundary=True, deformation = ale.deformation)\
                            + self.params['gamma']*alpha/h*trial[1]*test[1]*dir_mu[str(i)]*ds(element_boundary=True, deformation = ale.deformation)
        else:
            lhs =  CF(0)*grad(trial[1])*grad(test[0])*ds(deformation = ale.deformation)

        lhs += trial[1]*test[1]*ds(deformation = ale.deformation)

        if self.params['gamma']:
            lhs += -1*self.params['gamma']*grad(trial[0]).Trace()*grad(test[1]).Trace()*ds(deformation = ale.deformation)
                    
            if self.params['dir_c']:
                dir_c = {}
                alpha = 5 * self.fes_order * (self.fes_order+1)
                for i, (key, value) in enumerate(self.params['dir_c'].items()):
                    dir_c[str(i)] = GridFunction(facet_space)
                    dir_c[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                    lhs += self.params['D']*InnerProduct(nE, grad(trial[0]).Trace())*dir_c[str(i)]*test[1]*ds(element_boundary=True, deformation = ale.deformation) \
                        + self.params['D']*InnerProduct(nE, grad(test[1]).Trace())*dir_c[str(i)]*trial[0]*ds(element_boundary=True, deformation = ale.deformation)\
                        + self.params['D']*alpha/h*trial[0]*test[0]*dir_c[str(i)]*ds(element_boundary=True, deformation = ale.deformation)
            
        return lhs

    def GetRHS(self, data, test, ale):

        ns = specialcf.normal(data.mesh.dim)
        Ps = Id(self.dim) - OuterProduct(ns, ns) 
        tE = specialcf.tangential(data.mesh.dim)
        h = specialcf.mesh_size
        if data.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        if data.mesh.dim == 2:
            facet_space = self.fes
        else:
            facet_space = FacetSurface(data.mesh, order = 0)

        if self.params['rhs']:
            rhs = self.params['rhs'][0]*test[0]*ds(deformation = ale.deformation)
            rhs += self.params['rhs'][1]*test[1]*ds(deformation = ale.deformation)
        else:
            rhs = CF(0)*test[0]*ds(deformation = ale.deformation)
            rhs += CF(0)*test[1]*ds(deformation = ale.deformation)
        
        if self.params['dir_c']:
            dir_c = {}
            alpha = 5 * self.fes_order * (self.fes_order+1)
            for i, (key, value)  in enumerate(self.params['dir_c'].items()):
                dir_c[str(i)] = GridFunction(facet_space)
                dir_c[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                lhs += self.params['D']*InnerProduct(nE, grad(test[1]).Trace())*dir_c[str(i)]*value*ds(element_boundary=True, deformation = ale.deformation)\
                    + self.params['D']*alpha/h*value*test[0]*dir_c[str(i)]*ds(element_boundary=True, deformation = ale.deformation)
        if self.params['neu_c']:
            neu_c = {}
            for i, (key, value) in enumerate(self.params['neu_c'].items()):
                neu_c[str(i)] = GridFunction(facet_space)
                neu_c[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += InnerProduct(nE, value)*neu_c[str(i)]*test[1]*ds(element_boundary=True, deformation = ale.deformation)

        if self.params['dir_mu']:
            dir_mu = {}
            alpha = 5 * self.fes_order * (self.fes_order+1)
            for i, (key, value) in enumerate(self.params['dir_mu'].items()):
                dir_mu[str(i)] = GridFunction(facet_space)
                dir_mu[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                lhs += - self.params['gamma']*InnerProduct(nE, grad(test[0]).Trace())*dir_mu[str(i)]*value*ds(element_boundary=True, deformation = ale.deformation)\
                        + self.params['gamma']*alpha/h*value*test[1]*dir_mu[str(i)]*ds(element_boundary=True, deformation = ale.deformation)
        if self.params['neu_mu']:
            neu_mu = {}
            for i, (key, value) in enumerate(self.params['neu_mu'].items()):
                neu_mu[str(i)] = GridFunction(facet_space)
                neu_mu[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += InnerProduct(nE, value)*neu_mu[str(i)]*test[0]*ds(element_boundary=True, deformation = ale.deformation)
            
            
        return rhs

    def GetMass(self, data, trial, test, ale):

        mass = 1/data.dt*trial[0]*test[0]*ds(deformation = ale.deformation)
        return mass

    def GetNL(self, data, trial, test, ale):

        nonlin = -1*self.params['epsilon']*(trial[0]**3 - trial[0])*test[1]*ds(deformation = ale.deformation)
                    
        return nonlin
    
    def PostProcess(self, data, ale):

        super().PostProcess(data, ale)

        if self.params['BP'] and not self.params['MP']:

            gfu_vec = self.gfu.components[0].vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.params['BP'])
            self.gfu.components[0].vec.data = gfu_new

        elif self.params['MP']:

            if hasattr(data, 'dt'):
                dt = data.dt.Get()
            else:
                raise Exception('A time-dependent simulation is needed to impose conservative mass!')

            if data.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
                ds_lumped = ds(intrules = { SEGM : ir }, deformation=ale.deformation)
            elif data.mesh.dim == 3:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                ds_lumped = ds(intrules = { TRIG : ir }, deformation=ale.deformation)
            A = BilinearForm(self.gfu.components[0].space, symmetric = True)
            u, v = self.gfu.components[0].space.TnT()
            A += u*v*ds_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu.components[0].vec.Copy().FV().NumPy()

            if self.params['BP']:
                BP = self.params['BP']
            else:
                BP = [-np.inf, np.inf]

            gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                MP=self.params['MP'], mass0 =self.mass0, dt = dt)

            self.gfu.components[0].vec.data = gfu_new

    def Update(self, data, ale):

        data.mesh.SetDeformation(ale.deformation)

        self.gfu_save[0].Set(self.gfu.components[0], definedon = self.domain)
        self.gfu_save[1].Set(self.gfu.components[1], definedon = self.domain)

        for save in self.save_error:
            save.Save(data, self)

        for save in self.save_solution:
            save.Save(data, self)

        data.mesh.UnsetDeformation()

    def get_error(self, data, ex_sol, norm):

        err1 = compute_error(data=data, gfu = self.gfu.components[0], u_ex=ex_sol[0],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        err2 = compute_error(data=data, gfu = self.gfu_comp[1], u_ex=ex_sol[1],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err1, err2]

    def set_solution(self, value):

        self.gfu.Set(value, definedon=self.domain)

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