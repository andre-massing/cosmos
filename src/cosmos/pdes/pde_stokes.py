from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error
import numpy as np
from ngsolve.webgui import Draw

class Stokes(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs

        self.nfields = 2

        accepted_keys = ['rhs', 'mu', 'v0',
                         'neu', 'dir',
                         'domain', 'name']
        defaults = [None, None, None,
                    None, None,
                    '.*', ['velocity', 'pressure']]
        
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
        
        V1 = VectorH1(data.mesh, order = 2, definedon = self.domain)
        V2 = H1(data.mesh, order = 1, definedon = self.domain)
            
        self.fes = V1*V2

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)

        self.V_h, self.P_h = self.gfu.components 
        if self.params['v0']:
            self.V_h.Set(self.params['v0'], definedon = self.domain)
        else:
            A = BilinearForm(self.fes)
            A += self.GetLHS(data, self.get_trial(), self.get_test())
            A.Assemble()
            import scipy.sparse as sp
            import matplotlib.pylab as plt
            plt.rcParams['figure.figsize'] = (12, 12)
            AA = sp.csr_matrix(A.mat.CSR())
            fig = plt.figure(); ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
            ax1.set_xlabel("numerically non-zero"); ax1.spy(AA)
            ax2.set_xlabel("reserved entries (potentially non-zero)"); ax2.spy(AA,precision=-1)
            plt.show()
            F = LinearForm(self.fes)
            F += self.GetRHS(data, self.get_test())
            F.Assemble()

            self.gfu.vec.data = A.mat.Inverse(freedofs = self.fes.FreeDofs())*F.vec
        
        self.gfu_save = list(self.gfu.components)

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)
            
    def GetLHS(self, data, trial, test, dX = None):

        lhs = self.params['mu']*InnerProduct(grad(trial[0]), grad(test[0])) * dx(deformation=dX)
        lhs += -1* div(test[0]) * trial[1] * dx(deformation=dX)
        lhs += -1* div(trial[0]) * test[1] * dx(deformation=dX)

        if self.params['dir']:
            alpha = 5 * self.fes_order * (self.fes_order+1)
            n = specialcf.normal(data.mesh.dim)
            h = specialcf.mesh_size
            for key, value in self.params['dir'].items():
                lhs += - self.params['mu']*InnerProduct(grad(trial[0])*n, test[0])*ds(definedon = key, skeleton=True, deformation=dX) \
                    - self.params['mu']*InnerProduct(grad(test[0])*n, trial[0])*ds(definedon = key, skeleton=True, deformation=dX)\
                    + self.params['mu']*alpha/h*trial[0]*test[0]*ds(definedon = key, skeleton = True, deformation=dX)
        
        return lhs
        
    def GetRHS(self, data, test, dX = None):

        if self.params['rhs']:
            rhs = self.params['rhs']*test[0]*dx(deformation=dX)
        else:
            rhs = CF((0,)*data.mesh.dim)*test[0]*dx(deformation=dX)

        if self.params['neu']:
            ns = specialcf.normal(data.mesh.dim)
            dim = data.mesh.dim
            cf = data.mesh.BoundaryCF(self.params['neu'], default = CF((0,)*dim**2, dims = (dim, dim)))
            rhs += InnerProduct(cf*ns, test[0])*ds(deformation=dX)

        if self.params['dir']:
            alpha = 5 * self.fes_order * (self.fes_order+1)
            n = specialcf.normal(data.mesh.dim)
            h = specialcf.mesh_size
            cf = data.mesh.BoundaryCF(self.params['dir'], default = CF((0,)*data.mesh.dim))
            for key, value in self.params['dir'].items():
                rhs += - self.params['mu']*InnerProduct(grad(test[0])*n, value)*ds(definedon = key, skeleton=True, deformation=dX)\
                    + self.params['mu']*alpha/h*value*test[0]*ds(definedon = key, skeleton = True, deformation=dX)\
        
        
        return rhs

    def GetMass(self, data, trial, test, dX = None):

        mass = trial[0]*test[0]/data.dt*dx(deformation=dX)

        return mass
    
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

        err0 = compute_error(data=data, gfu = self.V_h, u_ex=ex_sol[0],
                            norm = norm, domain = self.domain, 
                            VorB = VOL)
        
        err1 = compute_error(data=data, gfu = self.P_h, u_ex=ex_sol[1],
                            norm = norm, domain = self.domain, 
                            VorB = VOL)
        
        return [err0, err1]

    
    def set_solution(self, value):

        self.V_h.Set(value[0], definedon=self.domain)
        self.P_h.Set(value[1], definedon=self.domain)

    def get_solution(self):

        return [self.V_h, self.P_h]

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