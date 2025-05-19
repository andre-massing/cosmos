from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error
import numpy as np
from ngsolve.webgui import Draw

class DistanceVol(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs
        self.nfields = 1

        accepted_keys = ['dirichlet',
                         'domain', 'name']
        defaults = [None,
                    '.*', "distance"]
        
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

        self.domain = data.mesh.Materials(self.params['domain'])
        self.name = self.params['name']
        
        if data.mesh.ne == 0:
            raise Exception('The mesh has no volume elements! The PDE ' 
                            + self.name + ' cannot be initialized')
        
        if self.params['dirichlet']:
            self.fes = Compress(H1(data.mesh, order = 1, definedon = self.domain,
                            dirichlet = self.params['dirichlet']))
        else:
            raise Exception('At least one zero distance boundary has to be defined')

        V = Compress(VectorH1(data.mesh, order = 1, definedon = self.domain))
        self.X = GridFunction(V)

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)
        self.dist_h = self.gfu
        self.gfu_save = self.gfu

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)
            
    def GetLHS(self, data, trial, test, ale):

        lhs = grad(trial[0])*grad(test[0])*dx(deformation=ale.deformation)
        
        return lhs
        
    def GetRHS(self, data, test, ale):

        rhs = (-div(self.X)*test[0])*dx(deformation = ale.deformation)
        
        return rhs

    def GetMass(self, data, trial, test, ale):

        mass = CF(0)*ds

        return mass
    
    def PreProcess(self, data, ale):

        self.prev_gfu.append(self.gfu.vec.Copy())      
        if len(self.prev_gfu)>6:
            self.prev_gfu.pop(0)


        data.mesh.SetDeformation(ale.deformation)
        aux = GridFunction(L2(data.mesh, definedon = self.domain))
        aux.Set(specialcf.mesh_size)
        delta = np.max(aux.vec.data)**2
        data.mesh.UnsetDeformation()

        h = specialcf.mesh_size
        u, v = self.fes.TnT()
        fes1 = Compress(H1(data.mesh, order = 2, definedon = self.domain,
                            dirichlet = self.params['dirichlet']))
        A = BilinearForm(fes1)
        A += (u*v + delta*grad(u)*grad(v))*dx(deformation=ale.deformation)
        F = LinearForm(fes1)
        u_h = GridFunction(fes1)
        u_h.Set(1, definedon = data.mesh.Boundaries(self.params['dirichlet']))

        A.Assemble()
        F.Assemble()
        res = F.vec
        res += -1*A.mat*u_h.vec
        u_h.vec.data += A.mat.Inverse(freedofs = fes1.FreeDofs())*res

        self.X.Set(-grad(u_h)/Norm(grad(u_h)))    
    
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

        err = compute_error(data=data, gfu = self.dist_h, u_ex=ex_sol,
                            norm = norm, domain = self.domain, 
                            VorB = VOL)
    
        
        return [err]
    
    def set_solution(self, value):

        self.dist_h.Set(value, definedon=self.domain)

    def get_solution(self):

        return self.dist_h

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