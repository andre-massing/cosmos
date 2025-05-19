from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error
import numpy as np

class MCBGN(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs

        self.nfields = 2

        accepted_keys = ['rhs', 'domain', 'name', 'postprocess', 'mc0']
        defaults = [None, '.*', ['displacement', 'mean_curvature'],
                    False, None]
        
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

        self.domain = data.mesh.Boundaries(self.params['domain'])
        self.name = self.params['name']
        self.postprocess = self.params['postprocess']
        
        V1 = Compress(VectorH1(data.mesh, order=self.fes_order,
                    definedon=self.domain))
            
        V2 = Compress(VectorH1(data.mesh, order=self.fes_order,
                    definedon=self.domain))
            
        self.fes = V1*V2

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)

        self.dX_h, self.kappa_h = self.gfu.components 
        if self.params['mc0']:
            self.kappa_h.Set(self.params['mc0'], definedon = self.domain)
            
        V_vol = VectorH1(data.mesh, order = self.fes_order)
        self.gfu_save = list(GridFunction(CompressCompound(V_vol*V_vol)).components)
        self.dX_save, self.kappa_save = self.gfu_save[0], self.gfu_save[1]
        self.dX_save.Set(self.dX_h, definedon = self.domain)
        self.kappa_save.Set(self.kappa_h, definedon = self.domain)

        self.X0 = GridFunction(V1)
        if data.mesh.dim == 2:
            self.X0.Set(CF((x,y)), definedon=self.domain)
        elif data.mesh.dim == 3:
            self.X0.Set(CF((x, y, z)), definedon=self.domain)

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)
            
    def GetLHS(self, data, trial, test, ale):

        if data.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = ale.deformation)
        elif data.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = ale.deformation)

        lhs = -InnerProduct(trial[1], test[0])*ds_lumped
        lhs += InnerProduct(trial[1], test[1])*ds_lumped
        lhs += (InnerProduct(grad(trial[0]).Trace(), grad(test[1]).Trace()))*ds(deformation = ale.deformation)
        
        return lhs
        
    def GetRHS(self, data, test, ale):

        rhs = -InnerProduct(grad(self.X0).Trace(), grad(test[1]).Trace())*ds(deformation = ale.deformation)
        if self.params['rhs']:
            rhs += InnerProduct(self.params['rhs'], test[0])*ds(deformation = ale.deformation)

        return rhs

    def GetMass(self, data, trial, test, ale):

        if data.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = ale.deformation)
        elif data.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = ale.deformation)

        mass = InnerProduct(trial[0], test[0])/data.dt*ds_lumped

        return mass
    
    def PreProcess(self, data, ale):

        self.gfu.components[0].vec.data = ale.deformation.vec.data
        self.prev_gfu.append(self.gfu.vec.Copy())      
        if len(self.prev_gfu)>6:
            self.prev_gfu.pop(0)
    
    def PostProcess(self, data, ale):

        super().PostProcess(data, ale)

        if self.postprocess:

            ns = specialcf.normal(data.mesh.dim)
            n_h = GridFunction(ale.deformation.space)
            data.mesh.SetDeformation(self.gfu.components[0])
            n_h.Set(ns, definedon=self.domain)
            data.mesh.UnsetDeformation()

            V1 = VectorH1(data.mesh, order=self.fes_order,
                        definedon=self.domain)
            V2 = H1(data.mesh, order=self.fes_order,
                        definedon=self.domain)
            
            fes = V1*V2
            w_h = GridFunction(fes)
            A = BilinearForm(fes)
            (w, kappa), (eta, mu) = fes.TnT()
            A += InnerProduct(grad(w).Trace(), grad(eta).Trace())*ds
            A += -1*InnerProduct(kappa*n_h, eta)*ds
            A += 1*InnerProduct(w, mu*n_h)*ds
            F = LinearForm(fes)
            F += -1*InnerProduct(grad(self.X0).Trace(), grad(eta).Trace())*ds
            F += -1*InnerProduct(grad(self.gfu.components[0]).Trace(), grad(eta).Trace())*ds
            A.Assemble()
            F.Assemble()
            w_h.vec.data = A.mat.Inverse(freedofs = fes.FreeDofs())*F.vec
            self.gfu.components[0].vec.data += w_h.components[0].vec.data
        
    def Update(self, data, ale):

        data.mesh.SetDeformation(ale.deformation)

        self.dX_save.Set(self.dX_h, definedon = self.domain)
        self.kappa_save.Set(self.kappa_h, definedon = self.domain)

        for save in self.save_error:
            save.Save(data, self)

        for save in self.save_solution:
            save.Save(data, self)

        data.mesh.UnsetDeformation()

    def get_error(self, data, ex_sol, norm):

        err0 = compute_error(data=data, gfu = self.dX_h, u_ex=ex_sol[0],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        err1 = compute_error(data=data, gfu = self.kappa_h, u_ex=ex_sol[1],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err0, err1]

    
    def set_solution(self, value):

        self.dX_h.Set(value[0], definedon=self.domain)
        self.kappa_h.Set(value[1], definedon=self.domain)

    def get_solution(self):

        return [self.dX_h, self.kappa_h]

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