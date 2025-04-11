from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error
from ngsolve.webgui import Draw
import numpy as np

class MCDziuk(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.nfields = 1

        accepted_keys = ['rhs', 'domain', 'name', 'postprocess']
        defaults = [None, '.*', "displacement", False]
        
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
        self.name = [self.params['name']]
        self.postprocess = self.params['postprocess']
        
        self.fes = Compress(VectorH1(data.mesh, order=self.fes_order,
                    definedon=self.domain))

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)
        self.X0 = GridFunction(self.fes)
        if data.mesh.dim == 2:
            self.X0.Set(CF((x,y)), definedon=self.domain)
        elif data.mesh.dim == 3:
            self.X0.Set(CF((x, y, z)), definedon=self.domain)
            
        V_vol = VectorH1(data.mesh, order = self.fes_order)
        self.gfu_save = [GridFunction(Compress(V_vol))]

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)
            
    def GetLHS(self, data, trial, test, dX = None):

        lhs = (InnerProduct(grad(trial[0]).Trace(), grad(test[0]).Trace()))*ds(deformation=dX)
        
        return lhs
        
    def GetRHS(self, data, test, dX = None):

        rhs = -1*InnerProduct(grad(self.X0).Trace(), grad(test[0]).Trace())*ds(deformation=dX)

        return rhs
    
    def GetMass(self, data, trial, test, dX = None):

        mass = trial[0]*test[0]/data.dt*ds(deformation=dX)

        return mass
    
    def PreProcess(self, data):
        
        self.gfu.vec.data = data.dX.vec.data
        self.prev_gfu.append(self.gfu.vec.Copy())      
        if len(self.prev_gfu)>6:
            self.prev_gfu.pop(0)
    
    def PostProcess(self, data):

        super().PostProcess(data)

        if self.postprocess:

            ns = specialcf.normal(data.mesh.dim)
            n_h = GridFunction(data.dX.space)
            data.mesh.SetDeformation(self.gfu)
            n_h.Set(ns, definedon=self.domain)
            vec = n_h.vec.FV().NumPy()
            size = int(len(n_h.vec.data)/data.mesh.dim)
            if data.mesh.dim == 2:
                norm = np.sqrt(vec[0:size]**2 + vec[size:2*size]**2)
            else:
                norm = np.sqrt(vec[0:size]**2 + vec[size:2*size]**2 +vec[2*size:3*size]**2)
            norm = np.tile(norm, data.mesh.dim)
            n_h.vec.data = vec/norm

            J = grad(self.X0).Trace()*grad(self.X0).Trace().trans + OuterProduct(n_h, n_h)
            invJ = Inv(J)
            dJ = sqrt(Det(J))

            V1 = VectorH1(data.mesh, order=self.fes_order,
                        definedon=self.domain)
            V2 = H1(data.mesh, order=self.fes_order,
                        definedon=self.domain)
            
            fes = V1*V2
            w_h = GridFunction(fes)
            A = BilinearForm(fes)
            (w, kappa), (eta, mu) = fes.TnT()
            A += InnerProduct(grad(w).Trace()*invJ, grad(eta).Trace())*dJ*ds
            A += -1*InnerProduct(kappa*n_h, eta)*dJ*ds
            A += -1*InnerProduct(w, mu*n_h)*dJ*ds
            F = LinearForm(fes)
            Ps = Id(data.mesh.dim) - OuterProduct(ns, ns)
            F += -1*InnerProduct(Ps*invJ, grad(eta).Trace())*dJ*ds
            A.Assemble()
            F.Assemble()
            w_h.vec.data = A.mat.Inverse(freedofs = fes.FreeDofs())*F.vec
            self.gfu.vec.data += w_h.components[0].vec.data

            data.mesh.UnsetDeformation()

        
    def Update(self, data):

        data.mesh.SetDeformation(data.dX)

        self.gfu_save[0].Set(self.gfu, definedon = self.domain)

        for save in self.save_error:
            save.Save(data, self)

        for save in self.save_solution:
            save.Save(data, self)

        data.mesh.UnsetDeformation()

    def get_error(self, data, ex_sol, norm):

        err = compute_error(data=data, gfu = self.gfu, u_ex=ex_sol,
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err]
    
    def set_solution(self, value):

        self.gfu.Set(value, definedon=self.domain)

    def get_solution(self):

        return self.gfu

    def print_info(self):

        print(60*'-')

        print('This is a solver for the Mean Curvature flow of Dziuk')
        print('It computes one step of it given the mesh')
        print('The algorithm is described in the article:')
        print('Stabilization for the mean curvature is implemented as in the article:')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')