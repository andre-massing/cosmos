from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error
import numpy as np
from ngsolve.webgui import Draw

class WillmoreBGN(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs

        self.nfields = 2

        accepted_keys = ['rhs', 'clamped_bnd', 'clamped_f',
                         'domain', 'name', 'sp_curv', 'postprocess',
                         'mc_autoupdate', 'mc0']
        defaults = [None, None, None,
                    '.*', ['displacement', 'mean_curvature'], CF(0),
                    False, True, None]
        
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
        
        if self.params['clamped_bnd']:
            if not set([self.params['clamped_bnd']]) <= set(data.boundary_markers + ['.*']):
                raise ValueError('Clamped boundary conditions are imposed on non-existing boundary!')
        
        if self.params['clamped_bnd']:
            self.V1 = Compress(VectorH1(data.mesh, order=self.fes_order,
                        definedon=self.domain,
                        dirichlet_bbnd = data.mesh.BBoundaries(self.params['clamped_bnd'])))
        else:
            self.V1 = Compress(VectorH1(data.mesh, order=self.fes_order,
                        definedon=self.domain))
        V2 = Compress(VectorH1(data.mesh, order=self.fes_order,
                    definedon=self.domain))
            
        self.fes = self.V1*V2

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)

        self.dX_h, self.Y_h = self.gfu.components 
        self.kappa_h = GridFunction(V2)
        if not self.params['mc0']:
            ComputeMC(data, self.kappa_h, self.params)
        else:
            self.kappa_h.Set(self.params['mc0'], definedon = self.domain)
        ns = specialcf.normal(data.mesh.dim)
        self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns, definedon = self.domain)
            
        V_vol = VectorH1(data.mesh, order = self.fes_order)
        self.gfu_save = list(GridFunction(CompressCompound(V_vol*V_vol)).components)
        self.dX_save, self.kappa_save = self.gfu_save[0], self.gfu_save[1]
        self.dX_save.Set(self.dX_h, definedon = self.domain)
        self.kappa_save.Set(self.kappa_h, definedon = self.domain)

        self.X0 = GridFunction(self.V1)
        if data.mesh.dim == 2:
            self.X0.Set(CF((x,y)), definedon=self.domain)
        elif data.mesh.dim == 3:
            self.X0.Set(CF((x, y, z)), definedon=self.domain)

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)
            
    def GetLHS(self, data, trial, test, dX = None):

        if data.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = dX)
        elif data.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = dX)

        lhs = -InnerProduct(grad(trial[1]).Trace(), grad(test[0]).Trace())*ds(deformation = dX)
        lhs += InnerProduct(trial[1], test[1])*ds_lumped
        lhs += (InnerProduct(grad(trial[0]).Trace(), grad(test[1]).Trace()))*ds(deformation = dX)
        
        return lhs
        
    def GetRHS(self, data, test, dX = None):

        ns = specialcf.normal(data.mesh.dim)
        Ps = Id(data.mesh.dim) - OuterProduct(ns, ns)

        if data.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = dX)
        elif data.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = dX)

        rhs = -InnerProduct(grad(self.X0).Trace(), grad(test[1]).Trace())*ds(deformation = dX)
        rhs += -self.params['sp_curv']*InnerProduct(ns, test[1])*ds_lumped
        if self.params['rhs']:
            rhs += InnerProduct(self.params['rhs'], test[0])*ds(deformation = dX)

        def D_s(chi, Ps):
            sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
            return sym
        if data.mesh.dim == 3:
            rhs += InnerProduct(Trace(grad(self.Y_h).Trace()),Trace(grad(test[0]).Trace()))*ds(deformation = dX)
            rhs += -2*InnerProduct(grad(self.Y_h).Trace().trans, D_s(test[0], Ps)*Ps.trans)*ds(deformation = dX)
            rhs += -1*InnerProduct(self.params['sp_curv']*self.kappa_h, grad(test[0]).Trace().trans*ns)*ds_lumped
            rhs += -0.5*InnerProduct((Norm(self.kappa_h - self.params['sp_curv']*ns)**2)*Ps,grad(test[0]).Trace())*ds_lumped
            rhs += InnerProduct(InnerProduct(self.Y_h, self.kappa_h)*Ps,grad(test[0]).Trace())*ds_lumped
        elif data.mesh.dim == 2 :
            rhs += InnerProduct(InnerProduct(self.Y_h, self.kappa_h)*Ps,grad(test[0]).Trace())*ds_lumped

        tE = specialcf.tangential(data.mesh.dim)
        if data.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)
        if self.params['clamped_bnd']:
            if data.mesh.dim == 3:
                gfF = GridFunction(FacetSurface(data.mesh, order=0))
                gfF.Set(1, definedon=data.mesh.BBoundaries(self.params['clamped_bnd']))
            elif data.mesh.dim == 2:
                gfF = GridFunction(H1(data.mesh, order =1, definedon=data.mesh.Boundaries('.*')))
                gfF.Set(1, definedon=data.mesh.BBoundaries(self.params['clamped_bnd']))
            
            if self.params['clamped_f']:
                rhs += InnerProduct(self.params['clamped_f'], test[1]) * gfF * ds(element_boundary=True)
            else:
                rhs += InnerProduct(nE, test[1]) * gfF * ds(element_boundary=True)

        return rhs

    def GetMass(self, data, trial, test, dX = None):

        if data.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = dX)
        elif data.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = dX)

        mass = InnerProduct(trial[0], test[0])/data.dt*ds_lumped

        return mass
    
    def PreProcess(self, data, dX = None):

        self.gfu.components[0].vec.data = data.dX.vec.data
        self.prev_gfu.append(self.gfu.vec.Copy())      
        if len(self.prev_gfu)>6:
            self.prev_gfu.pop(0)

        ns = specialcf.normal(data.mesh.dim)
        if self.params['mc_autoupdate']:
            data.mesh.SetDeformation(dX)
            ComputeMC(data, self.kappa_h, self.params)
            self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns, definedon = self.domain)
            data.mesh.UnsetDeformation()
    
    def PostProcess(self, data, dX = None):

        super().PostProcess(data)

        if self.postprocess:

            ns = specialcf.normal(data.mesh.dim)
            n_h = GridFunction(data.dX.space)
            data.mesh.SetDeformation(self.gfu.components[0])
            n_h.Set(ns, definedon=self.domain)
            data.mesh.UnsetDeformation()

            V2 = H1(data.mesh, order=self.fes_order,
                        definedon=self.domain)
            
            fes = self.V1*V2
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
        
    def Update(self, data):

        data.mesh.SetDeformation(data.dX)

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

def ComputeMC(data, gfu, params):

    if data.mesh.dim == 2:
        ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ds_lumped = ds(intrules = { SEGM : ir })
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir })
    elif data.mesh.dim == 3:
        ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        ds_lumped = ds(intrules = { TRIG : ir })
        ds_el_lumped = ds(element_boundary=True, intrules = { TRIG : ir })

    ns = specialcf.normal(data.mesh.dim)
    tE = specialcf.tangential(data.mesh.dim)
    if data.mesh.dim == 2:
        nE = tE
    else:
        nE = Cross(ns, tE)
    Ps = Id(data.mesh.dim) - OuterProduct(ns, ns)

    fes0 = gfu.space
    gfu0 = GridFunction(fes0)
    kappa0, eta0 = fes0.TnT()
    A0 = BilinearForm(fes0)
    F0 = LinearForm(fes0)
    A0 += kappa0*eta0*ds_lumped
    A0.Assemble()
    F0 += -InnerProduct(Ps, grad(eta0).Trace())*ds

    if params['clamped_bnd']:

        if data.mesh.dim == 3:
            gfF = GridFunction(FacetSurface(data.mesh, order=0))
            gfF.Set(1, definedon=data.mesh.BBoundaries(params['clamped_bnd']))
            F0 += InnerProduct(nE, eta0) * gfF * ds_el_lumped

        elif data.mesh.dim == 2:
            gfF = GridFunction(H1(data.mesh, order =1,\
                    definedon=data.mesh.Boundaries('.*')))
            gfF.Set(1, definedon=data.mesh.BBoundaries(params['clamped_bnd']))
            F0 += InnerProduct(gfF*nE, eta0) * ds_el_lumped

    F0.Assemble()
    gfu0.vec.data = A0.mat.Inverse(fes0.FreeDofs())*F0.vec
    gfu.vec.data = gfu0.vec.data