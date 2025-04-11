from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check, compute_stab_mc
from cosmos.pdes.pde_tools import compute_error

class WillmoreStab(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        raise Exception('MUST BE UPDATED!')

        self.params = kwargs

        self.fields = 3

        accepted_keys = ['rhs', 'clamped_bnd', 'clamped_f',
                         'domain', 'name', 'sp_curv',
                         'stab', 'mc0', 'ALE']
        defaults = [None, None, None,
                    '.*', ['displacement', 'mean_curvature'], CF(0),
                    CF(1e-3), None, None]
        
        if kwargs:
            params_check(kwargs, accepted_keys, defaults)
            self.params = kwargs
        else:
            self.params = {}
            params_check(self.params, accepted_keys, defaults)

    def Initialize(self, data):

        self.dim = data.mesh.dim
        self.domain = data.mesh.Boundaries(self.params['domain'])
        self.name = self.params['name']
        if self.ale:
            self.ale = self.params['ALE']
        else:
            self.ale = GridFunction(VectorH1(data.mesh))
        
        if self.params['clamped_bnd']:
            if not set([self.params['clamped_bnd']]) <= set(data.boundary_markers + ['.*']):
                raise ValueError('Dirichlet boundary conditions are imposed on boundary regions which name is not in the mesh boundary list')
        
        if self.params['clamped_bnd']:
            V1 = Compress(VectorH1(data.mesh, order=self.fes_order,
                        definedon=self.domain,
                        dirichlet_bbnd = data.mesh.BBoundaries(self.params['clamped_bnd'])))
        else:
            V1 = Compress(VectorH1(data.mesh, order=self.fes_order,
                        definedon=self.domain))
        V2 = Compress(VectorH1(data.mesh, order=self.fes_order,
                    definedon=self.domain))
        
        if data.mesh.dim == 2:
            dV = Compress(H1(data.mesh, order=1,\
                     definedon = self.domain))
        elif data.mesh.dim == 3:
            dV = Compress(VectorFacetSurface(data.mesh, order=1,\
                    definedon = self.domain))
            
        self.fes = V1*V2*dV

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.gfu_comp = self.gfu.components
        self.gfu_old_comp = self.gfu_old.components

        self.dX_h, self.Y_h = self.gfu_comp[0], self.gfu_comp[1]
        self.kappa_h = GridFunction(V2)
        if self.params['mc0']:
            self.kappa_h.Set(self.params['mc0'], definedon = self.domain)  
        else:
            compute_stab_mc(data, self.kappa_h, self.params)
        ns = specialcf.normal(data.mesh.dim)
        self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns,
                        definedon = data.mesh.Boundaries('.*'))  
            
        V_vol = VectorH1(data.mesh, order = self.fes_order)
        self.gfu_save = list(GridFunction(CompressCompound(V_vol*V_vol)).components)
        self.dX_save, self.kappa_save = self.gfu_save[0], self.gfu_save[1]
        self.dX_save.Set(self.dX_h, definedon = self.domain)
        self.kappa_save.Set(self.kappa_h, definedon = self.domain)

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)
            
    def GetLHS(self, data, trial, test):

        if data.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir })
        elif data.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir })

        # Compute mesh size, normal and tangential vectors 
        h = specialcf.mesh_size
        ns = specialcf.normal(data.mesh.dim)
        tE = specialcf.tangential(data.mesh.dim)
        if data.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        if data.mesh.dim == 2:
            dkappa = trial[2]*tE
            deta = test[2]*tE
            jump_dkappadn = (trial[1].Trace().Deriv()*nE-dkappa)
            jump_detadn = (test[1].Trace().Deriv()*nE-deta)

        elif data.mesh.dim == 3:
            jump_dkappadn = (trial[1].Trace().Deriv()*nE-trial[2].Trace())
            jump_detadn = (test[1].Trace().Deriv()*nE-test[2].Trace())

        lhs = -InnerProduct(grad(trial[1]).Trace(), grad(test[0]).Trace())*ds
        lhs += InnerProduct(trial[1], test[1])*ds_lumped
        lhs += (InnerProduct(grad(trial[0]).Trace(), grad(test[1]).Trace()))*ds

        lhs += self.params['stab']*h*InnerProduct(jump_dkappadn,jump_detadn)\
            *ds(element_boundary=True)
        
        return lhs
        
    def GetRHS(self, data, test):

        compute_stab_mc(data, self.kappa_h, self.params)
        ns = specialcf.normal(data.mesh.dim)
        self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns,
                        definedon = self.domain) 

        if data.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir })
        elif data.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir })

        tE = specialcf.tangential(data.mesh.dim)
        if data.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)
        Ps = Id(data.mesh.dim) - OuterProduct(ns, ns)

        rhs = -InnerProduct(Ps, grad(test[1]).Trace())*ds
        rhs += -self.params['sp_curv']*InnerProduct(ns, test[1])*ds
        if self.params['rhs']:
            rhs += InnerProduct(self.params['rhs'], test[1])*ds

        def D_s(chi, Ps):
            sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
            return sym
        if data.mesh.dim == 3:
            rhs += InnerProduct(Trace(grad(self.Y_h).Trace()),Trace(grad(test[0]).Trace()))*ds
            rhs += -2*InnerProduct(grad(self.Y_h).Trace().trans, D_s(test[0], Ps)*Ps.trans)*ds
            rhs += -self.params['sp_curv']*InnerProduct(self.kappa_h, grad(test[0]).Trace().trans*ns)*ds_lumped
            rhs += -0.5*InnerProduct((Norm(self.kappa_h - self.params['sp_curv']*ns)**2)*Ps,grad(test[0]).Trace())*ds_lumped
            rhs += InnerProduct(InnerProduct(self.Y_h, self.kappa_h)*Ps,grad(test[0]).Trace())*ds_lumped
        elif data.mesh.dim == 2 :
            rhs += InnerProduct((Norm(self.Y_h)**2)*Ps,grad(test[0]).Trace())*ds_lumped

        ## Addition for the open boundary
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
                gfu = GridFunction(data.mesh.deformation.space)
                gfu.vec.data = -1*data.mesh.deformation.vec.data
                rhs += InnerProduct(nE, test[1]) * gfF * ds(element_boundary=True, deformation=gfu)

        return rhs

    def GetMass(self, data, trial, test):

        if data.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir })
        elif data.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir })

        mass = InnerProduct(trial[0], test[0])/data.dt*ds_lumped

        return mass
    
    def GetMassOld(self, data, trial, test):

        mass = InnerProduct(CF(0), test[0])/data.dt*ds

        return mass
        
    def PreProcess(self, data):

        pass
    
    def PostProcess(self, data):

        pass
        
    def Update(self, data):

        self.kappa_h.Set(self.Y_h + self.params['sp_curv']*specialcf.normal(data.mesh.dim),
                         definedon=self.domain)

        self.gfu_old.vec.data = self.gfu.vec.data
        self.dX_save.Set(self.dX_h, definedon = self.domain)
        self.kappa_save.Set(self.kappa_h, definedon = self.domain)

        for save in self.save_error:
            save.Save(data, self)

        for save in self.save_solution:
            save.Save(data, self)

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

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with value ', str(value))

        print(60*'-', '\n')