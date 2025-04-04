from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check, compute_mc, compute_stab_mc
from cosmos.pdes.pde_tools import compute_error

class MeanCurvature(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs

        self.fields = 2

        accepted_keys = ['rhs', 'domain', 'name']
        defaults = [None, '.*', ['displacement', 'mean_curvature']]
        
        if kwargs:
            params_check(kwargs, accepted_keys, defaults)
            self.params = kwargs
        else:
            self.params = {}
            params_check(self.params, accepted_keys, defaults)    

        self.gfu = [CF((0, 0)), CF((0, 0))]

    def Initialize(self, data):

        self.dim = data.mesh.dim
        self.domain = data.mesh.Boundaries(self.params['domain'])
        self.name = self.params['name']
        
        V1 = Compress(VectorH1(data.mesh, order=self.fes_order,
                    definedon=self.domain))
            
        V2 = Compress(VectorH1(data.mesh, order=self.fes_order,
                    definedon=self.domain))
            
        self.fes = V1*V2

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.gfu_comp = self.gfu.components
        self.gfu_old_comp = self.gfu_old.components

        self.dX_h, self.kappa_h = self.gfu_comp[0], self.gfu_comp[1]

        compute_mc(data, self.kappa_h, self.params)
            
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

        lhs = -InnerProduct(trial[1], test[0])*ds_lumped
        lhs += InnerProduct(trial[1], test[1])*ds_lumped
        lhs += (InnerProduct(grad(trial[0]).Trace(), grad(test[1]).Trace()))*ds
        
        return lhs
        
    def GetRHS(self, data, test):

        # Compute mesh size, normal and tangential vectors 
        ns = specialcf.normal(data.mesh.dim)
        Ps = Id(data.mesh.dim) - OuterProduct(ns, ns)

        rhs = -InnerProduct(Ps, grad(test[1]).Trace())*ds
        if self.params['rhs']:
            rhs += InnerProduct(self.params['rhs'], test[1])*ds

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

        print('This is a solver for the Mean Curvature flow')
        print('It computes one step of it given the mesh')
        print('The algorithm is described in the article:')
        print('Stabilization for the mean curvature is implemented as in the article:')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')