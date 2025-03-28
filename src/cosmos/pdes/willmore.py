from ngsolve import *
from cosmos.pdes.base_pde import BasePDE
from cosmos.utils.tools import params_check, compute_error, compute_mc, compute_stab_mc
import os
import csv
from ngsolve.webgui import Draw
import numpy as np
import scipy.sparse as scipy

class Willmore(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs

        self.fields = 2

        accepted_keys = ['rhs', 'stab', 'sp_curv', 'clamped_bnd', 'domain',
                         'name', 'stationary']
        defaults = [CF(0), False, CF(0.0), {}, '.*',
                    ['displacement', 'mean_curvature'], False]
        
        params_check(self.params, accepted_keys, defaults)

        self.gfu = [CF((0, 0)), CF((0, 0))]

    def Initialize(self, mesh_data):

        self.dim = mesh_data['mesh'].dim
        
        if self.params['clamped_bnd']:
            if not set([self.params['clamped_bnd']]) <= set(mesh_data['boundary_mrk'] + ['.*']):
                raise ValueError('Dirichlet boundary conditions are imposed on boundary regions which name is not in the mesh boundary list')
        
        if self.params['clamped_bnd']:
            V1 = Compress(VectorH1(mesh_data['mesh'], order=self.fes_order,
                        definedon=mesh_data['mesh'].Boundaries(self.params['domain']),
                        dirichlet_bbnd = mesh_data['mesh'].BBoundaries(self.params['clamped_bnd'])))
        else:
            V1 = Compress(VectorH1(mesh_data['mesh'], order=self.fes_order,
                        definedon=mesh_data['mesh'].Boundaries(self.params['domain'])))
            
        V2 = Compress(VectorH1(mesh_data['mesh'], order=self.fes_order,
                    definedon=mesh_data['mesh'].Boundaries(self.params['domain'])))
            
        self.fes = V1*V2

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        self.dX_h, self.Y_h = self.gfu.components
        self.kappa_h = GridFunction(V2)

        if self.params['stab']:
            compute_stab_mc(mesh_data, self.kappa_h, self.params)
        else:
            compute_mc(mesh_data, self.kappa_h, self.params)
        ns = specialcf.normal(mesh_data['mesh'].dim)
        self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns,
                        definedon = mesh_data['mesh'].Boundaries('.*'))  
            
        V_vol = VectorH1(mesh_data['mesh'], order = self.fes_order)
        self.gfu_save = GridFunction(CompressCompound(V_vol*V_vol))
        self.dX_save, self.kappa_save = self.gfu_save.components
        self.dX_save.Set(self.dX_h, definedon = mesh_data['mesh'].Boundaries(self.params['domain']))
        self.kappa_save.Set(self.kappa_h, definedon = mesh_data['mesh'].Boundaries(self.params['domain']))

        if self.error_params:

            file_path = os.path.join(self.error_params['folderpath'], self.error_params['filename'])
            os.makedirs(self.error_params['folderpath'], exist_ok=True)

        if self.save_params:

            if mesh_data['mesh'].ne == 0:
                self.save_params['VorB'] = BND
            else:
                self.save_params['VorB'] = VOL

            file_path = os.path.join(self.save_params['folderpath'], self.save_params['filename'])
            os.makedirs(self.save_params['folderpath'], exist_ok=True)

            self.save_params['vtk'] = VTKOutput(mesh_data['mesh'],
                        coefs = [self.dX_save, self.kappa_save],
                        names = self.params['name'],
                        filename = file_path,
                        subdivision = self.save_params['subdivision'])
            
    def GetLHS(self, mesh_data, trial, test, dX):

        if mesh_data['mesh'].dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = dX)
        elif mesh_data['mesh'].dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = dX)

        # Compute mesh size, normal and tangential vectors 
        h = specialcf.mesh_size
        ns = specialcf.normal(mesh_data['mesh'].dim)
        tE = specialcf.tangential(mesh_data['mesh'].dim)
        if mesh_data['mesh'].dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)
        Ps = Id(mesh_data['mesh'].dim) - OuterProduct(ns, ns)

        lhs = -InnerProduct(grad(trial[1]).Trace(), grad(test[0]).Trace())*ds(deformation = dX)
        lhs += InnerProduct(trial[1], test[1])*ds_lumped
        lhs += (InnerProduct(grad(trial[0]).Trace(), grad(test[1]).Trace()))*ds(deformation = dX)
        
        return lhs
        
    def GetRHS(self, mesh_data, test, dX):

        mesh_data['mesh'].SetDeformation(dX)

        if self.params['stab']:
            compute_stab_mc(mesh_data, self.kappa_h, self.params)
        else:
            compute_mc(mesh_data, self.kappa_h, self.params)
        ns = specialcf.normal(mesh_data['mesh'].dim)
        self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns,
                        definedon = mesh_data['mesh'].Boundaries('.*')) 
        
        mesh_data['mesh'].UnsetDeformation()

        if mesh_data['mesh'].dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = dX)
        elif mesh_data['mesh'].dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = dX)

        ns = specialcf.normal(mesh_data['mesh'].dim)
        tE = specialcf.tangential(mesh_data['mesh'].dim)
        if mesh_data['mesh'].dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)
        Ps = Id(mesh_data['mesh'].dim) - OuterProduct(ns, ns)

        rhs = -InnerProduct(Ps, grad(test[1]).Trace())*ds(deformation = dX)
        rhs += -self.params['sp_curv']*InnerProduct(ns, test[1])*ds(deformation = dX)
        rhs += InnerProduct(self.params['rhs'], test[1])*ds(deformation = dX)

        def D_s(chi, Ps):
            sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
            return sym
        if mesh_data['mesh'].dim == 3:
            rhs += InnerProduct(Trace(grad(self.Y_h).Trace()),Trace(grad(test[0]).Trace()))*ds(deformation = dX)
            rhs += -2*InnerProduct(grad(self.Y_h).Trace().trans, D_s(test[0], Ps)*Ps.trans)*ds(deformation = dX)
            rhs += -self.params['sp_curv']*InnerProduct(self.kappa_h, grad(test[0]).Trace().trans*ns)*ds_lumped
            rhs += -0.5*InnerProduct((Norm(self.kappa_h - self.params['sp_curv']*ns)**2)*Ps,grad(test[0]).Trace())*ds_lumped
            rhs += InnerProduct(InnerProduct(self.Y_h, self.kappa_h)*Ps,grad(test[0]).Trace())*ds_lumped
        elif mesh_data['mesh'].dim == 2 :
            rhs += InnerProduct((Norm(self.Y_h)**2)*Ps,grad(test[0]).Trace())*ds_lumped

        ## Addition for the open boundary
        if self.params['clamped_bnd']:

            if mesh_data['mesh'].dim == 3:
                
                gfF = GridFunction(FacetSurface(mesh_data['mesh'], order=0))
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(self.params['clamped_bnd']))

                rhs += InnerProduct(nE, test[1]) * gfF * ds(element_boundary=True)

            elif mesh_data['mesh'].dim == 2:

                gfF = GridFunction(H1(mesh_data['mesh'], order =1, definedon=mesh_data['mesh'].Boundaries('.*')))
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(self.params['clamped_bnd']))

                rhs += InnerProduct(gfF*nE, test[1]) * ds(element_boundary=True)

        return rhs

    def GetMass(self, mesh_data, trial, test, dX, gfu):

        if mesh_data['mesh'].dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = dX)
        elif mesh_data['mesh'].dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = dX)

        if self.params['stationary']:
            factor = 0
        else:
            factor = 1

        mass = factor*InnerProduct(trial[0], test[0])/mesh_data['dt']*ds_lumped
        mass_gfu = InnerProduct(0*gfu.components[0], test[0])/mesh_data['dt']*ds_lumped

        return mass, mass_gfu
        
    def Update(self, mesh_data, dX):

        mesh_data['mesh'].SetDeformation(dX)
        self.kappa_h.Set(self.Y_h + self.params['sp_curv']*specialcf.normal(mesh_data['mesh'].dim),
                         definedon=mesh_data['mesh'].Boundaries('.*'))
        mesh_data['mesh'].UnsetDeformation()

        self.gfu_old.vec.data = self.gfu.vec.data

        self.dX_save.Set(self.dX_h, definedon = mesh_data['mesh'].Boundaries(self.params['domain']))
        self.kappa_save.Set(self.kappa_h, definedon = mesh_data['mesh'].Boundaries(self.params['domain']))

    def get_error(self, mesh_data):

        err0 = compute_error(data=mesh_data, gfu = self.dX_h, u_ex=self.error_params['ex_sol'][0],
                            norm = self.error_params['norm'], domain = self.params['domain'], 
                            VorB = BND)
        
        err1 = compute_error(data=mesh_data, gfu = self.kappa_h, u_ex=self.error_params['ex_sol'][1],
                            norm = self.error_params['norm'], domain = self.params['domain'], 
                            VorB = BND)
        
        return [err0, err1]

    
    def set_solution(self, mesh_data, value):

        self.dX_h.Set(value[0], definedon=mesh_data['mesh'].Boundaries(self.params['domain']))
        self.kappa_h.Set(value[1], definedon=mesh_data['mesh'].Boundaries(self.params['domain']))

    def get_solution(self):

        return [self.dX_h, self.kappa_h]
    
    def draw_solution(self, mesh_data, **kwargs):

        if 'dX' in mesh_data:
            dX = mesh_data['dX']
        else:
            dX = GridFunction(VectorH1(mesh_data['mesh']))

        Draw(self.dX_save, deformation = dX)
        Draw(self.kappa_save, deformation = dX) 

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