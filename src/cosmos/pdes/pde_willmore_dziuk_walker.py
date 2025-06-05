from ngsolve import *
from cosmos.pdes.pde_mc_base import BaseMC
from cosmos.pdes.pde_tools import compute_error
import numpy as np
from cosmos.solvers.fields import Field
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady
from ngsolve.webgui import Draw

class WillmoreDziukWalker(BaseMC):

    def __init__(self, rhs = None, clamped_bnd = None, clamped_f = None,
                 domain:str = '.*', name = ['displacement', 'mean_curvature'],
                 mc0 = None,
                 time_scheme = BDF1(), postprocess = False):

        super().__init__()

        self.rhs = Field(rhs)
        self.clamped_bnd = clamped_bnd
        self.clamped_f = Field(clamped_f)
        self.domain = domain
        self.name = name
        self.mc0 = Field(mc0)
        self.time_scheme = time_scheme
        self.postprocess = postprocess

        self.nfields = 2
        self.displacement = self.gfu

    def Initialize(self, solverdata):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = solverdata.mesh.Boundaries(self.domain)
        
        if self.clamped_bnd:
            self.V1 = Compress(VectorH1(solverdata.mesh, order=self.fes_order,
                        definedon=self.domain,
                        dirichlet_bbnd = solverdata.mesh.BBoundaries(self.clamped_bnd)))
        else:
            self.V1 = Compress(VectorH1(solverdata.mesh, order=self.fes_order,
                        definedon=self.domain))
        self.V2 = Compress(H1(solverdata.mesh, order=self.fes_order,
                    definedon=self.domain))
            
        self.fes = self.V1*self.V2

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)

        self.displacement, self.mean_curvature = self.gfu.components 
        ns = specialcf.normal(solverdata.mesh.dim)
        if not self.mc0():
            self.Wein = ComputeW(solverdata)
            self.mean_curvature.Set(Trace(self.Wein), definedon = self.domain)
        else:
            self.mean_curvature.Set(self.mc0(), definedon = self.domain)
            
        V1_vol = VectorH1(solverdata.mesh, order = self.fes_order)
        V2_vol = H1(solverdata.mesh, order = self.fes_order)
        self.gfu_save = list(GridFunction(CompressCompound(V1_vol*V2_vol)).components)
        self.dX_save, self.kappa_save = self.gfu_save[0], self.gfu_save[1]
        self.dX_save.Set(self.displacement, definedon = self.domain)
        self.kappa_save.Set(self.mean_curvature, definedon = self.domain)

        self.X0 = GridFunction(self.V1)
        if solverdata.mesh.dim == 2:
            self.X0.Set(CF((x,y)), definedon=self.domain)
        elif solverdata.mesh.dim == 3:
            self.X0.Set(CF((x, y, z)), definedon=self.domain)
        self.displacement_tot = GridFunction(self.V1)

        for save in self.save_error:
            save.Initialize(solverdata, self)

        for save in self.save_solution:
            save.Initialize(solverdata, self)
            
    def GetLHS(self, solverdata, trial, test):

        ns = specialcf.normal(solverdata.mesh.dim)
        Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns)

        if solverdata.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = solverdata.ale.deformation)
        elif solverdata.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = solverdata.ale.deformation)

        lhs = 1/solverdata.dt*trial[0]*test[0]*ds_lumped
        lhs += -InnerProduct(grad(trial[1]).Trace(), grad(test[1]).Trace())*ds(deformation = solverdata.ale.deformation)
        lhs += InnerProduct(trial[1]*ns, test[0])*ds_lumped
        lhs += InnerProduct(grad(trial[0]).Trace(), grad(test[0]).Trace())*ds(deformation = solverdata.ale.deformation)

        lhs += -0.5*InnerProduct(InnerProduct(self.mean_curvature, self.mean_curvature)*trial[1],test[1])*ds_lumped
        
        return lhs
        
    def GetRHS(self, solverdata, test):

        ns = specialcf.normal(solverdata.mesh.dim)
        Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns)

        if solverdata.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = solverdata.ale.deformation)
        elif solverdata.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = solverdata.ale.deformation)

        rhs = -InnerProduct(Ps, grad(test[0]).Trace())*ds(deformation = solverdata.ale.deformation)
        if self.rhs():
            rhs += InnerProduct(self.rhs(), test[1])*ds(deformation = solverdata.ale.deformation)
        rhs +=  - InnerProduct(InnerProduct(self.Wein, self.Wein)*self.mean_curvature, test[1])*ds_lumped

        tE = specialcf.tangential(solverdata.mesh.dim)
        if solverdata.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)
        if self.clamped_bnd:
            if solverdata.mesh.dim == 3:
                gfF = GridFunction(FacetSurface(solverdata.mesh, order=0))
                gfF.Set(1, definedon=solverdata.mesh.BBoundaries(self.clamped_bnd))
            elif solverdata.mesh.dim == 2:
                gfF = GridFunction(H1(solverdata.mesh, order =1, definedon=solverdata.mesh.Boundaries('.*')))
                gfF.Set(1, definedon=solverdata.mesh.BBoundaries(self.clamped_bnd))
            
            if self.clamped_f():
                rhs += InnerProduct(self.clamped_f(), test[1]) * gfF * ds(element_boundary=True)
            else:
                rhs += InnerProduct(nE, test[1]) * gfF * ds(element_boundary=True)

        return rhs

    def GetMass(self, solverdata, trial, test):

        mass = CF(0)*ds

        return mass
    
    def PreProcess(self, solverdata):
        
        self.prev_gfu.append(self.gfu.vec.Copy())      
        if len(self.prev_gfu)>6:
            self.prev_gfu.pop(0)

        ns = specialcf.normal(solverdata.mesh.dim)
        solverdata.mesh.SetDeformation(solverdata.ale.deformation)
        self.Wein = ComputeW(solverdata)
        self.mean_curvature.Set(Trace(self.Wein), definedon = self.domain)
        solverdata.mesh.UnsetDeformation()
    
    def PostProcess(self, solverdata):

        super().PostProcess(solverdata)

        if self.postprocess:

            ns = specialcf.normal(solverdata.mesh.dim)
            n_h = GridFunction(VectorH1(solverdata.mesh, order=self.fes_order,
                        definedon=self.domain))
            solverdata.mesh.SetDeformation(self.displacement_tot)
            n_h.Set(ns, definedon=self.domain)
            solverdata.mesh.UnsetDeformation()

            V3 = H1(solverdata.mesh, order=self.fes_order,
                        definedon=self.domain)
            
            fes = self.V1*V3
            w_h = GridFunction(fes)
            A = BilinearForm(fes)
            (w, kappa), (eta, mu) = fes.TnT()
            A += InnerProduct(grad(w).Trace(), grad(eta).Trace())*ds
            A += -1*InnerProduct(kappa*n_h, eta)*ds
            A += 1*InnerProduct(w, mu*n_h)*ds
            F = LinearForm(fes)
            F += -1*InnerProduct(grad(self.X0).Trace(), grad(eta).Trace())*ds
            F += -1*InnerProduct(grad(self.displacement_tot).Trace(), grad(eta).Trace())*ds
            A.Assemble()
            F.Assemble()
            w_h.vec.data = A.mat.Inverse(freedofs = fes.FreeDofs())*F.vec
            self.displacement.vec.data += w_h.components[0].vec.data
            self.displacement_tot.vec.data += w_h.components[0].vec.data
        
    def Update(self, solverdata):

        solverdata.mesh.SetDeformation(solverdata.ale.deformation)

        self.dX_save.Set(self.displacement, definedon = self.domain)
        self.kappa_save.Set(self.mean_curvature, definedon = self.domain)

        for save in self.save_error:
            save.Save(solverdata, self)

        for save in self.save_solution:
            save.Save(solverdata, self)

        solverdata.mesh.UnsetDeformation()

    def get_error(self, solverdata, ex_sol, norm):

        err0 = compute_error(data=solverdata, gfu = self.displacement, u_ex=ex_sol[0],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        err1 = compute_error(data=solverdata, gfu = self.mean_curvature, u_ex=ex_sol[1],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err0, err1]

    
    def set_solution(self, value):

        self.displacement.Set(value[0], definedon=self.domain)
        self.mean_curvature.Set(value[1], definedon=self.domain)

    def get_solution(self):

        return [self.displacement, self.mean_curvature]

    def print_info(self):

        print(60*'-')

        print('This is a solver for the Willmore flow')
        print('It computes one step of it given the mesh')
        print('The algorithm is described in the article:')
        print('Stabilization for the mean curvature is implemented as in the article:')
        print('It uses conforming H-1 elements of order ', self.fes_order)

def ComputeW(solverdata):

    mesh = solverdata.mesh
    order = solverdata.mesh.GetCurveOrder()

    n = specialcf.normal(3)
    t = specialcf.tangential(3)
    mu = Cross(n,t)
    
    # Average normal vector
    gfF = GridFunction(VectorFacetSurface(mesh,order=order-1))
    gfF.Set(n, dual=True, definedon=mesh.Boundaries(".*"))
    
    fes = HDivDivSurface(mesh,order=order-1)
    sigma,tau = fes.TnT()
    sigma,tau = sigma.Trace(),tau.Trace()
    
    a = BilinearForm(fes, symmetric=True)
    a += InnerProduct(sigma,tau)*ds
    
    # Grad(n) = specialcf.Weingarten(3)
    f = LinearForm(fes)
    f += -1*(InnerProduct(Grad(n),tau))*ds \
            + -1*((pi/2-acos(Normalize(gfF)*mu))*tau*mu*mu)*ds(element_boundary=True)
    
    gflift = GridFunction(fes)
    
    with TaskManager():
        a.Assemble()
        f.Assemble()
        gflift.vec.data = a.mat.Inverse(fes.FreeDofs(),inverse="sparsecholesky")*f.vec
        
    return gflift