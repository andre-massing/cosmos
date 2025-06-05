from ngsolve import *
from cosmos.pdes.pde_mc_base import BaseMC
from cosmos.pdes.pde_tools import compute_error
import numpy as np
from cosmos.solvers.fields import Field
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady

class WillmoreDziukImplicit(BaseMC):

    def __init__(self, rhs = None, clamped_bnd = None, clamped_f = None,
                 domain:str = '.*', name = ['displacement', 'mean_curvature'],
                 mc_autoupdate = False, mc0 = None, kappa = None,
                 time_scheme = BDF1(), ale = False, postprocess = False):

        super().__init__()

        self.rhs = Field(rhs)
        self.clamped_bnd = clamped_bnd
        self.clamped_f = Field(clamped_f)
        self.domain = domain
        self.name = name
        self.mc_autoupdate = mc_autoupdate
        self.mc0 = Field(mc0)
        if not kappa:
            self.kappa= Field(1.0)
        else:
            self.kappa = Field(kappa)
        self.time_scheme = time_scheme
        self.ale = ale
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
        self.V2 = Compress(VectorH1(solverdata.mesh, order=self.fes_order,
                    definedon=self.domain))
            
        self.fes = self.V1*self.V2

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)

        self.displacement, self.mean_curvature = self.gfu.components 
        if not self.mc0():
            ComputeMC(solverdata, self.mean_curvature, self.clamped_bnd)
        else:
            self.mean_curvature.Set(self.mc0(), definedon = self.domain)
        ns = specialcf.normal(solverdata.mesh.dim)
            
        V_vol = VectorH1(solverdata.mesh, order = self.fes_order)
        self.gfu_save = list(GridFunction(CompressCompound(V_vol*V_vol)).components)
        self.dX_save, self.kappa_save = self.gfu_save[0], self.gfu_save[1]
        self.dX_save.Set(self.displacement, definedon = self.domain)
        self.kappa_save.Set(self.mean_curvature, definedon = self.domain)

        self.X0 = GridFunction(self.V1)
        if solverdata.mesh.dim == 2:
            self.X0.Set(CF((x,y)), definedon=self.domain)
        elif solverdata.mesh.dim == 3:
            self.X0.Set(CF((x, y, z)), definedon=self.domain)

        for save in self.save_error:
            save.Initialize(solverdata, self)

        for save in self.save_solution:
            save.Initialize(solverdata, self)
            
    def GetLHS(self, solverdata, trial, test):

        ns = specialcf.normal(solverdata.mesh.dim)
        Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns)

        def D_s(chi, Ps):
            sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
            return sym
        
        lhs = 1/solverdata.dt*trial[0]*test[0]*ds(deformation=solverdata.ale.deformation)
        lhs += -InnerProduct(grad(trial[1]).Trace(), grad(test[0]).Trace())*ds(deformation = solverdata.ale.deformation)
        lhs += InnerProduct(trial[1], test[1])*ds(deformation = solverdata.ale.deformation)
        lhs += InnerProduct(grad(trial[0]).Trace(), grad(test[1]).Trace())*ds(deformation = solverdata.ale.deformation)

        if solverdata.mesh.dim == 3:
            lhs += -1*InnerProduct(Trace(grad(trial[1]).Trace()),Trace(grad(test[0]).Trace()))*ds(deformation = solverdata.ale.deformation)
            lhs += 2*InnerProduct(grad(trial[1]).Trace(), D_s(test[0], Ps)*Ps)*ds(deformation = solverdata.ale.deformation)
            lhs += -0.5*InnerProduct(InnerProduct(self.mean_curvature, self.mean_curvature)*grad(trial[0]).Trace(),grad(test[0]).Trace())*ds(deformation = solverdata.ale.deformation)
        elif solverdata.mesh.dim == 2 :
            lhs += -0.5*InnerProduct(InnerProduct(self.mean_curvature, self.mean_curvature)*grad(trial[0]).Trace(),grad(test[0]).Trace())*ds(deformation = solverdata.ale.deformation)
        
        return lhs
        
    def GetRHS(self, solverdata, test):

        ns = specialcf.normal(solverdata.mesh.dim)
        Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns)

        rhs = -InnerProduct(Ps, grad(test[1]).Trace())*ds(deformation = solverdata.ale.deformation)
        if self.rhs():
            rhs += InnerProduct(self.rhs(), test[0])*ds(deformation = solverdata.ale.deformation)
        rhs +=  0.5*InnerProduct(InnerProduct(self.mean_curvature, self.mean_curvature)*grad(self.X0).Trace(),grad(test[0]).Trace())*ds(deformation = solverdata.ale.deformation)

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

        if self.mc_autoupdate:
            solverdata.mesh.SetDeformation(solverdata.ale.deformation)
            ComputeMC(solverdata, self.mean_curvature, self.clamped_bnd)
            solverdata.mesh.UnsetDeformation()
    
    def PostProcess(self, solverdata):

        super().PostProcess(solverdata)

        if self.postprocess:

            ns = specialcf.normal(solverdata.mesh.dim)
            n_h = GridFunction(self.V2)
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

        if self.ale:
            self.gfu.Set(solverdata.ale.deformation, definedon=self.domain)
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

def ComputeMC(solverdata, gfu, clamped_bnd=None):

    ns = specialcf.normal(solverdata.mesh.dim)
    tE = specialcf.tangential(solverdata.mesh.dim)
    if solverdata.mesh.dim == 2:
        nE = tE
    else:
        nE = Cross(ns, tE)
    Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns)

    fes0 = gfu.space
    gfu0 = GridFunction(fes0)
    kappa0, eta0 = fes0.TnT()
    A0 = BilinearForm(fes0)
    F0 = LinearForm(fes0)
    A0 += kappa0*eta0*ds
    A0.Assemble()
    F0 += -InnerProduct(Ps, grad(eta0).Trace())*ds

    if clamped_bnd:

        if solverdata.mesh.dim == 3:
            gfF = GridFunction(FacetSurface(solverdata.mesh, order=0))
            gfF.Set(1, definedon=solverdata.mesh.BBoundaries(clamped_bnd))
            F0 += InnerProduct(nE, eta0) * gfF * ds(element_boundary=True)

        elif solverdata.mesh.dim == 2:
            gfF = GridFunction(H1(solverdata.mesh, order =1,\
                    definedon=solverdata.mesh.Boundaries('.*')))
            gfF.Set(1, definedon=solverdata.mesh.BBoundaries(clamped_bnd))
            F0 += InnerProduct(gfF*nE, eta0) * ds(element_boundary=True)

    F0.Assemble()
    gfu0.vec.data = A0.mat.Inverse(fes0.FreeDofs())*F0.vec
    gfu.vec.data = gfu0.vec.data