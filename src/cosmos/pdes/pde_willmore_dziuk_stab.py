from ngsolve import *
from cosmos.pdes.pde_mc_base import BaseMC
from cosmos.pdes.pde_tools import compute_error
import numpy as np
from cosmos.solvers.fields import Field
from cosmos.solvers.time_schemes import BDF1
from myngspy import *

class WillmoreDziukStab(BaseMC):

    def __init__(self, rhs = None, clamped_bnd = None, clamped_f = None,
                 domain:str = '.*', name = ['displacement', 'mean_curvature'],
                 sp_curv = None, kappa = 1, mc_autoupdate = False, mc0 = None,
                 time_scheme = BDF1(), stab = 1e-3, postprocess = False):

        super().__init__()

        self.rhs = Field(rhs)
        self.clamped_bnd = clamped_bnd
        self.clamped_f = Field(clamped_f)
        self.domain = domain
        self.name = name
        self.kappa = kappa
        if not sp_curv:
            self.sp_curv = Field(0.0)
        else:
            self.sp_curv = Field(sp_curv)
        self.mc_autoupdate = mc_autoupdate
        self.mc0 = Field(mc0)
        self.stab = stab
        self.time_scheme = time_scheme
        self.postprocess = postprocess
        if self.postprocess:
            self.mc_autoupdate = True

        self.nfields = 3
        self.displacement = self.gfu

    def Initialize(self, solverdata):

        true_compile = False
        wait = True

        if solverdata.mesh.dim == 2:
            raise ValueError('Stabilizaiton not implemented for 1D manifolds!')

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = solverdata.mesh.Boundaries(self.domain)
        
        if self.clamped_bnd:
            if not set([self.clamped_bnd]) <= set(solverdata.boundary_markers + ['.*']):
                raise ValueError('Clamped boundary conditions are imposed on non-existing boundary!')
        
        if self.clamped_bnd:
            V1 = VectorH1(solverdata.mesh, order=self.fes_order,
                definedon=self.domain,
                dirichlet_bbnd = solverdata.mesh.BBoundaries(self.clamped_bnd))
        else:
            V1 = VectorH1(solverdata.mesh, order=self.fes_order,
                definedon=self.domain)
        V2 = VectorH1(solverdata.mesh, order=self.fes_order,
            definedon=self.domain)
        # if solverdata.mesh.dim == 2:
        #     dV = H1(solverdata.mesh, order=1,definedon = self.domain)
        # elif solverdata.mesh.dim == 3:
        dV = NormalFacetSurface(solverdata.mesh, order=0,\
                definedon = self.domain)
        V3 = H1(solverdata.mesh, order=self.fes_order,
            definedon=self.domain)
        

        h = MyMeshSize()
        tE = specialcf.tangential(solverdata.mesh.dim)
        ns = specialcf.normal(solverdata.mesh.dim)
        Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns)
        if solverdata.mesh.dim == 2:
            nE = tE
            tEc = CF((-ns[1], ns[0]))
        else:
            nE = Cross(ns, tE)

        # ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        
        ds0_lumped = ds(intrules = { TRIG: ir_trig })
        ds_lumped = ds(intrules = { TRIG: ir_trig }, deformation = solverdata.ale.deformation)
        ds0_el_lumped = ds(element_boundary=True, intrules = { TRIG: ir_trig })

        fes0 = CompressCompound(V2*dV)
        self.gfu0 = GridFunction(fes0)
        (kappa0, dkappa0), (eta0, deta0) = fes0.TnT()
        # if solverdata.mesh.dim == 2:
        #     dkappa0 = dkappa0*tE
        #     deta0 = deta0*tE
        #     jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0*tEc)
        #     jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0*tEc)
        # elif solverdata.mesh.dim == 3:
        jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0.Trace())
        jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0.Trace())
        self.A0 = BilinearForm(fes0)
        self.F0 = LinearForm(fes0)

        self.A0 += (kappa0*eta0).Compile(true_compile, wait)*ds0_lumped
        # if solverdata.mesh.dim == 2:
        #     facet_space = H1(solverdata.mesh, order = 1, definedon=self.domain)
        # else:
        facet_space = FacetSurface(solverdata.mesh, order = 0, definedon = self.domain)
        if self.clamped_bnd:
            gfFone = GridFunction(facet_space)
            gfFone.Set(1, definedon=self.domain, dual = True)
            gfFBB = GridFunction(facet_space)
            gfFBB.Set(1, definedon=solverdata.mesh.BBoundaries(self.clamped_bnd))
            self.A0 += (self.stab*(gfFone-gfFBB)*h*InnerProduct(jump_dkappadn0,jump_detadn0)).Compile(true_compile, wait)\
                    *ds(element_boundary=True)
            # if solverdata.mesh.dim == 2:
            #     self.A0 +=  gfFBB*InnerProduct(dkappa0, deta0)\
            #                 *ds(element_boundary=True)
            #     self.F0 += InnerProduct(nE, eta0) * gfFBB * ds0_el_lumped
            # elif solverdata.mesh.dim == 3:
            self.A0 +=  (gfFBB*dkappa0.Trace()*deta0.Trace()).Compile(true_compile, wait)*ds(element_boundary=True)
            self.F0 += (InnerProduct(nE, eta0)*gfFBB).Compile(true_compile, wait)*ds0_el_lumped
        else:
            self.A0 += (self.stab*h*InnerProduct(jump_dkappadn0,jump_detadn0)).Compile(true_compile, wait)\
                    *ds(element_boundary=True)
        self.F0 += (-InnerProduct(Ps, grad(eta0).Trace())).Compile(true_compile, wait)*ds
            
        self.A0.Assemble()
        self.invA0 = self.A0.mat.Inverse(freedofs = fes0.FreeDofs())
        self.F0.Assemble()
                
        self.fes = CompressCompound(V1*V2*dV)
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        self.gfu = GridFunction(self.fes)

        self.displacement, self.Y_h, _ = self.gfu.components
        self.mean_curvature = GridFunction(V2)
        self.gf_sp_curv = GridFunction(V3)
        self.gf_rhs = GridFunction(V2)
        self.gf_clamped_f = GridFunction(V2)

        self.UpdateParams(solverdata, init=True)
        if self.mc0():
            self.mean_curvature.Set(self.mc0(), dual = True, definedon = self.domain)
            
        V_vol = VectorH1(solverdata.mesh, order = self.fes_order)
        self.gfu_save = list(GridFunction(CompressCompound(V_vol*V_vol)).components)
        self.dX_save, self.kappa_save = self.gfu_save[0], self.gfu_save[1]
        self.dX_save.Set(self.displacement, dual = True, definedon = self.domain)
        self.kappa_save.Set(self.mean_curvature, dual = True, definedon = self.domain)

        self.X0 = GridFunction(V1)
        if solverdata.mesh.dim == 2:
            self.X0.Set(CF((x,y)), dual = True, definedon=self.domain)
        elif solverdata.mesh.dim == 3:
            self.X0.Set(CF((x, y, z)), dual = True, definedon=self.domain)
        self.displacement_tot = GridFunction(V1)

        for save in self.save_error:
            save.Initialize(solverdata, self)
        for save in self.save_solution:
            save.Initialize(solverdata, self)

        self.A = BilinearForm(self.fes)
        
        # if solverdata.mesh.dim == 2:
        #     dkappa = self.trial[2]*tEc
        #     deta = self.test[2]*tEc
        #     jump_dkappadn = (self.trial[1].Trace().Deriv()*nE-dkappa)
        #     jump_detadn = (self.test[1].Trace().Deriv()*nE-deta)
        # elif solverdata.mesh.dim == 3:
        jump_dkappadn = (self.trial[1].Trace().Deriv()*nE-self.trial[2].Trace())
        jump_detadn = (self.test[1].Trace().Deriv()*nE-self.test[2].Trace())
        self.A += (1/solverdata.dt*self.trial[0]*self.test[0] 
                   + 1/self.kappa*InnerProduct(self.trial[1], self.test[1])).Compile(true_compile, wait)*ds_lumped
        self.A += (-InnerProduct(grad(self.trial[1]).Trace(), grad(self.test[0]).Trace()) +
                   InnerProduct(grad(self.trial[0]).Trace(), grad(self.test[1]).Trace())).Compile(true_compile, wait)*ds(deformation = solverdata.ale.deformation)
        
        self.F = LinearForm(self.fes)
        self.F += (-InnerProduct(Ps, grad(self.test[1]).Trace())).Compile(true_compile, wait)*ds(deformation = solverdata.ale.deformation)
        self.F += (-self.gf_sp_curv*InnerProduct(ns, self.test[1])).Compile(true_compile, wait)*ds_lumped
        self.F += (InnerProduct(self.gf_rhs, self.test[0])).Compile(true_compile, wait)*ds_lumped

        def D_s(chi, Ps):
            sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
            return sym
        # elif solverdata.mesh.dim == 2 :
        #     self.F += InnerProduct(InnerProduct(self.Y_h, self.mean_curvature)*Ps,grad(self.test[0]).Trace())*ds_lumped
        # if solverdata.mesh.dim == 3:
        self.F += (InnerProduct(Trace(grad(self.Y_h).Trace()),Trace(grad(self.test[0]).Trace()))).Compile(true_compile, wait)*ds(deformation = solverdata.ale.deformation)
        self.F += (-2*InnerProduct(grad(self.Y_h).Trace().trans, D_s(self.test[0], Ps)*Ps.trans)).Compile(true_compile, wait)*ds(deformation = solverdata.ale.deformation)
        self.F += (-self.kappa*InnerProduct(self.gf_sp_curv*self.mean_curvature, grad(self.test[0]).Trace().trans*ns)).Compile(true_compile, wait)*ds_lumped
        self.F += (-0.5*InnerProduct(self.kappa*(Norm(self.mean_curvature - self.gf_sp_curv*ns)**2)*Ps,grad(self.test[0]).Trace())).Compile(true_compile, wait)*ds_lumped
        self.F += (InnerProduct(InnerProduct(self.Y_h, self.mean_curvature)*Ps,grad(self.test[0]).Trace())).Compile(true_compile, wait)*ds_lumped

        if self.clamped_bnd:
            self.A += (self.stab*(gfFone - gfFBB)*h*InnerProduct(jump_dkappadn,jump_detadn)).Compile(true_compile, wait)\
            *ds(element_boundary=True, deformation = solverdata.ale.deformation)
            # if solverdata.mesh.dim == 2:
            #     self.A +=  (gfFBB*InnerProduct(self.trial[2],self.test[2])).Compile(true_compile, True)\
            #                 *ds(element_boundary=True, deformation = solverdata.ale.deformation)
            # elif solverdata.mesh.dim == 3:
            self.A +=  (gfFBB*self.trial[2].Trace()*self.test[2].Trace()).Compile(true_compile, wait)*ds(element_boundary=True, deformation = solverdata.ale.deformation)
            if self.clamped_f():
                self.F += (InnerProduct(self.clamped_f(), self.test[1])*gfFBB).Compile(true_compile, wait)*ds0_el_lumped
            else:
                self.F += (InnerProduct(nE, self.test[1])*gfFBB).Compile(true_compile, wait)*ds0_el_lumped
        else:
            self.A += (self.stab*h*InnerProduct(jump_dkappadn,jump_detadn)).Compile(true_compile, wait)\
            *ds(element_boundary=True, deformation = solverdata.ale.deformation)

        Precond_A = Preconditioner(self.A, "multigrid")
        self.A.Assemble()
        self.invA = GMRESSolver(self.A.mat, Precond_A.mat, maxsteps = 1000)

        if self.postprocess:

            fes_pp = CompressCompound(V1*V3)
            self.A_pp = BilinearForm(fes_pp, symmetric = True)
            self.F_pp = LinearForm(fes_pp)
            self.w_h = GridFunction(fes_pp)
            (w, kappa), (eta, mu) = fes_pp.TnT()

            self.A_pp += (InnerProduct(grad(w).Trace(), grad(eta).Trace())).Compile(true_compile, wait)*ds
            self.A_pp += (-1*InnerProduct(kappa*ns, eta)).Compile(true_compile, wait)*ds(deformation=self.displacement_tot, intrules = { TRIG : ir_trig })
            self.A_pp += (-1*InnerProduct(w, mu*ns)).Compile(true_compile, wait)*ds(deformation=self.displacement_tot, intrules = { TRIG : ir_trig })
            self.A_pp.Assemble()
            Precond_pp = Preconditioner(self.A_pp, "multigrid")
            self.A_pp.Assemble()
            self.invA_pp = CGSolver(self.A_pp.mat, Precond_pp.mat, maxsteps = 1000)

            self.F_pp += (-1*InnerProduct(Ps, grad(eta).Trace())).Compile(true_compile, wait)*ds
            self.F_pp += (-1*InnerProduct(grad(self.displacement_tot).Trace(), grad(eta).Trace())).Compile(true_compile, wait)*ds

    def UpdateParams(self, solverdata, init = False):

        solverdata.mesh.SetDeformation(solverdata.ale.deformation)
        ns = specialcf.normal(solverdata.mesh.dim)
        if self.mc_autoupdate or init:
            self.ComputeStabMC()
            self.Y_h.Set(self.kappa*(self.mean_curvature - self.sp_curv()*ns), dual = True, definedon = self.domain)
        self.gf_sp_curv.Set(self.sp_curv(), dual = True, definedon = self.domain)
        if self.rhs():
            self.gf_rhs.Set(self.rhs(), dual = True, definedon = self.domain)
        if self.clamped_f():
            self.gf_clamped_f.Set(self.clamped_f(), dual = True, definedon = self.domain)
        solverdata.mesh.UnsetDeformation()
    
    def PostProcess(self, solverdata):

        super().PostProcess(solverdata)

        solverdata.mesh.SetDeformation(solverdata.ale.deformation)
        ns = specialcf.normal(solverdata.mesh.dim)
        self.mean_curvature.Set(1/self.kappa*self.Y_h + self.gf_sp_curv*ns, dual = True, definedon = self.domain)
        solverdata.mesh.UnsetDeformation()
        self.displacement_tot.Set(self.displacement+solverdata.ale.deformation, dual = True, definedon = self.domain)

        if self.postprocess:

            self.A_pp.Assemble()
            self.F_pp.Assemble()

            self.w_h.vec.data = self.invA_pp*self.F_pp.vec
            self.displacement.vec.data += self.w_h.components[0].vec.data
        
    def Update(self, solverdata):

        solverdata.mesh.SetDeformation(solverdata.ale.deformation)

        self.dX_save.Set(self.displacement, dual = True, definedon = self.domain)
        self.kappa_save.Set(self.mean_curvature, dual = True, definedon = self.domain)

        for save in self.save_error:
            save.Save(solverdata, self)

        for save in self.save_solution:
            save.Save(solverdata, self)

        solverdata.mesh.UnsetDeformation()

    def GetEnergy(self, solverdata):

        ns = specialcf.normal(solverdata.mesh.dim)
        solverdata.mesh.SetDeformation(solverdata.ale.deformation)
        energy = 0.5*Integrate(Norm(self.mean_curvature - self.gf_sp_curv*ns)**2, mesh=solverdata.mesh, VOL_or_BND=BND)
        solverdata.mesh.UnsetDeformation()

        return energy

    def get_error(self, solverdata, ex_sol, norm):

        err0 = compute_error(data=solverdata, gfu = self.displacement, u_ex=ex_sol[0],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        err1 = compute_error(data=solverdata, gfu = self.mean_curvature, u_ex=ex_sol[1],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err0, err1]

    
    def set_solution(self, value):

        self.displacement.Set(value[0], dual = True, definedon=self.domain)
        self.mean_curvature.Set(value[1], dual = True,  definedon=self.domain)

    def get_solution(self):

        return [self.displacement, self.mean_curvature]

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

    def ComputeStabMC(self):

        self.A0.Assemble()
        self.invA0.Update()
        self.F0.Assemble()

        self.gfu0.vec.data = self.invA0*self.F0.vec
        self.mean_curvature.vec.data = self.gfu0.components[0].vec.data
