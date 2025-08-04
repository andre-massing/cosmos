from ngsolve import *
from cosmos.pdes.pde_mc_base import BaseMC
from cosmos.pdes.pde_tools import compute_error
import numpy as np
from cosmos.solvers.fields import Field
from cosmos.solvers.time_schemes import BDF1
from myngspy import *

class MCBGNStab(BaseMC):

    def __init__(self, rhs = None,
                 domain:str = '.*', name = ['displacement', 'mean_curvature'],
                 alpha = 1, beta = 0,
                 time_scheme = BDF1(), stab = 1e-3, postprocess = False):

        super().__init__()

        self.rhs = Field(rhs)
        self.domain = domain
        self.name = name
        self.alpha = alpha
        self.beta = beta
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
        
        V1 = VectorH1(solverdata.mesh, order=self.fes_order,
                definedon=self.domain)
        # if solverdata.mesh.dim == 2:
        #     dV = H1(solverdata.mesh, order=1,definedon = self.domain)
        # elif solverdata.mesh.dim == 3:
        dV = NormalFacetSurface(solverdata.mesh, order=0,\
                definedon = self.domain)
        V2 = H1(solverdata.mesh, order=self.fes_order,
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
        
        ds_lumped = ds(intrules = { TRIG: ir_trig }, deformation = solverdata.ale.deformation)
        ds0_el_lumped = ds(element_boundary=True, intrules = { TRIG: ir_trig })
                
        self.fes = CompressCompound(V1*V1*dV)
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        self.gfu = GridFunction(self.fes)

        self.displacement, self.mean_curvature, _ = self.gfu.components
        self.gf_sp_curv = GridFunction(V2)
        self.gf_rhs = GridFunction(V1)

        self.UpdateParams(solverdata, init=True)
            
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
                   + InnerProduct(self.trial[1], self.test[1])).Compile(true_compile, wait)*ds_lumped
        self.A += (-self.beta*InnerProduct(grad(self.trial[1]).Trace(), grad(self.test[0]).Trace())
                   +InnerProduct(grad(self.trial[0]).Trace(), grad(self.test[1]).Trace())).Compile(true_compile, wait)*ds(deformation = solverdata.ale.deformation)
        self.A += (-self.alpha*InnerProduct(self.trial[1], self.test[0])).Compile(true_compile, wait)*ds_lumped
        self.A += (self.stab*h*InnerProduct(jump_dkappadn,jump_detadn)).Compile(true_compile, wait)\
            *ds(element_boundary=True, deformation = solverdata.ale.deformation)
        
        self.F = LinearForm(self.fes)
        self.F += (-InnerProduct(Ps, grad(self.test[1]).Trace())).Compile(true_compile, wait)*ds(deformation = solverdata.ale.deformation)
        self.F += (InnerProduct(self.gf_rhs, self.test[0])).Compile(true_compile, wait)*ds_lumped


        Precond_A = Preconditioner(self.A, "multigrid")
        self.A.Assemble()
        self.invA = GMRESSolver(self.A.mat, Precond_A.mat, maxsteps = 1000)

        if self.postprocess:

            fes_pp = CompressCompound(V1*V2)
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
        if self.rhs():
            self.gf_rhs.Set(self.rhs(), dual = True, definedon = self.domain)
        solverdata.mesh.UnsetDeformation()
    
    def PostProcess(self, solverdata):

        super().PostProcess(solverdata)

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

        solverdata.mesh.SetDeformation(solverdata.ale.deformation)
        energy = Integrate(1, mesh=solverdata.mesh, VOL_or_BND=BND)
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