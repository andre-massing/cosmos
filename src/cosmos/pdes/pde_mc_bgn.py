from ngsolve import *
from cosmos.pdes.pde_mc_base import BaseMC
from cosmos.pdes.pde_tools import compute_error
from cosmos.solvers.fields import Field
from cosmos.solvers.time_schemes import BDF1

class MCBGN(BaseMC):

    def __init__(self, rhs = None,
                 domain:str = '.*', name = ['displacement', 'mean_curvature'],
                 kappa = None, mc0 = None,
                 time_scheme = BDF1(), ale = False, postprocess = False):

        super().__init__()

        self.rhs = Field(rhs)
        self.domain = domain
        self.name = name
        if not kappa:
            self.kappa= Field(1.0)
        else:
            self.kappa = Field(kappa)
        self.mc0 = Field(mc0)
        self.time_scheme = time_scheme
        self.postprocess = postprocess

        self.nfields = 2
        self.displacement = self.gfu

    def Initialize(self, solverdata):

        true_compile = False

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = solverdata.mesh.Boundaries(self.domain)


        V1 = VectorH1(solverdata.mesh, order=self.fes_order,
                        definedon=self.domain)
        V2 = VectorH1(solverdata.mesh, order=self.fes_order,
                    definedon=self.domain)
        V3 = H1(solverdata.mesh, order=self.fes_order,
            definedon=self.domain)
        
        tE = specialcf.tangential(solverdata.mesh.dim)
        ns = specialcf.normal(solverdata.mesh.dim)
        Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns)
        if solverdata.mesh.dim == 2:
            nE = tE
            tEc = CF((-ns[1], ns[0]))
        else:
            nE = Cross(ns, tE)
        
        if solverdata.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = solverdata.ale.deformation)
            ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir })
        elif solverdata.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = solverdata.ale.deformation)
            ds_el_lumped = ds(element_boundary=True, intrules = { TRIG : ir })
            
        self.fes = CompressCompound(V1*V2)
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        self.gfu = GridFunction(self.fes)

        self.displacement, self.mean_curvature = self.gfu.components 
        self.n_h = GridFunction(V2)
        self.n_aux = GridFunction(V2)
        self.gf_kappa = GridFunction(V3)
        self.gf_rhs = GridFunction(V2)
        
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

        self.A += (1/solverdata.dt*self.trial[0]*self.test[0]
                   -self.gf_kappa*InnerProduct(self.trial[1], self.test[0])
                   +InnerProduct(self.trial[1], self.test[1])).Compile(true_compile, True)*ds_lumped
        self.A += (InnerProduct(grad(self.trial[0]).Trace(), grad(self.test[1]).Trace())).Compile(true_compile, True)*ds(deformation = solverdata.ale.deformation)

        self.F = LinearForm(self.fes)

        self.F += -InnerProduct(Ps, grad(self.test[1]).Trace())*ds(deformation = solverdata.ale.deformation)
        self.F += InnerProduct(self.gf_rhs, self.test[0])*ds_lumped

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())

        if self.postprocess:

            fes_pp = V1*V3
            self.A_pp = BilinearForm(fes_pp, symmetric = True)
            self.F_pp = LinearForm(fes_pp)
            self.w_h = GridFunction(fes_pp)
            (w, kappa), (eta, mu) = fes_pp.TnT()

            self.A_pp += (InnerProduct(grad(w).Trace(), grad(eta).Trace())).Compile(true_compile, True)*ds
            self.A_pp += (-1*InnerProduct(kappa*self.n_h, eta)).Compile(true_compile, True)*ds(deformation=self.displacement_tot)
            self.A_pp += (-1*InnerProduct(w, mu*self.n_h)).Compile(true_compile, True)*ds(deformation=self.displacement_tot)
            self.A_pp.Assemble()
            self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

            self.F_pp += -1*InnerProduct(Ps, grad(eta).Trace())*ds
            self.F_pp += -1*InnerProduct(grad(self.displacement_tot).Trace(), grad(eta).Trace())*ds
    
    def UpdateParams(self, solverdata, init = False):

        solverdata.mesh.SetDeformation(solverdata.ale.deformation)
        ns = specialcf.normal(solverdata.mesh.dim)
        self.n_aux.Set(ns, dual = True, definedon=self.domain)
        self.n_h.Set(Normalize(self.n_aux), dual = True, definedon=self.domain)
        self.gf_kappa.Set(self.kappa(), dual = True, definedon = self.domain)
        if self.rhs():
            self.gf_rhs.Set(self.rhs(), dual = True, definedon = self.domain)
        solverdata.mesh.UnsetDeformation()
    
    def PostProcess(self, solverdata):

        super().PostProcess(solverdata)

        self.displacement_tot.Set(self.displacement+solverdata.ale.deformation, dual = True, definedon = self.domain)

        if self.postprocess:

            ns = specialcf.normal(solverdata.mesh.dim)
            solverdata.mesh.SetDeformation(self.displacement_tot)
            self.n_aux.Set(ns, dual = True, definedon=self.domain)
            self.n_h.Set(Normalize(self.n_aux), dual = True, definedon=self.domain)
            solverdata.mesh.UnsetDeformation()

            self.A_pp.Assemble()
            self.invA_pp.Update()
            self.F_pp.Assemble()

            self.w_h.vec.data = self.invA_pp*self.F_pp.vec
            self.displacement.vec.data += self.w_h.components[0].vec.data
            self.displacement_tot.vec.data += self.w_h.components[0].vec.data
            
        
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
        self.mean_curvature.Set(value[1], dual = True, definedon=self.domain)

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