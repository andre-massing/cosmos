from ngsolve import *
from cosmos.pdes.pde_mc_base import BaseMC
from cosmos.pdes.pde_tools import compute_error
import numpy as np
from cosmos.solvers.fields import Field
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady

class WillmoreDziukStab(BaseMC):

    def __init__(self, rhs = None, clamped_bnd = None, clamped_f = None,
                 domain:str = '.*', name = ['displacement', 'mean_curvature'],
                 sp_curv = None, kappa = None, mc_autoupdate = False, mc0 = None,
                 time_scheme = BDF1(), stab = 1e-3, ale = False, postprocess = False,
                 volume_preserving = False):

        super().__init__()

        self.rhs = Field(rhs)
        self.clamped_bnd = clamped_bnd
        self.clamped_f = Field(clamped_f)
        self.domain = domain
        self.name = name
        if not sp_curv:
            self.sp_curv = Field(0.0)
        else:
            self.sp_curv = Field(sp_curv)
        if not kappa:
            self.kappa= Field(1.0)
        else:
            self.kappa = Field(kappa)
        self.mc_autoupdate = mc_autoupdate
        self.mc0 = Field(mc0)
        self.stab = stab
        self.time_scheme = time_scheme
        self.ale = ale
        self.postprocess = postprocess
        self.vp = volume_preserving

        self.nfields = 3
        self.displacement = self.gfu

    def Initialize(self, solverdata):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = solverdata.mesh.Boundaries(self.domain)
        
        if self.clamped_bnd:
            if not set([self.clamped_bnd]) <= set(solverdata.boundary_markers + ['.*']):
                raise ValueError('Clamped boundary conditions are imposed on non-existing boundary!')
        
        if self.clamped_bnd:
            self.V1 = VectorH1(solverdata.mesh, order=self.fes_order,
                        definedon=self.domain,
                        dirichlet_bbnd = solverdata.mesh.BBoundaries(self.clamped_bnd))
        else:
            self.V1 = VectorH1(solverdata.mesh, order=self.fes_order,
                        definedon=self.domain)
        self.V2 = VectorH1(solverdata.mesh, order=self.fes_order,
                    definedon=self.domain)
        if solverdata.mesh.dim == 2:
            dV = H1(solverdata.mesh, order=1,\
                     definedon = self.domain)
        elif solverdata.mesh.dim == 3:
            dV = NormalFacetSurface(solverdata.mesh, order=0,\
                    definedon = self.domain)
            
        self.fes = CompressCompound(self.V1*self.V2*dV)

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)

        self.displacement, self.Y_h, _ = self.gfu.components
        self.mean_curvature = GridFunction(self.V2)
        if not self.mc0():
            ComputeStabMC(solverdata, self.mean_curvature, self.clamped_bnd, self.stab, self.domain)
        else:
            self.mean_curvature.Set(self.mc0(), definedon = self.domain)
        ns = specialcf.normal(solverdata.mesh.dim)
        self.Y_h.Set(self.kappa()*(self.mean_curvature - self.sp_curv()*ns), definedon = self.domain)
            
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
        self.displacement_tot = GridFunction(self.V1)


        for save in self.save_error:
            save.Initialize(solverdata, self)

        for save in self.save_solution:
            save.Initialize(solverdata, self)

        dim = solverdata.mesh.dim
        J = specialcf.JacobianMatrix(dim, dim-1) 
        if dim == 3:
            area = sqrt(Det(J.trans*J))/2
            self.h_f = sqrt(area/pi)
        elif dim == 2:
            self.h_f = Norm(J[:, 0])

        tE = specialcf.tangential(solverdata.mesh.dim)
        ns = specialcf.normal(solverdata.mesh.dim)
        if solverdata.mesh.dim == 2:
            self.nE = tE
            self.tEc = CF((-ns[1], ns[0]))
        else:
            self.nE = Cross(ns, tE)
            
    def GetLHS(self, solverdata, trial, test):
        
        if solverdata.mesh.dim == 2:
            dkappa = trial[2]*self.tEc
            deta = test[2]*self.tEc
            jump_dkappadn = (trial[1].Trace().Deriv()*self.nE-dkappa)
            jump_detadn = (test[1].Trace().Deriv()*self.nE-deta)
        elif solverdata.mesh.dim == 3:
            jump_dkappadn = (trial[1].Trace().Deriv()*self.nE-trial[2].Trace())
            jump_detadn = (test[1].Trace().Deriv()*self.nE-test[2].Trace())

        if solverdata.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = solverdata.ale.deformation)
        elif solverdata.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = solverdata.ale.deformation)

        lhs = 1/solverdata.dt*trial[0]*test[0]*ds_lumped
        lhs += -InnerProduct(grad(trial[1]).Trace(), grad(test[0]).Trace())*ds(deformation = solverdata.ale.deformation)
        lhs += 1/self.kappa()*InnerProduct(trial[1], test[1])*ds_lumped
        lhs += (InnerProduct(grad(trial[0]).Trace(), grad(test[1]).Trace()))*ds(deformation = solverdata.ale.deformation)

        if solverdata.mesh.dim == 2:
            facet_space = H1(solverdata.mesh, order = 1, definedon=self.domain)
        else:
            facet_space = FacetSurface(solverdata.mesh, order = 0)
        gfFone = GridFunction(facet_space)
        gfFone.Set(1, definedon=self.domain, dual = True)
        gfFBB = GridFunction(facet_space)
        if self.clamped_bnd:
            gfFBB.Set(1, definedon=solverdata.mesh.BBoundaries(self.clamped_bnd))

        lhs += self.stab*(gfFone - gfFBB)*self.h_f*InnerProduct(jump_dkappadn,jump_detadn)\
            *ds(element_boundary=True, deformation = solverdata.ale.deformation)
        
        if solverdata.mesh.dim == 2:
            lhs +=  gfFBB*InnerProduct(trial[2],test[2])\
                        *ds(element_boundary=True, deformation = solverdata.ale.deformation)
        elif solverdata.mesh.dim == 3:
            lhs +=  gfFBB*trial[2].Trace()*test[2].Trace()*ds(element_boundary=True, deformation = solverdata.ale.deformation)
        
        return lhs
        
    def GetRHS(self, solverdata, test):

        ns = specialcf.normal(solverdata.mesh.dim)
        Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns)

        if solverdata.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = solverdata.ale.deformation)
            ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir })
        elif solverdata.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = solverdata.ale.deformation)
            ds_el_lumped = ds(element_boundary=True, intrules = { TRIG : ir })

        rhs = -InnerProduct(Ps, grad(test[1]).Trace())*ds(deformation = solverdata.ale.deformation)
        rhs += -self.sp_curv()*InnerProduct(ns, test[1])*ds_lumped
        if self.rhs():
            rhs += InnerProduct(self.rhs(), test[0])*ds_lumped

        def D_s(chi, Ps):
            sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
            return sym
        if solverdata.mesh.dim == 3:
            rhs += InnerProduct(Trace(grad(self.Y_h).Trace()),Trace(grad(test[0]).Trace()))*ds(deformation = solverdata.ale.deformation)
            rhs += -2*InnerProduct(grad(self.Y_h).Trace().trans, D_s(test[0], Ps)*Ps.trans)*ds(deformation = solverdata.ale.deformation)
            rhs += -self.kappa()*InnerProduct(self.sp_curv()*self.mean_curvature, grad(test[0]).Trace().trans*ns)*ds_lumped
            rhs += -0.5*InnerProduct(self.kappa()*(Norm(self.mean_curvature - self.sp_curv()*ns)**2)*Ps,grad(test[0]).Trace())*ds_lumped
            rhs += InnerProduct(InnerProduct(self.Y_h, self.mean_curvature)*Ps,grad(test[0]).Trace())*ds_lumped
        elif solverdata.mesh.dim == 2 :
            rhs += InnerProduct(InnerProduct(self.Y_h, self.mean_curvature)*Ps,grad(test[0]).Trace())*ds_lumped

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
                rhs += InnerProduct(self.clamped_f(), test[1]) * gfF * ds_el_lumped
            else:
                rhs += InnerProduct(nE, test[1]) * gfF * ds_el_lumped

        if self.vp:

            Q = NumberSpace(solverdata.mesh)
            lam, mu = Q.TnT()
            lambda_h = GridFunction(Q)
            A = BilinearForm(Q)
            A += InnerProduct(self.mean_curvature, self.mean_curvature)*lam*mu*ds_lumped

            F = LinearForm(Q)
            if self.rhs():
                F += -InnerProduct(self.rhs(), self.mean_curvature)*mu*ds_lumped
            F += -InnerProduct(grad(self.Y_h).Trace(), grad(self.mean_curvature).Trace())*mu*ds(deformation = solverdata.ale.deformation)
            if solverdata.mesh.dim == 3:
                F += -InnerProduct(Trace(grad(self.Y_h).Trace()),Trace(grad(self.mean_curvature).Trace()))*mu*ds(deformation = solverdata.ale.deformation)
                F += 2*InnerProduct(grad(self.Y_h).Trace().trans, D_s(self.mean_curvature, Ps)*Ps.trans)*mu*ds(deformation = solverdata.ale.deformation)
                F += self.kappa()*InnerProduct(self.sp_curv()*self.mean_curvature, grad(self.mean_curvature).Trace().trans*ns)*mu*ds_lumped
                F += 0.5*InnerProduct(self.kappa()*(Norm(self.mean_curvature - self.sp_curv()*ns)**2)*Ps,grad(self.mean_curvature).Trace())*mu*ds_lumped
                F += -InnerProduct(InnerProduct(self.Y_h, self.mean_curvature)*Ps,grad(self.mean_curvature).Trace())*mu*ds_lumped
            elif solverdata.mesh.dim == 2 :
                F += -InnerProduct(InnerProduct(self.Y_h, self.mean_curvature)*Ps,grad(self.mean_curvature).Trace())*mu*ds_lumped

            A.Assemble()
            F.Assemble()

            lambda_h.vec.data = A.mat.Inverse(freedofs = Q.FreeDofs())*F.vec

            rhs += InnerProduct(lambda_h*self.mean_curvature, test[0])*ds_lumped

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
            ns = specialcf.normal(solverdata.mesh.dim)
            ComputeStabMC(solverdata, self.mean_curvature, self.clamped_bnd, self.stab, self.domain)
            self.Y_h.Set(self.kappa()*(self.mean_curvature - self.sp_curv()*ns), definedon = self.domain)
            solverdata.mesh.UnsetDeformation()
    
    def PostProcess(self, solverdata):

        super().PostProcess(solverdata)

        ns = specialcf.normal(solverdata.mesh.dim)
        solverdata.mesh.SetDeformation(solverdata.ale.deformation)
        self.mean_curvature.Set(1/self.kappa()*self.Y_h + self.sp_curv()*ns, definedon = self.domain)
        solverdata.mesh.UnsetDeformation()
        self.displacement_tot.Set(self.displacement+solverdata.ale.deformation_old, definedon = self.domain)

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

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')

def ComputeStabMC(solverdata, gfu, clamped_bnd, stab, domain):

    if solverdata.mesh.dim == 2:
        ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ds_lumped = ds(intrules = { SEGM : ir })
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir })
    elif solverdata.mesh.dim == 3:
        ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        ds_lumped = ds(intrules = { TRIG : ir })
        ds_el_lumped = ds(element_boundary=True, intrules = { TRIG : ir })

    if solverdata.mesh.dim == 2:
        dV = H1(solverdata.mesh, order=1,\
                    definedon = domain)
    elif solverdata.mesh.dim == 3:
        dV = NormalFacetSurface(solverdata.mesh, order=0,\
                definedon = domain)
        
    ns = specialcf.normal(solverdata.mesh.dim)
    tE = specialcf.tangential(solverdata.mesh.dim)
    if solverdata.mesh.dim == 2:
        nE = tE
        tEc = CF((-ns[1], ns[0]))
    else:
        nE = Cross(ns, tE)
    Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns)
    dim = solverdata.mesh.dim
    J = specialcf.JacobianMatrix(dim, dim-1) 
    if dim == 3:
        area = sqrt(Det(J.trans*J))/2
        h_f = sqrt(area/pi)
    elif dim == 2:
        h_f = Norm(J[:, 0])

    fes0 = gfu.space*dV
    gfu0 = GridFunction(fes0)
    (kappa0, dkappa0), (eta0, deta0) = fes0.TnT()
    if solverdata.mesh.dim == 2:
        dkappa0 = dkappa0*tE
        deta0 = deta0*tE
        jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0*tEc)
        jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0*tEc)
    elif solverdata.mesh.dim == 3:
        jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0.Trace())
        jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0.Trace())
    A0 = BilinearForm(fes0)
    F0 = LinearForm(fes0)
    A0 += kappa0*eta0*ds_lumped

    if solverdata.mesh.dim == 2:
        facet_space = H1(solverdata.mesh, order = 1, definedon=domain)
    else:
        facet_space = FacetSurface(solverdata.mesh, order = 0, definedon = domain)

    if clamped_bnd:
        gfFone = GridFunction(facet_space)
        gfFone.Set(1, definedon=domain, dual = True)
        gfFBB = GridFunction(facet_space)
        gfFBB.Set(1, definedon=solverdata.mesh.BBoundaries(clamped_bnd))
        A0 += stab*(gfFone-gfFBB)*h_f*InnerProduct(jump_dkappadn0,jump_detadn0)\
            *ds(element_boundary=True)
    
        if solverdata.mesh.dim == 2:
            A0 +=  gfFBB*InnerProduct(dkappa0, deta0)\
                        *ds(element_boundary=True)
            F0 += InnerProduct(gfFBB*nE, eta0) * ds_el_lumped
        elif solverdata.mesh.dim == 3:
            A0 +=  gfFBB*dkappa0.Trace()*deta0.Trace()*ds(element_boundary=True)
            F0 += InnerProduct(nE, eta0) * gfFBB * ds_el_lumped
    else:
        A0 += stab*h_f*InnerProduct(jump_dkappadn0,jump_detadn0)\
            *ds(element_boundary=True)
        
    A0.Assemble()
    F0 += -InnerProduct(Ps, grad(eta0).Trace())*ds
    F0.Assemble()
    gfu0.vec.data = A0.mat.Inverse(fes0.FreeDofs())*F0.vec
    gfu.vec.data = gfu0.components[0].vec.data