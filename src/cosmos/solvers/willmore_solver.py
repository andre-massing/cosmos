# %%
from cosmos.utils.generate_surface_meshes import *
from cosmos.solvers.base import *
from ngsolve import *

class WillmoreSolver(UnsteadySolver):

    def __init__(self, mesh, fes_order=1, dt=Parameter(0.1), t=Parameter(0.0), T=1.0, params = {}, verbose = 0):
        """_summary_

        Args:
            mesh (_type_): _description_
            fes_order (int, optional): _description_. Defaults to 1.
            dt (_type_, optional): _description_. Defaults to Parameter(0.1).
            t (_type_, optional): _description_. Defaults to Parameter(0.0).
            T (float, optional): _description_. Defaults to 1.0.
            params (dict, optional): _description_. Defaults to {}.
            verbose (int, optional): _description_. Defaults to 0.
        """
        
        super().__init__(mesh, dt = dt, t = t, T=T, verbose=verbose)

        self.fes_order = fes_order
        self.params = params

        # Updating the parameters
        self.__update_params__()

        # Constructing (without assembling) the bilinear and linear forms
        self.__build_MAF__()

    def __update_params__(self):
        """_summary_

        Raises:
            Exception: _description_
        """

        # Initialize the parameters
        accepted_keys = ['rhs', 'domain_name', 'clamped_bnd', 'displacement', 'initial_mc_type',
                          'postprocessing_type', 'stabilized_mc', 'gamma_stab', 'spontaneous_curv']
        defaults = [CF( (0,)*self.mesh.dim ), '.*', None, CF((0,)*self.mesh.dim), 0, None, False, 1e-3, 0]
        BaseSolver.__params_setup__(self.params, accepted_keys, defaults)

        accepted_keys = [None, 'harmonic', 'harmonic_step']
        if self.params['postprocessing_type'] not in accepted_keys:
            raise Exception('The only acceptable keys for -postprocessing type- are:', accepted_keys)

    def __build_MAF__(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        if self.params['clamped_bnd'] == None:
            V1_0 = VectorH1(self.mesh, order=self.fes_order,
                        definedon=self.mesh.Boundaries(self.params['domain_name']))
        else:
            V1_0 = VectorH1(self.mesh, order=self.fes_order,
                            definedon=self.mesh.Boundaries(self.params['domain_name']),
                            dirichlet_bbnd = self.mesh.BBoundaries(self.params['clamped_bnd']))
        V1 = VectorH1(self.mesh, order=self.fes_order,
                      definedon=self.mesh.Boundaries(self.params['domain_name']))
        V2 = H1(self.mesh, order=self.fes_order,
                definedon=self.mesh.Boundaries(self.params['domain_name']))

        self.displ_h = GridFunction(V1_0)
        self.displ_h_old = GridFunction(V1_0)
        self.X_m = GridFunction(V1_0)
        self.displ_normal = GridFunction(V2)

        if self.mesh.dim == 3:
            Ident = CF((x,y,z))
        elif self.mesh.dim == 2:
            Ident = CF((x,y))
        self.X_m.Set(Ident, definedon = self.mesh.Boundaries(self.params['domain_name']))

        def gradient(phi):

            return Grad(phi).Trace()
        
        if self.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(definedon=self.mesh.Boundaries(self.params['domain_name']),\
                           intrules = { SEGM : ir }, deformation=self.displ_h)
        elif self.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(definedon=self.mesh.Boundaries(self.params['domain_name']),\
                           intrules = { TRIG : ir }, deformation=self.displ_h)
        
        # BGN SOLVER

        self.bgn_solver(V1_0, V1, gradient, ds_lumped)

        # POSSIBLE MESH REDISTRIBUTION
        if self.params['postprocessing_type'] != None:
            self.mesh_redistribution(V1_0, V2, gradient)
        
        self.sol_h = [self.displ_h, self.kappa_h]

    def mesh_redistribution(self, V1_0, V2, gradient):
        """_summary_

        Args:
            V1_0 (_type_): _description_
            V2 (_type_): _description_
            gradient (_type_): _description_
        """

        nu = specialcf.normal(self.mesh.dim)

        self.fes2 = V1_0*V2

        (u2, p),(v2, q) = self.fes2.TnT()

        self.dXk2_h = GridFunction(self.fes2)
        self.dX2_h, self.kappa2_h = self.dXk2_h.components

        if self.params['postprocessing_type'] == 'harmonic':

            ds_pp = ds(definedon=self.mesh.Boundaries(self.params['domain_name']))

        elif self.params['postprocessing_type'] == 'harmonic_step':

            ds_pp = ds(definedon=self.mesh.Boundaries(self.params['domain_name']),\
                       deformation = self.displ_h_old)


        self.M2 = BilinearForm(self.fes2)
        self.M2 += InnerProduct(gradient(u2), gradient(v2))*ds_pp
        self.M2 += - p*nu*v2*ds(deformation = self.displ_h)
        self.M2 += - q*nu*u2*ds(deformation = self.displ_h)

        self.F2 = LinearForm(self.fes2)
        self.F2 += (-InnerProduct(gradient(self.X_m), \
                                            gradient(v2)))*ds_pp

    def bgn_solver(self, V1_0, V1, gradient, ds_lumped):
        """_summary_

        Args:
            V1_0 (_type_): _description_
            V1 (_type_): _description_
            gradient (_type_): _description_
            ds_lumped (_type_): _description_

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """

        if self.fes_order>1:
            raise Exception("Lumped mass not defined for fes_order>1 !")
        
        def gradient(phi):

            return Grad(phi).Trace()
        
        if self.params['stabilized_mc']:

            self.__stab_mc_bgn__(V1_0, V1, gradient, ds_lumped)

        else:

            self.__nostab_mc_bgn__(V1_0, V1, gradient, ds_lumped)
        
        self.__initialize_mc__(V1, gradient, ds_lumped)
        
        return
    
    def __stab_mc_bgn__(self, V1_0, V1, gradient, ds_lumped):
        """_summary_

        Args:
            V1_0 (_type_): _description_
            V1 (_type_): _description_
            gradient (_type_): _description_
            ds_lumped (_type_): _description_

        Returns:
            _type_: _description_
        """

        ds_def = ds(deformation = self.displ_h,\
                     definedon=self.mesh.Boundaries(self.params['domain_name']))

        # Compute mesh size, normal and tangential vectors 
        h = specialcf.mesh_size
        ns = specialcf.normal(self.mesh.dim)
        tE = specialcf.tangential(self.mesh.dim)
        if self.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)
        Ps = Id(self.mesh.dim) - OuterProduct(ns, ns)
        Qs = OuterProduct(ns, ns)
        
        def D_s(chi):
            sym = 0.5*Ps*(gradient(chi)+gradient(chi).trans)*Ps
            return sym
        
        if self.mesh.dim == 3:

            dV = VectorFacetSurface(self.mesh, order=1,\
                    definedon = self.mesh.Boundaries(self.params['domain_name']))

        elif self.mesh.dim == 2:

            dV = H1(self.mesh, order=1,\
                     definedon = self.mesh.Boundaries(self.params['domain_name']))


        self.fes = V1_0*V1*dV
        (dX, Y, dkappa), (chi, eta, deta) = self.fes.TnT()

        self.gfu1_h = GridFunction(self.fes)
        self.dX1_h, self.Y_h, dkappa1_h = self.gfu1_h.components
        self.kappa_h = GridFunction(V1)

        if self.mesh.dim == 2:

            dkappa = dkappa*tE
            deta = deta*tE

            jump_dkappadn = (Y.Trace().Deriv()*nE-dkappa)
            jump_detadn = (eta.Trace().Deriv()*nE-deta)

        elif self.mesh.dim == 3:

            jump_dkappadn = (Y.Trace().Deriv()*nE-dkappa.Trace())
            jump_detadn = (eta.Trace().Deriv()*nE-deta.Trace())

        self.M = BilinearForm(self.fes)
        self.F = LinearForm(self.fes)

        self.M += InnerProduct(dX, chi)/self.dt*ds_lumped
        self.M += -InnerProduct(gradient(Y), gradient(chi))*ds_def

        if self.mesh.dim == 3:
            self.F += InnerProduct(Trace(gradient(self.Y_h)),Trace(gradient(chi)))*ds_def
            self.F += -2*InnerProduct(gradient(self.Y_h).trans, D_s(chi)*Ps.trans)*ds_def
            self.F += -self.params['spontaneous_curv']*InnerProduct(self.kappa_h, gradient(chi).trans*ns)*ds_lumped
            self.F += -0.5*InnerProduct((Norm(self.kappa_h - self.params['spontaneous_curv']*ns)**2)*Ps,gradient(chi))*ds_lumped
            self.F += InnerProduct(InnerProduct(self.Y_h, self.kappa_h)*Ps,gradient(chi))*ds_lumped
        elif self.mesh.dim == 2 :
            self.F += InnerProduct((Norm(self.Y_h)**2)*Ps,gradient(chi))*ds_lumped

        self.M += InnerProduct(Y, eta)*ds_def
        self.M += (InnerProduct(gradient(dX), gradient(eta)))*ds_def

        gamma_E = self.params['gamma_stab']
        self.M += gamma_E*h*InnerProduct(jump_dkappadn,jump_detadn)\
            *ds(definedon=self.mesh.Boundaries(self.params['domain_name']),
                element_boundary=True, deformation = self.displ_h)

        self.F += -InnerProduct(Ps, gradient(eta))*ds_def

        self.F += -self.params['spontaneous_curv']*InnerProduct(ns, eta)*ds_def

        self.F += InnerProduct(self.params['rhs'], eta)*ds_def

        ## Addition for the open boundary
        if self.params['clamped_bnd'] != None:

            if self.mesh.dim == 3:
                gfF = GridFunction(FacetSurface(self.mesh, order=0))
                gfF.Set(1, definedon=self.mesh.BBoundaries(self.params['clamped_bnd']))

                self.F += InnerProduct(nE, eta) * gfF * ds(element_boundary=True)

            elif self.mesh.dim == 2:

                gfF = GridFunction(H1(self.mesh, order =1, definedon=self.mesh.Boundaries(self.params['domain_name'])))
                gfF.Set(1, definedon=self.mesh.BBoundaries(self.params['clamped_bnd']))

                self.F += InnerProduct(gfF*nE, eta) * ds(definedon = self.mesh.Boundaries(self.params['domain_name'])
                            , element_boundary=True)
    
    def __nostab_mc_bgn__(self, V1_0, V1, gradient, ds_lumped):
        """_summary_

        Args:
            V1_0 (_type_): _description_
            V1 (_type_): _description_
            gradient (_type_): _description_
            ds_lumped (_type_): _description_

        Returns:
            _type_: _description_
        """

        ds_def = ds(definedon=self.mesh.Boundaries(self.params['domain_name']),\
                    deformation = self.displ_h)

        ns = specialcf.normal(self.mesh.dim)
        tE = specialcf.tangential(self.mesh.dim)
        if self.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)
        Ps = Id(self.mesh.dim) - OuterProduct(ns, ns)
        Qs = OuterProduct(ns, ns)
        
        def D_s(chi):
            sym = 0.5*Ps*(Grad(chi).Trace()+Grad(chi).Trace().trans)*Ps
            return sym

        self.fes = V1_0*V1
        (dX, Y), (chi, eta) = self.fes.TnT()

        self.gfu1_h = GridFunction(self.fes)
        self.dX1_h, self.Y_h = self.gfu1_h.components
        self.kappa_h = GridFunction(V1)

        self.M = BilinearForm(self.fes)
        self.F = LinearForm(self.fes)

        self.M += InnerProduct(dX, chi)/self.dt*ds_lumped # To be corrected with the correct one!!!!
        self.M += -InnerProduct(gradient(Y), gradient(chi))*ds_def

        if self.mesh.dim == 3:
            self.F += InnerProduct(Trace(gradient(self.Y_h)),Trace(gradient(chi)))*ds_def
            self.F += -2*InnerProduct(gradient(self.Y_h).trans, D_s(chi)*Ps.trans)*ds_def
            self.F += -self.params['spontaneous_curv']*InnerProduct(self.kappa_h, gradient(chi).trans*ns)*ds_lumped
            self.F += -0.5*InnerProduct((Norm(self.kappa_h - self.params['spontaneous_curv']*ns)**2)*Ps,gradient(chi))*ds_lumped
            self.F += InnerProduct(InnerProduct(self.Y_h, self.kappa_h)*Ps,gradient(chi))*ds_lumped
        elif self.mesh.dim == 2 :
            self.F += InnerProduct((Norm(self.Y_h)**2)*Ps,gradient(chi))*ds_lumped

        self.M += InnerProduct(Y, eta)*ds_lumped
        self.M += (InnerProduct(gradient(dX), gradient(eta)))*ds_def

        self.F += -self.params['spontaneous_curv']*InnerProduct(ns, eta)*ds_def

        self.F += -InnerProduct(Ps, gradient(eta))*ds_lumped

        self.F += InnerProduct(self.params['rhs'], eta)*ds_def

        ## Addition for the open boundary
        if self.params['clamped_bnd'] != None:

            if self.mesh.dim == 3:

                gfF = GridFunction(FacetSurface(self.mesh, order=0))
                gfF.Set(1, definedon=self.mesh.BBoundaries(self.params['clamped_bnd']))

                self.F += InnerProduct(nE, eta) * gfF \
                    * ds_lumped(definedon=self.mesh.Boundaries(self.params['domain_name']),
                                                           element_boundary=True)

            elif self.mesh.dim == 2:

                gfF = GridFunction(H1(self.mesh, order =1, definedon=self.mesh.Boundaries(self.params['domain_name'])))
                gfF.Set(1, definedon=self.mesh.BBoundaries(self.params['clamped_bnd']))

                self.F += InnerProduct(gfF*nE, eta) \
                    * ds_lumped(definedon = self.mesh.Boundaries(self.params['domain_name'])
                            , element_boundary=True)
    
    def __initialize_mc__(self, V1, gradient, ds_lumped):
        """_summary_

        Args:
            V1 (_type_): _description_
            gradient (_type_): _description_
            ds_lumped (_type_): _description_
        """

        h = specialcf.mesh_size
        ns = specialcf.normal(self.mesh.dim)
        tE = specialcf.tangential(self.mesh.dim)
        if self.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)
        Ps = Id(self.mesh.dim) - OuterProduct(ns, ns)

        if self.mesh.dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation=self.displ_h)
        elif self.mesh.dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation=self.displ_h)

        if self.params['initial_mc_type'] == 0:

            (kappa0, eta0) = V1.TnT()
            M0 = BilinearForm(V1)
            l0 = LinearForm(V1)
            M0 += kappa0*eta0*ds_lumped
            M0.Assemble()
            l0 += -InnerProduct(Ps, gradient(eta0))\
                *ds(definedon = self.mesh.Boundaries(self.params['domain_name']))

            if self.params['clamped_bnd'] != None:

                if self.mesh.dim == 3:

                    gfF = GridFunction(FacetSurface(self.mesh, order=0))
                    gfF.Set(1, definedon=self.mesh.BBoundaries(self.params['clamped_bnd']))

                    l0 += InnerProduct(nE, eta0) * gfF \
                        * ds(definedon = self.mesh.Boundaries(self.params['domain_name'])
                                , element_boundary=True)

                elif self.mesh.dim == 2:

                    gfF = GridFunction(H1(self.mesh, order =1, definedon=self.mesh.Boundaries(self.params['domain_name'])))
                    gfF.Set(1, definedon=self.mesh.BBoundaries(self.params['clamped_bnd']))

                    l0 += InnerProduct(gfF*nE, eta0) \
                        * ds_lumped(definedon = self.mesh.Boundaries(self.params['domain_name'])
                                , element_boundary=True)

            l0.Assemble()

            self.kappa_h.vec.data = M0.mat.Inverse(V1.FreeDofs())*l0.vec
            self.Y_h.Set(self.kappa_h - self.params['spontaneous_curv']*ns,
                         definedon = self.mesh.Boundaries(self.params['domain_name']))


        elif self.params['initial_mc_type'] == 1:

            if self.mesh.dim == 3:

                dV = VectorFacetSurface(self.mesh, order=1,\
                     definedon = self.mesh.Boundaries(self.params['domain_name']))

            elif self.mesh.dim == 2:

                dV = H1(self.mesh, order=1,\
                     definedon = self.mesh.Boundaries(self.params['domain_name']))

            W = V1*dV
            W_h = GridFunction(W)
            
            (kappa0, dkappa0), (eta0, deta0) = W.TnT()

            if self.mesh.dim == 2:

                dkappa0 = dkappa0*tE
                deta0 = deta0*tE

                jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0)
                jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0)

            elif self.mesh.dim == 3:

                jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0.Trace())
                jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0.Trace())

            M0 = BilinearForm(W)
            l0 = LinearForm(W)
            M0 += kappa0*eta0*ds  # Adding traces here?
            gamma_E = self.params['gamma_stab']
            M0 += gamma_E*h*InnerProduct(jump_dkappadn0,jump_detadn0)\
                *ds(definedon = self.mesh.Boundaries(self.params['domain_name']), element_boundary=True)
            M0.Assemble()

            l0 += -InnerProduct(Ps, Grad(eta0).Trace())*ds

            if self.params['clamped_bnd'] != None:

                if self.mesh.dim == 3:
                    gfF = GridFunction(FacetSurface(self.mesh, order=0))
                    gfF.Set(1, definedon=self.mesh.BBoundaries(self.params['clamped_bnd']))

                    l0 += InnerProduct(nE, eta0) * gfF * ds(element_boundary=True)

                elif self.mesh.dim == 2:

                    gfF = GridFunction(H1(self.mesh, order =1,\
                         definedon=self.mesh.Boundaries(self.params['domain_name'])))
                    gfF.Set(1, definedon=self.mesh.BBoundaries(self.params['clamped_bnd']))

                    l0 += InnerProduct(gfF*nE, eta0) \
                        * ds(definedon = self.mesh.Boundaries(self.params['domain_name'])
                                , element_boundary=True)

            l0.Assemble()
            W_h.vec.data = M0.mat.Inverse(W.FreeDofs())*l0.vec

            self.kappa_h.vec.data = W_h.components[0].vec.data
            self.Y_h.Set(self.kappa_h - self.params['spontaneous_curv']*ns,
                          definedon = self.mesh.Boundaries(self.params['domain_name']))

        else:

            print('Initialization using Schoberl algorithm is still to be implemented!')

        
    def __call__(self):
        """_summary_
        """

        if self.verbose>0:
            print('The solver is called')

        # Updating the parameters
        self.__update_params__()

        # Constructing (without assembling) the bilinear and linear forms
        self.__build_MAF__()

        if self.verbose>0:
            print('The solver has been initialized')

        yield

        while self.t.Get()<self.T- 0.5 * self.dt.Get():

            if self.verbose>0:
                print('Starts a temporal loop step')

            self.__initialize_step__()

            self.__solve_step__()

            self.__finalize_step__()

            if self.verbose>0:
                print('The solver has finished the temporal loop step')

            yield 

    def __initialize_step__(self):
        """_summary_
        """

        self.__update_params__()

        self.displ_h_old.vec.data = self.displ_h.vec.data

    def __solve_step__(self):
        """_summary_
        """

        with TaskManager():

            self.M.Assemble()
            Minv = self.M.mat.Inverse(self.fes.FreeDofs(), inverse="umfpack")

            self.F.Assemble()

            self.gfu1_h.vec.data = Minv*self.F.vec
            self.mesh.SetDeformation(self.displ_h)
            self.kappa_h.Set(self.Y_h + self.params['spontaneous_curv']*specialcf.normal(self.mesh.dim), 
                             definedon=self.mesh.Boundaries(self.params['domain_name']))
            self.displ_normal.Set(self.displ_h*specialcf.normal(self.mesh.dim),
                                 definedon=self.mesh.Boundaries(self.params['domain_name']))
            self.mesh.UnsetDeformation()

            self.displ_h.vec.data += self.dX1_h.vec
            self.X_m.vec.data += self.dX1_h.vec
            

            if self.params['postprocessing_type'] != None:

                self.M2.Assemble() 
                M2inv = self.M2.mat.Inverse(self.fes2.FreeDofs(), inverse="umfpack")

                self.F2.Assemble()

                self.dXk2_h.vec.data = M2inv*self.F2.vec
                self.displ_h.vec.data += self.dX2_h.vec
                self.X_m.vec.data += self.dX2_h.vec

    def __finalize_step__(self):
        """_summary_
        """

        self.t.Set(self.t.Get() + self.dt.Get())

# %%
