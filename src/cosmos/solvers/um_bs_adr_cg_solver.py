# %%
from cosmos.utils.generate_surface_meshes import *
from cosmos.solvers.base import *
from ngsolve import *

class um_sb_ADR_CGSolver(UnsteadySolver):

    def __init__(self, mesh, fes_order=1, dt=0.1, t=Parameter(0.0), T=1.0, params = {}, verbose = 0):
        """_summary_

        Args:
            mesh (_type_): _description_
            fes_order (int, optional): _description_. Defaults to 1.
            dt (float, optional): _description_. Defaults to 0.1.
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
        """

        # Initialize the parameters
        accepted_keys = ['rhs_v',  'rhs_s', 'convection_v', 'convection_s',
                'k_diffusion_v', 'k_diffusion_s', 'k_reaction_v', 'k_reaction_s',
                'initial_c_v', 'initial_c_s', 'displacement', 'coupling_coefs',
                'boundary_c']
        defaults = [CF(0.0), CF(0.0), CF((0,)*self.mesh.dim), CF((0,)*self.mesh.dim),
                    CF(0.0), CF(0.0), CF(0.0), CF(0.0),
                    CF(0.0), CF(0.0), CF((0,)*self.mesh.dim), [CF(0.0), CF(0.0)],
                    {}]
        BaseSolver.__params_setup__(self.params, accepted_keys, defaults)

        # Initialize boundary conditions
        accepted_keys = ['convective_flux', 'diffusive_flux']
        defaults = [['.*', CF((0,)*self.mesh.dim)],
                    ['.*', CF((0,)*self.mesh.dim)]]  
        BaseSolver.__params_setup__(self.params['boundary_c'], accepted_keys, defaults)

    def __build_MAF__(self):
        """_summary_
        """

        # Setting up the finite element space
        fes_v = H1(self.mesh, order=self.fes_order, dgjumps = True)
        fes_s = H1(self.mesh, order=self.fes_order, definedon=self.mesh.Boundaries('.*'))
        self.fes = fes_v*fes_s
        V = VectorH1(self.mesh, order=self.fes_order)
        (u_v, u_s), (v_v, v_s) = self.fes.TnT()

        # Creating the bilinear and linear forms
        self.A = BilinearForm(self.fes)
        self.M = BilinearForm(self.fes, symmetric = True)
        self.F = LinearForm(self.fes)
        self.M_old = BilinearForm(self.fes, symmetric = True)

        # Creating the grid functions
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.displ_h = GridFunction(V)
        self.displ_h_old = GridFunction(V)

        # Auxiliary functions for the bilinear forms
        n = specialcf.normal(self.mesh.dim)
        h = specialcf.mesh_size

        ## VOLUME PART

        diffusion_v = self.params['k_diffusion_v']*grad(u_v)*grad(v_v) * dx(deformation = self.displ_h)
        reaction_v = self.params['k_reaction_v'] * u_v*v_v * dx(deformation = self.displ_h)

        # CIP stabilization for convection part
        S_int = 0.5*Norm(self.params['convection_v']*n) # Parameter corresponding to upwind stabilization
        jump_u = n*(grad(u_v) - (grad(u_v)).Other())
        jump_v = n*(grad(v_v) - (grad(v_v)).Other())
        convection_v = -self.params['convection_v']*grad(v_v) * u_v *dx(deformation = self.displ_h) \
            + h**2*S_int*jump_u*jump_v*dx(skeleton=True, deformation = self.displ_h)
        
        coupling_v = (self.params['coupling_coefs'][0]*u_v \
                      - self.params['coupling_coefs'][1]*u_s)*v_v*ds(deformation = self.displ_h)

        self.A += diffusion_v + reaction_v + convection_v + coupling_v

        ## SURFACE PART
        
        diffusion_s = self.params['k_diffusion_s']*grad(u_s).Trace()*grad(v_s).Trace() * ds(deformation = self.displ_h)
        reaction_s = self.params['k_reaction_s']*u_s*v_s*ds(deformation = self.displ_h)
        convection_s = -self.params['convection_s']*grad(v_s).Trace() * u_s *ds(deformation = self.displ_h)
        coupling_s = -(self.params['coupling_coefs'][0]*u_v \
                        - self.params['coupling_coefs'][1]*u_s)*v_s*ds(deformation = self.displ_h)

        self.A += diffusion_s + reaction_s + convection_s + coupling_s

        # Time mass matrix
        self.M += 1/self.dt*u_v*v_v*dx(deformation = self.displ_h)
        self.M += 1/self.dt*u_s*v_s*ds(deformation = self.displ_h)

        # Old time mass matrix
        self.M_old += 1/self.dt*u_v*v_v*dx(deformation = self.displ_h_old)
        self.M_old += 1/self.dt*u_s*v_s*ds(deformation = self.displ_h_old)

        # Right-hand side
        self.F += self.params['rhs_v']*v_v*dx(deformation = self.displ_h)
        self.F += self.params['rhs_s']*v_s*ds(deformation = self.displ_h)

        self.sol_h = [self.gfu.components[0], self.gfu.components[1]]

        self.sol_h[0].Set(self.params['initial_c_v'])
        self.sol_h[1].Set(self.params['initial_c_s'], definedon = self.mesh.Boundaries(".*"))
        self.displ_h_old.Set(self.params['displacement'])
        self.displ_h.Set(self.params['displacement'])

    def __call__(self):
        """_summary_
        """

        # Updating the parameters
        self.__update_params__()

        # Constructing (without assembling) the bilinear and linear forms
        self.__build_MAF__()

        yield

        while self.t.Get()<self.T - 0.5 * self.dt.Get():

            self.__initialize_step__()

            self.__solve_step__()

            self.__finalize_step__()

            yield

    
    def __initialize_step__(self):
        """_summary_
        """

        self.__update_params__()

        self.gfu_old.vec.data = self.gfu.vec.data
        self.displ_h_old.vec.data = self.displ_h.vec.data

    def __solve_step__(self):
        """_summary_
        """

        with TaskManager():

            # Building the forms
            self.M_old.Assemble()
            self.F.Assemble()

            res = self.F.vec.CreateVector()
            # Previous time-step
            res.data =  self.M_old.mat*self.gfu_old.vec

            self.t.Set(self.t.Get() + self.dt.Get())
            self.displ_h.Set(self.params['displacement'])

            self.A.Assemble()
            self.M.Assemble()
            self.F.Assemble()

            # Right-hand side
            res.data += self.F.vec

            Mstar = self.M.mat.CreateMatrix()
            Mstar.AsVector().data = self.M.mat.AsVector() + self.A.mat.AsVector()

            # Possible dirichlet boundary conditions
            self.sol_h[0].Set(0)
            self.sol_h[1].Set(0, definedon = self.mesh.Boundaries(".*"))
            res.data -= Mstar*self.gfu.vec

            # corresponds to M* = M + A
            invMstar = Mstar.Inverse(freedofs=self.fes.FreeDofs())

            self.gfu.vec.data += invMstar*res

    def __finalize_step__(self):
        """_summary_
        """

        pass
# %%
