# %%
from cosmos.utils.generate_surface_meshes import *
from cosmos.solvers.base import *
from ngsolve import *

class VectorPoissonSolver(BaseSolver):

    def __init__(self, mesh, fes_order=1, params = None, verbose = 0):
        """_summary_

        Args:
            mesh (_type_): _description_
            fes_order (int, optional): _description_. Defaults to 1.
            params (_type_, optional): _description_. Defaults to None.
            verbose (int, optional): _description_. Defaults to 0.
        """
        
        super().__init__(mesh, verbose=verbose)

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
        accepted_keys = ['rhs', 'boundary_c']
        defaults = [CF((0,)*self.mesh.dim), {}]  
        BaseSolver.__params_setup__(self.params, accepted_keys, defaults)

        # Initialize boundary conditions
        accepted_keys = ['dirichlet', 'neumann']
        defaults = [[None],
                    [['.*', CF((0,)*self.mesh.dim*self.mesh.dim, dims = (self.mesh.dim, self.mesh.dim))]]]  
        BaseSolver.__params_setup__(self.params['boundary_c'], accepted_keys, defaults)
            
    def __build_MAF__(self):
        """_summary_
        """

        self.dir_bnd = ''
        self.dirichlet_bnds = {}
        for elem in self.params['boundary_c']['dirichlet']: 

            if elem == None:

                self.dir_bnd = None
                break

            self.dir_bnd += elem[0] + '|'
            self.dirichlet_bnds[elem[0]] = elem[1]
        self.dir_cf = self.mesh.BoundaryCF(self.dirichlet_bnds, default= CF ((0,)*self.mesh.dim))

        neu_bnd = ''
        self.neumann_bnds = {}
        for elem in self.params['boundary_c']['neumann']: 
            neu_bnd += elem[0] + '|'
            self.neumann_bnds[elem[0]] = elem[1]
        neu_cf = self.mesh.BoundaryCF(self.neumann_bnds, \
            default=CF((0,)*self.mesh.dim*self.mesh.dim, dims = (self.mesh.dim, self.mesh.dim)))

        if self.dir_bnd!=None:

            V = VectorH1(self.mesh, order= self.fes_order, dirichlet = self.dir_bnd)
            self.fes = V
            u, v = self.fes.TnT()

        else:
            # In case of fully Neumann boundary conditions
            # self.A zero-mean solution is enforced through self.A lagrange multiplier
            V = H1(self.mesh, order= self.fes_order, dim = self.mesh.dim)
            Q = NumberSpace(self.mesh, dim = self.mesh.dim) # Lagrange multiplier for zero mean solution
            self.fes = V*Q
            (u, lam), (v, mu) = self.fes.TnT()

        self.A = BilinearForm(self.fes)
        self.A += InnerProduct(Grad(u), Grad(v))*dx
        if self.dir_bnd == None:
            self.A += (InnerProduct(lam,v)-InnerProduct(mu,u))*dx
        self.F = LinearForm(self.fes)
        self.F += self.params['rhs']*v*dx

        n = specialcf.normal(self.mesh.dim)
        self.F += neu_cf*n*v*ds(definedon = self.mesh.Boundaries(neu_bnd))

        # self.gfu is the grid function used for computations and containing all the variables
        # u_h is self.A pointer to the function I want the boundary conditions to be applied to
        self.gfu = GridFunction(self.fes)
        if self.dir_bnd == None:
            self.sol_h = [self.gfu.components[0]]
        else:
            self.sol_h = [self.gfu]
    
    def __call__(self):
        """_summary_
        """

        # Updating the parameters
        self.__update_params__()

        # Constructing (without assembling) the bilinear and linear forms
        self.__build_MAF__()

        res = self.F.vec.CreateVector()  
          
        with TaskManager():

            self.A.Assemble()
            self.F.Assemble()

            res = self.F.vec
            if self.dir_bnd != None: 
                self.sol_h[0].Set(self.dir_cf, definedon = self.mesh.Boundaries(self.dir_bnd))

                res.data += - self.A.mat * self.gfu.vec

                self.gfu.vec.data += self.A.mat.Inverse(freedofs=self.fes.FreeDofs()) * res
            else:

                self.gfu.vec.data = self.A.mat.Inverse(freedofs=self.fes.FreeDofs()) * self.F.vec

        yield

# %%
