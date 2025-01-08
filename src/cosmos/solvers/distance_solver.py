# %%
from cosmos.utils.generate_surface_meshes import *
from cosmos.solvers.base import *
from ngsolve import *
import numpy as np

class DistanceSolver(BaseSolver):

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
        accepted_keys = ['zero_bnd', 'displacement']
        defaults = [CF(0.0), CF((0,)*self.mesh.dim)]  
        BaseSolver.__params_setup__(self.params, accepted_keys, defaults)

            
    def __build_MAF__(self):
        """_summary_
        """

        self.fes = H1(self.mesh, order= self.fes_order, dirichlet = self.mesh.Boundaries(self.params['zero_bnd']))

        self.fes_ho = H1(self.mesh, order= self.fes_order+1, dirichlet = self.mesh.Boundaries(self.params['zero_bnd']))
        fes_v = VectorH1(self.mesh, order = self.fes_order)

        u,v = self.fes.TnT()
        u_ho, v_ho = self.fes_ho.TnT()
        alpha = specialcf.mesh_size

        self.displ_h = GridFunction(fes_v)
        self.displ_h.Set(self.params['displacement'])

        self.mesh.SetDeformation(self.displ_h)

        # Defining the correct parameter
        aux = GridFunction(self.fes)
        aux.Set(alpha)
        self.delta = np.max(aux.vec.data)**2

        self.mesh.UnsetDeformation()

        self.gfu_ho = GridFunction(self.fes_ho)
        self.gfu_ho.Set(1, definedon = self.mesh.Boundaries(self.params['zero_bnd']))

        self.A1 = BilinearForm(self.fes_ho)
        self.A1 += (self.delta*grad(u_ho)*grad(v_ho) + u_ho*v_ho)*dx(deformation=self.displ_h)

        self.X = GridFunction(fes_v)

        self.A2 = BilinearForm(self.fes)
        self.A2 += (grad(u)*grad(v))*dx(deformation=self.displ_h)
        
        self.F2 = LinearForm(self.fes)
        self.F2 += self.X*grad(v)*dx(deformation=self.displ_h)
        self.gfu = GridFunction(self.fes)

        # self.gfu is the grid function used for computations and containing all the variables
        # u_h is a pointer to the function I want the boundary conditions to be applied to
        self.sol_h = [self.gfu, self.X]
    
    def __call__(self):
        """_summary_
        """

        # Updating the parameters
        self.__update_params__()

        # Constructing (without assembling) the bilinear and linear forms
        self.__build_MAF__()

        with TaskManager():

            self.A1.Assemble()
            r = - self.A1.mat * self.gfu_ho.vec
            self.gfu_ho.vec.data += self.A1.mat.Inverse(freedofs=self.fes_ho.FreeDofs()) * r

            self.X.Set(grad(self.gfu_ho)/Norm(grad(self.gfu_ho)))

            self.A2.Assemble()
            self.F2.Assemble()

            self.gfu.Set(0, definedon = self.params['zero_bnd'])
            r = self.F2.vec - self.A2.mat * self.gfu.vec

            self.gfu.vec.data += self.A2.mat.Inverse(freedofs=self.fes.FreeDofs()) * r

            self.gfu.vec.data = -self.gfu.vec.data

            # self.X.Set(-grad(self.gfu))

        yield
