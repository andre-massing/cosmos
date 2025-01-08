# %%
from ngsolve import *
from collections import Counter
from cosmos.utils.generate_surface_meshes import *

class BaseSolver():
    """_summary_
    """

    def __init__(self, mesh, verbose = 0):
        """_summary_

        Args:
            mesh (_type_): _description_
            verbose (int, optional): _description_. Defaults to 0.
        """

        # Solver characteristic mesh
        self.mesh = mesh
        # Level of info printed
        self.verbose = verbose

        if self.verbose > 0:
            self.__print_mesh_info__()


    def __print_mesh_info__(self):
        """_summary_
        """

        print(60*'-')

        print('The mesh is a ', str(self.mesh.dim) + 'D mesh')

        if self.mesh.ne == 0:
            print('! The mesh is an hypersurface embedded in ', str(self.mesh.dim) + 'D !')

        print('The number of:')
        print(5*' '+'vertices is: ', self.mesh.nv)
        print(5*' '+'edges is: ', self.mesh.nedge)
        print(5*' '+'faces is: ', self.mesh.nface)
        print(5*' '+'facets is: ', self.mesh.nfacet)
        print(5*' '+'elements is: ', self.mesh.ne, '(=0 if hypersurface)')

        self.bnd_mesh = [Counter(self.mesh.GetBoundaries()), 
                    Counter(self.mesh.GetBBoundaries()), 
                    Counter(self.mesh.GetBBBoundaries())]

        for i in range(self.mesh.dim):

            self.bnd_mesh[i]

            if self.mesh.ne == 0:

                if i == 0:
                    print('The surface regions are:')
                    for key, value in self.bnd_mesh[i].items():
                        print('  - There is(are) ', value, 'region(s) called ', key)
                else:
                    print('Boundaries of co-dimension ', i,':')
                    for key, value in self.bnd_mesh[i].items():
                        print('  - There is(are) ', value, 'region(s) called ', key)

            else:
                print('Boundaries of co-dimension ', i+1,':')
                for key, value in self.bnd_mesh[i].items():
                    print('  - There is(are) ', value, 'region(s) called ', key)

        print(60*'-', '\n')

    def __params_setup__(params={}, accepted_keys=[], defaults=[]):
        """_summary_

        Args:
            params (dict, optional): _description_. Defaults to {}.
            accepted_keys (list, optional): _description_. Defaults to [].
            defaults (list, optional): _description_. Defaults to [].

        Raises:
            Exception: _description_
        """
        
        for key in params.keys():

            if key not in accepted_keys:

                raise Exception("Dictionary key ", key, " is unknown as parameter!")
        
        for i, key in enumerate(accepted_keys):

            if key not in params.keys():

                params[key] = defaults[i]

    def __call__(self):
        """_summary_
        """
        
        print(65*'!')
        print("The steady solver base class is being called. No action performed")
        print(65*'!', '\n')

class UnsteadySolver(BaseSolver):

    def __init__(self, mesh, dt, t, T, verbose = 0):
        """_summary_

        Args:
            mesh (_type_): _description_
            dt (_type_): _description_
            t (_type_): _description_
            T (_type_): _description_
            verbose (int, optional): _description_. Defaults to 0.
        """

        super().__init__(mesh, verbose = verbose)

        self.dt = dt
        self.T = T
        self.t = t

    def __call__(self):
        """_summary_
        """

        print(70*'!')
        print("The unsteady solver base class is being called. No action performed")
        print(70*'!', '\n')
