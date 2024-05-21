# %%
from ngsolve import *

class SteadySolver():

    def __init__(self, mesh=None, bnd_cond=None, verbose = 0):

        # Definition of some internal variables
        self.mesh = mesh
        # Type of simulation, used in the manufactured solution tool to 
        # understand if one has to iterate or not. It can probably be
        # imoproved, for example using yield for a steady solution, too
        self.type = 'steady'
        # Level of info printed
        self.verbose = verbose
        # Boundary conditions
        self.bnd_cond = bnd_cond

    def __setup__(self):

        if self.mesh != None:
            self.__setup_mesh__()

            if self.bnd_cond != None:
                self.__setup_bnd__()
            else:
                if self.verbose >0:
                    print('No boundary conditions given, homogeneous natural boundary conditions are applied')
        else:
            raise Exception("You have to provide a mesh in order to perform the simulation")

    def __setup_mesh__(self):

        # The geometry order is stored so that in case of manufactured solutions
        # I know how to curve the newly created/refined mesh
        self.geo_order = self.mesh.GetCurveOrder()

        # Checking if the mesh is a bulk mesh or a surface mesh and print some useful info
        if self.verbose == 1:
            if self.mesh.ne == 0:
                if self.mesh.dim == 3: 
                    print("The mesh is a surface in " + str(self.mesh.dim) + "D called " + self.mesh.GetBoundaries()[0])
                    if self.mesh.GetBBoundaries()[0] == '':
                        print("The surface has no boundary")
                    else:
                        print("The mesh boundaries are called ", self.mesh.GetBBoundaries())
                elif self.mesh.dim == 2:
                    print("The mesh is a curve in " + str(self.mesh.dim) + "D called " + self.mesh.GetBBoundaries()[0])
                    if self.mesh.GetBBBoundaries()[0] == '':
                        print("The curve has no boundary")
                    else:
                        print("The mesh boundaries are called ", self.mesh.GetBBBoundaries())
            else:
                print('The mesh is a bulk mesh in ' + str(self.mesh.dim) + 'D')
                print('The boundaries are called ', self.mesh.GetBoundaries())


    def __setup_bnd__(self):

        # Setting up labels and coefficient functions for the boundary conditions
        # Labels will be grouped in the string format  bnd_type="bnd1|bnd2|bnd3" so that they can be
        # easily used in the method Set(..., definedon = bnd_type)
        # Coefficient functions are constructed piecewise using a dictionary type_dict
        # For now only Dirichlet and Neumann b.c. are allowed
        dir_first_iteration = True
        neu_first_iteration = True
        dir_dict = {}
        neu_dict = {}
        self.dir_bnd = None
        self.neu_bnd = None
        for bnd_val in self.bnd_cond:
            bnd_type, bnd_name, bnd_funct = bnd_val
            if bnd_type == 'dir':
                if dir_first_iteration:
                    self.dir_bnd = bnd_name
                    dir_first_iteration = False
                else:
                    self.dir_bnd = self.dir_bnd + '|' + bnd_name
                dir_dict[bnd_name] = bnd_funct
            elif bnd_type == 'neu':
                if neu_first_iteration:
                    self.neu_bnd = bnd_name
                    neu_first_iteration = False
                else:
                    self.neu_bnd = self.neu_bnd + '|' + bnd_name
                neu_dict[bnd_name] = bnd_funct
            else:
                raise Exception("Only flags -dir- and -neu- are allowed, i.e. dirichlet or neumann boundary conditions")
        # Creating the boundary coefficient functions
        self.dir_cf = self.mesh.BoundaryCF(dir_dict, default=0)
        self.neu_cf = self.mesh.BoundaryCF(neu_dict, default=CF((0,) * self.mesh.dim))
            
        # Some message describing the boundary conditions
        if self.verbose == 1:
            if self.dir_bnd != None:
                print("Dirichlet boundary conditions are applied in ", self.dir_bnd)

            if self.neu_bnd != None:
                print("Neumann boundary conditions are applied in ", self.neu_bnd)


    def __call__(self):

        self.__setup__()
        print("The steady solver base class is being called. No action performed")

class UnsteadySolver(SteadySolver):

    def __init__(self, mesh, dt, t=Parameter(0.0), T=1.0, bnd_cond=None):

        super().__init__(mesh, bnd_cond)
        self.type = 'unsteady'

        self.dt = dt
        self.T = T
        self.t = t

    def __call__(self):

        self.__setup__()
        print("The unsteady solver base class is being called. No action performed")


# if __name__ == "__main__":

#     pass
# %%
