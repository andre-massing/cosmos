# %%
from ngsolve import *

# # Only needed for some testing
# import sys
# sys.path.insert(0, '../geometries/')
# from generate_meshes import *

class SteadySolver():

    def __init__(self, mesh, bnd_cond=[["dir", "default", CF(0.0)]]):
        self.mesh = mesh
        self.geo_order = self.mesh.GetCurveOrder()

        ## Checking if the mesh is a bulk mesh or a surface mesh
        if self.mesh.GetMaterials()[0] == '':
            if self.mesh.dim == 3: 
                print("The mesh is a surface in " + str(self.mesh.dim) + "D called " + self.mesh.GetBoundaries()[0])
            elif mesh.dim == 2:
                print("The mesh is a curve in " + str(self.mesh.dim) + "D called " + self.mesh.GetBoundaries()[0])

            if mesh.GetBBoundaries()[0] == '':
                print("The surface/curve has no boundary")
            else:
                print("The mesh boundaries are called ", self.mesh.GetBBoundaries())
        else:
            print('The mesh is a bulk mesh in ' + str(self.mesh.dim) + 'D')
            print('The boundaries are called ', self.mesh.GetBoundaries())

        # Setting up labels and coefficient functions for the boundary conditions
        # Labels will be grouped in the string format  bnd_type="bnd1|bnd2|bnd3" so that they can be
        # easily used in the method Set(..., definedon = bnd_type)
        # Coefficient functions are constructed piecewise using a dictionary type_dict
        # For now only Dirichlet and Neumann b.c. are allowed
        i = 0
        j = 0
        dir_dict = {}
        neu_dict = {}
        for bnd_val in bnd_cond:
            bnd_type, bnd_name, bnd_funct = bnd_val
            if bnd_type == 'dir':
                if i==0:
                    self.dir_bnd = bnd_name
                    i = 1
                else:
                    self.dir_bnd = self.dir_bnd + '|' + bnd_name
                dir_dict[bnd_name] = bnd_funct
            elif bnd_type == 'neu':
                if j==0:
                    self.neu_bnd = bnd_name
                    j = 1
                else:
                    self.neu_bnd = self.neu_bnd + '|' + bnd_name
                neu_dict[bnd_name] = bnd_funct

        # Creating the boundary coefficient functions
        self.dir_cf = self.mesh.BoundaryCF(dir_dict, default=0)
        self.neu_cf = self.mesh.BoundaryCF(neu_dict, default=0)

        # Some message describing the boundary conditions
        if self.dir_bnd != '':
            print("Dirichlet boundary conditions are applied in ", self.dir_bnd)
        if self.neu_bnd != '':
            print("Neumann boundary conditions are applied in ", self.neu_bnd)

    def __call__(self):
        print("")

class UnsteadySolver(SteadySolver):

    def __init__(self, mesh, dt, t=Parameter(0.0), T=1.0, bnd_cond=[["dir", "default", CF(0.0)]]):

        super().__init__(mesh, bnd_cond)

        self.dt = dt
        self.T = T
        self.t = t

    def __call__(self):
        
        pass

    def update(self):
        
        pass


if __name__ == "__main__":

    pass
# %%
