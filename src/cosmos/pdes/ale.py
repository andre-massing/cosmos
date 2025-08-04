from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.pdes.pde_tools import compute_error
from cosmos.solvers.fields import Field
from ngsolve.webgui import Draw

class ale:

    def __init__(self, solverdata):

        self.solverdata = solverdata
        self.dim = self.solverdata.mesh.dim
        if self.solverdata.mesh.ne == 0:
            self.domain = self.solverdata.mesh.Boundaries('.*')
            V1 = VectorH1(self.solverdata.mesh, order=self.solverdata.mesh.GetCurveOrder(),
                    definedon=self.domain)
        else:
            self.domain = self.solverdata.mesh.Materials('.*')
            V1 = VectorH1(self.solverdata.mesh, order=self.solverdata.mesh.GetCurveOrder(),
                    definedon=self.domain)
        
        
        self.deformation = GridFunction(V1)
        self.deformation_new = GridFunction(V1)
        self.velocity = GridFunction(V1)
        self.material_velocity = GridFunction(V1)
        self._deformation_field = Field(None)
        self._velocity_field = Field(None)
        self._material_velocity_field = Field(None)

    def compute_new_def(self):

        self.solverdata.mesh.SetDeformation(self.deformation)
        if self.solverdata.mesh.ne == 0:
            if self._deformation_field():
                self.deformation_new.Set(self._deformation_field() + self.deformation, dual = True, definedon = self.domain)
            if self._velocity_field():
                self.velocity.Set(self._velocity_field(), dual = True, definedon = self.domain)
            if self._material_velocity_field():
                self.material_velocity.Set(self._material_velocity_field(), dual = True, definedon = self.domain)
        else:
            if self._deformation_field():
                volume_deformation = self.VectorPoisson(self.solverdata.mesh, self._deformation_field)
                self.deformation_new.Set(volume_deformation + self.deformation, dual = True, definedon = self.domain)
            if self._velocity_field():
                volume_velocity = self.VectorPoisson(self.solverdata.mesh, self._velocity_field)
                self.velocity.Set(volume_velocity, dual = True, definedon = self.domain)
            if self._material_velocity_field():
                volume_mat_vel = self.VectorPoisson(self.solverdata.mesh, self._material_velocity_field)
                self.material_velocity.Set(volume_mat_vel, dual = True, definedon = self.domain)
        self.solverdata.mesh.UnsetDeformation()

    def update_ale(self):

        self.deformation.vec.data = self.deformation_new.vec.data

    @property
    def deformation_field(self):
        return self._deformation_field

    @deformation_field.setter
    def deformation_field(self, new_value):
        if isinstance(new_value, GridFunction) or isinstance(new_value, CoefficientFunction):
            self._deformation_field.value = new_value
        elif callable(new_value):
            self._deformation_field.value = new_value
        else:
            raise TypeError(f"Unsupported type for Field: {type(new_value)}")
        
    @property
    def velocity_field(self):
        return self._velocity_field

    @velocity_field.setter
    def velocity_field(self, new_value):
        if isinstance(new_value, GridFunction) or isinstance(new_value, CoefficientFunction):
            self._velocity_field.value = new_value
        elif callable(new_value):
            self._velocity_field.value = new_value
        else:
            raise TypeError(f"Unsupported type for Field: {type(self._obj)}")
        
    @property
    def mat_velocity_field(self):
        return self._material_velocity_field

    @mat_velocity_field.setter
    def mat_velocity_field(self, new_value):
        if isinstance(new_value, GridFunction) or isinstance(new_value, CoefficientFunction):
            self._material_velocity_field.value = new_value
        elif callable(new_value):
            self._material_velocity_field.value = new_value
        else:
            raise TypeError(f"Unsupported type for Field: {type(self._obj)}")
        
    def VectorPoisson(self, mesh, dir):

        V = VectorH1(self.solverdata.mesh, order=self.solverdata.mesh.GetCurveOrder(),
                    definedon=self.domain, dirichlet = '.*')

        # Trial and test functions
        u = V.TrialFunction()
        v = V.TestFunction()

        # Bilinear and linear forms
        a = BilinearForm(V)
        a += InnerProduct(grad(u), grad(v))*dx

        u_sol = GridFunction(V)
        u_sol.Set(dir(), definedon = mesh.Boundaries('.*'))  # Set boundary condition

        # Solve the system
        a.Assemble()

        # Apply boundary conditions and solve
        res = - a.mat * u_sol.vec
        u_sol.vec.data += a.mat.Inverse(freedofs = V.FreeDofs()) * res

        return u_sol