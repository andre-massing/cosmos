import logging
logger = logging.getLogger(__name__)

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import InputField, OutputField
from cosmos.core.solver import Solver
from ngsolve.webgui import Draw

class DisplacementBoundaryBDF1Coupling(BasePDEModel):

    def __init__(self, solver:Solver, model_order:int, displacement_field, name:str = 'DisplacementBoundaryBDF1Model'):
        
        super().__init__()

        self.name = name
        self._solver = solver
        self.model_order = model_order

        solver._attach_model(self, self.model_order)
        self.domain = self._solver.mesh.domain
        self.gfu = GridFunction(self._solver.mesh.curr_deformation.space)

        self.input_fields["displacement"] = InputField(self.gfu, displacement_field, "displacement", self.domain)
        self.output_fields["displacement"] = OutputField(self.gfu, "displacement", BND)

    def PreProcess(self):
        
        pass

    def Solve(self):

        # self._solver.time.advance_tcoef()
        self.update_input_fields()
        self._solver.mesh.curr_deformation.vec.data = self._solver.mesh.prev_deformation[-1].vec.data + self.gfu.vec.data
        # self._solver.time.reset_tcoef()

    def PostProcess(self):
        
        self._solver.mesh.update_state()

    @property
    def displacement(self):
        return self.gfu