from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.pdes.ale import ale
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady
from cosmos.pdes.pde_tools import compute_error
from ngsolve.webgui import Draw
import numpy as np
import time

class BaseMC(BasePDE):

    def Solve(self, solverdata):

        self.PreProcess(solverdata)
        self.SystemSolve(solverdata)
        self.PostProcess(solverdata)

    def SystemSolve(self, solverdata):

        self.UpdateParams(solverdata)

        if isinstance(self.time_scheme, CN):
            raise Exception('Crack-Nicholson scheme not implemented for this solver')
        if isinstance(self.time_scheme, BDF2):
            raise Exception('BDF2 scheme not implemented for this solver')
        if isinstance(self.time_scheme, Steady):
            raise Exception('Steady solver not yet implemented for this solver')
        
        self.A.Assemble()
        self.F.Assemble()

        self.gfu.vec.data = self.invA*self.F.vec
