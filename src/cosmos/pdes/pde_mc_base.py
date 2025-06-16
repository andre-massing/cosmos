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

        t1 = time.time()
        self.PreProcess(solverdata)
        t2 = time.time()
        # print('Preprocess time: ', t2-t1)
        self.SystemSolve(solverdata)
        t3 = time.time()
        # print('System solve time: ', t3-t2)
        self.PostProcess(solverdata)
        t4 = time.time()
        # print('Postprocess time: ', t4-t3)

    def SystemSolve(self, solverdata):

        if isinstance(self.time_scheme, CN):
            raise Exception('Crack-Nicholson scheme not implemented for this solver')
        if isinstance(self.time_scheme, BDF2):
            raise Exception('BDF2 scheme not implemented for this solver')
        if isinstance(self.time_scheme, Steady):
            raise Exception('Steady solver not yet implemented for this solver')

        ale_curr = ale(solverdata)
        ale_curr.deformation.vec.data = solverdata.ale.deformation.vec.data
        ale_curr.velocity.vec.data = solverdata.ale.velocity.vec.data

        solverdata.ale.deformation.vec.data = solverdata.prev_def[-1].data
        solverdata.ale.velocity.vec.data = solverdata.prev_vel[-1].data

        t1 = time.time()
        self.A.Assemble()
        t2 = time.time()
        # print('Assembly of A: ', t2-t1)
        self.F.Assemble()
        t3 = time.time()
        # print('Assembly of F: ', t3-t2)
        self.invA.Update()
        t4 = time.time()
        # print('Update of invA: ', t4-t3)

        self.gfu.vec.data = self.invA*self.F.vec
        t5 = time.time()
        # print('Solution: ', t5-t4)

        solverdata.t.Set(solverdata.t.Get() - solverdata.dt.Get())
        solverdata.ale.deformation.vec.data = ale_curr.deformation.vec.data
        solverdata.ale.velocity.vec.data = ale_curr.velocity.vec.data