import logging
logger = logging.getLogger(__name__)

import numpy as np
from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.pde.base import BasePDEModel
from typing import Dict
import time

class SolverIterator:

    def __init__(self, pdes:Dict[str, BasePDEModel], subiter_bool:bool = False):

        self._pdes = pdes
        self._ordered_pdes = {}
        self.subiter_bool = subiter_bool
        self.subiter_count = None

    def _order_pdes(self):

        self._ordered_pdes = dict(sorted(self._pdes.items(), key=lambda x: x[1].model_order))

    def preprocess(self):

        self._order_pdes()
        for _, pde in self._ordered_pdes.items():
            pde.PreProcess()

    def solve_step(self):

        old_sol = []
        tol = 1e-8
        errors = np.ones(len(self._ordered_pdes.keys()))*1e5

        for i, pde in enumerate(self._ordered_pdes.values()):
            pde.Solve()
            old_sol.append(pde.gfu.vec.Copy())

        if self.subiter_bool:
            self.subiter_count = 0
            while np.max(errors)>tol and self.subiter_count < 20:
                for i, pde in enumerate(self._ordered_pdes.values()):
                    pde.Solve()
                    errors[i] = Norm(pde.gfu.vec-old_sol[i])/len(pde.gfu.vec)
                    old_sol[i] = pde.gfu.vec.Copy()
                self.subiter_count += 1
                logger.debug(f'Step subiter_bool count: {self.subiter_count} | Max error {np.max(np.array(errors)):.2e}')

            if self.subiter_count == 20:
                raise Exception('Internal solver iteration exceeded max number of 20 iterations')
            
        logger.debug(f'Subiter solved successfully')
                

    def postprocess(self):

        for _, pde in self._ordered_pdes.items():
            pde.PostProcess()