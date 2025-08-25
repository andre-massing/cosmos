import logging
logger = logging.getLogger(__name__)

import numpy as np
from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.pde.base import BasePDEModel
from cosmos.config.parameters import get_config
from typing import Dict

class SolverIterator:

    def __init__(self, pdes:Dict[str, BasePDEModel], iter:bool = False):

        self._pdes = pdes
        self._ordered_pdes = {}
        self.iter = iter

    def _order_pdes(self):

        self._ordered_pdes = dict(sorted(self._pdes.items(), key=lambda x: x[1].model_order))

    def preprocess(self):

        self._order_pdes()
        for _, pde in self._ordered_pdes.items():
            pde.PreProcess()

    def solve_step(self):

        old_sol = []
        tol = 1e-10
        errors = np.zeros(len(self._ordered_pdes.keys()))

        for i, pde in enumerate(self._ordered_pdes.values()):
            pde.Solve()
            old_sol.append(pde.gfu.vec.Copy())

        iter_count = 0
        if self.iter:
            while np.max(errors)>tol or iter_count == 0:
                for i, pde in enumerate(self._ordered_pdes.values()):
                    pde.Solve()
                    errors[i] = Norm(pde.gfu.vec-old_sol[i])
                    old_sol[i] = pde.gfu.vec.Copy()
                iter_count += 1
                logger.info(f'Step iter count: {iter_count} | Max error {np.max(np.array(errors)):.2e}')
                

    def postprocess(self):

        for _, pde in self._ordered_pdes.items():
            pde.PostProcess()