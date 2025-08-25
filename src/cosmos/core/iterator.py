import logging
logger = logging.getLogger(__name__)

import numpy as np
from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.config.parameters import get_config
from typing import Dict

class SolverIterator:

    def __init__(self, pdes:Dict[str, BasePDEModel]):

        self._pdes = pdes
        self._ordered_pdes = {}

    def _order_pdes(self):

        self._ordered_pdes = dict(sorted(self._pdes.items(), key=lambda x: x[1].model_order))

    def preprocess(self):

        self._order_pdes()
        for _, pde in self._ordered_pdes.items():
            pde.PreProcess()

    def solve_step(self):

        old_sol = []
        new_sol = []
        tol = 1e-6
        errors = np.zeros(len(self._ordered_pdes.keys()))

        for i, pde in enumerate(self._ordered_pdes.values()):
            pde.Solve()
            old_sol.append(pde.gfu.vec.Copy())

        # while np.max(errors)>tol and new_sol == []:
        #     for i, pde in enumerate(self._ordered_pdes.values()):
        #         pde.Solve()
        #         new_sol.append(pde.gfu.vec.Copy())
        #         errors[i] = Norm(new_sol[i]-old_sol[i])
                

    def postprocess(self):

        for _, pde in self._ordered_pdes.items():
            pde.PostProcess()