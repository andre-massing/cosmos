# Drives the registered PDE models through their PreProcess/Solve/PostProcess
# lifecycle each time step, in ascending model_order, with optional fixed-point
# sub-iteration for implicitly-coupled multi-physics systems.

import logging
logger = logging.getLogger(__name__)

import numpy as np
from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.pde.base import BasePDEModel
from typing import Dict
import time

class SolverIterator:
    """Orchestrates one time step's worth of PreProcess/Solve/PostProcess calls
    across all PDE models attached to a Solver."""

    def __init__(self, pdes:Dict[str, BasePDEModel], subiter_bool:bool = False):
        """
        Args:
            pdes: Name-keyed registry of attached PDE models (shared reference into
                Solver.pdes, so newly attached models are picked up automatically).
            subiter_bool: If True, solve_step() repeats the solve of every model in a
                fixed-point loop until convergence, resolving implicit coupling
                between models within a single time step.
        """

        self._pdes = pdes
        self._ordered_pdes = {}
        self.subiter_bool = subiter_bool
        self.subiter_count = None

    def _order_pdes(self):
        """Sort the registered models by ascending `model_order`."""

        self._ordered_pdes = dict(sorted(self._pdes.items(), key=lambda x: x[1].model_order))

    def preprocess(self):
        """Re-order the models by model_order, then call PreProcess() on each in order."""

        self._order_pdes()
        for _, pde in self._ordered_pdes.items():
            pde.PreProcess()

    def solve_step(self):
        """Call Solve() once on every model, in model_order.

        If `subiter_bool` is set, repeats Solve() on every model in a fixed-point
        (Picard) loop, tracking the relative change of each model's solution vector,
        until all changes fall below `tol` or 20 sub-iterations are exhausted (in
        which case an exception is raised).
        """

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
        """Call PostProcess() on each model in model_order (e.g. ALEModel commits
        the new mesh deformation here)."""

        for _, pde in self._ordered_pdes.items():
            pde.PostProcess()