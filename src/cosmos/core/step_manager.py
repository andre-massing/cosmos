import logging
logger = logging.getLogger(__name__)

import os
import time
from ngsolve import *
from typing import Dict, TYPE_CHECKING
import traceback
import numpy as np
from ngsolve.webgui import Draw

if TYPE_CHECKING:
    from cosmos.core.model import CosmosModel

class CosmosStepManager:

    def __init__(self, kwargs):

        self.iter = None
        self.params = kwargs
        self.coupling_type = 'explicit'
        self.step_elasped_time = None

    def initialize(self, model: "CosmosModel"):

        if 'coupling_type' in  self.params.keys():
            if {self.params['coupling_type']} <= {'implicit', 'explicit'}:
                self.coupling_type = self.params['coupling_type']
            else:
                raise Exception('Coupling type must be either implicit or explicit')

        for pde in model.pdes:
            pde.Initialize() 

    def solve_step(self, model: "CosmosModel"):

        start = time.time()

        for pde in model.pdes:
            pde.PreProcess()

        old_sol = []
        tol = 1e-8

        errors = np.ones(len(model.pdes))*1e5

        for i, pde in enumerate(model.pdes):
            pde.Solve()
            old_sol.append(pde.gfu.vec.Copy())

        if self.coupling_type == 'implicit' and len(model.pdes)>0:
            print('ok')
            self.iter = 0
            while np.max(errors)>tol and self.iter < 20:
                for i, pde in enumerate(model.pdes):
                    pde.Solve()
                    errors[i] = Norm(pde.gfu.vec-old_sol[i])/np.max([len(pde.gfu.vec), Norm(old_sol[i])])
                    old_sol[i] = pde.gfu.vec.Copy()
                self.iter += 1
                print(errors)
                logger.debug(f'Step subiter_bool count: {self.iter} | Max error {np.max(np.array(errors)):.2e}')

            if self.iter == 20:
                raise Exception('Internal solver iteration exceeded max number of 20 iterations')
        
        model.ale.solve_ale(model)
        model.ale.finalize(model)
            
        logger.debug(f'Subiter solved successfully')

        for pde in model.pdes:
            pde.PostProcess()

        stop = time.time()
        self.step_elasped_time = stop-start