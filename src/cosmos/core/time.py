import logging
logger = logging.getLogger(__name__)
from cosmos.config.parameters import get_config

import numbers
import numpy as np
from ngsolve import *

class SolverTime:

    def __init__(self, dt = Parameter(0.1), initial_t = 0, final_t = 1, t_coef = None):

        self.dynamic = False
        self.iter = 0
        self.buffer = get_config().buffer

        if final_t<initial_t:
            raise Exception("Final time is smaller than initial time!")

        self.t = Parameter(initial_t)
        self.initial_t = initial_t
        self.final_t = final_t
        self.t_coef = t_coef

        if isinstance(dt, Parameter):
            self.dt = dt
            self.dynamic = True
        elif isinstance(dt, list):
            self.dt = Parameter(dt[self.iter])
            self.dynamic = True
        elif isinstance(dt, np.ndarray):
            self.dt = Parameter(dt[self.iter])
            self.dynamic = True
        elif isinstance(dt, numbers.Number):
            self.dt = Parameter(dt)
        else:
            raise Exception("dt must be either a number or a list of numbers", exc_info=True)

        if self.dt.Get() <= 0:
            raise Exception("Time-step has been set to negative value")
        
        if isinstance(self.t_coef, Parameter):
            self.t_coef.Set(initial_t + self.dt.Get())
        elif t_coef != None:
            logger.warning("No valid t_coef provided, value of t_coef is ignored")
            self.t_coef = None

        self.input_params = {
            "dt": dt,
            "initial_t": initial_t,
            "final_t": final_t,
            "t_coef": t_coef
        }

        self.prev_dt = [self.dt.Get()]
        self.prev_t = [self.t.Get()]

        logger.debug('Time Manager has been initialized correctly')

    def preprocess(self):

        pass

    def postprocess(self):

        self.iter += 1
        if self.t_coef != None:
            self.advance_tcoef()
        self.t.Set(self.t.Get() + self.dt.Get())

        if isinstance(self.input_params["dt"], list):
            self.dt.Set( self.input_params["dt"][self.iter] )
            if self.dt.Get() <= 0:
                raise Exception("Time-step has been set to negative value")
        elif isinstance(self.input_params["dt"], np.ndarray):
            self.dt.Set(self.input_params["dt"][self.iter])
            if self.dt.Get() <= 0:
                raise Exception("Time-step has been set to negative value")

        self.prev_dt.append(self.dt.Get())
        if len(self.prev_dt)>self.buffer:
            self.prev_dt.pop(0)
        self.prev_t.append(self.t.Get())
        if len(self.prev_t)>self.buffer:
            self.prev_t.pop(0)

        logger.debug('Time had been advanced correctly')

    def get_state(self):
        time_state = {
            "iter": self.iter,
            "time": self.t.Get(),
            "dt": self.dt.Get(),
            "initial_t": self.initial_t,
            "final_t": self.final_t,
            "prev_dt": self.prev_dt,
            "prev_t": self.prev_t
        }    
        return time_state
    
    def advance_tcoef(self):
        self.t_coef.Set(self.t.Get() + self.dt.Get())

    def reset_tcoef(self):
        self.t_coef.Set(self.t.Get())
    
    def print_state(self):
        print(self.get_state())