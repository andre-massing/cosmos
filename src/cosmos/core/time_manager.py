import logging
logger = logging.getLogger(__name__)

import os
import time
from dataclasses import dataclass
from ngsolve import *
import numpy as np
import numbers

@dataclass
class CosmosTimeManager:
    
    def __init__(self, kwargs):

        self.params = kwargs

        self.iter = 0
        self.dt = Parameter(0.1)
        self.t = Parameter(0)

        self.prev_dt = []
        self.prev_t = []

    def initialize(self):
        
        self.helper = CosmosTimeHelper(**self.params)
        self.helper.initialize(self.t, self.dt)
        self.t1 = self.helper.t1
        self.t0 = self.helper.t0
        self.prev_dt.append(self.dt.Get())
        self.prev_t.append(self.t.Get())

    def next(self):

        self.iter += 1
        self.helper.next(self.iter, self.t, self.dt)
        self.prev_dt.append(self.dt.Get())
        if len(self.prev_dt)>6:
            self.prev_dt.pop(0)
        self.prev_t.append(self.t.Get())
        if len(self.prev_t)>6:
            self.prev_t.pop(0)

    def modify_dt(self, dt_new):
        self.dt.Set(dt_new)
        self.prev_dt[-1] = dt_new


class CosmosTimeHelper:

    def __init__(self, **kwargs):
        
        self.params = kwargs
        
        if {'t0', 't1', 'dt'} <= kwargs.keys():

            if isinstance(kwargs['t0'], numbers.Number):
                self.t0 = kwargs['t0']
            else:
                raise Exception("Initial time t0 must be a number")

            if isinstance(kwargs['t1'], numbers.Number):
                self.t1 = kwargs['t1']
            else:
                raise Exception("Final time t1 must be a number")

            if isinstance(kwargs['dt'], Parameter):
                self.dt = kwargs['dt']
            elif isinstance(kwargs['dt'], list):
                self.dt = Parameter(kwargs['dt'][0])
            elif isinstance(kwargs['dt'], np.ndarray):
                self.dt = Parameter(kwargs['dt'][0])
            elif isinstance(kwargs['dt'], numbers.Number):
                self.dt = Parameter(kwargs['dt'])
            else:
                raise Exception("dt must be either a number or a list of numbers")

            if self.dt.Get() <= 0:
                raise Exception("Time-step has been set to negative value")
            else:
                self.dt0 = self.dt.Get()
            
        else:
            raise Exception('The parameters t0, t1 and dt are needed for the model')
        
        if 't' in kwargs.keys():
            if isinstance(kwargs['t'], Parameter):
                self.t = kwargs['t']
                self.t.Set(self.t0)
            else:
                raise Exception('Variable t must be a Parameter')
        else:
            self.t = Parameter(self.t0)

    def initialize(self, t, dt):

        t.Set(self.t0)
        self.t.Set(self.t0)
        dt.Set(self.dt.Get())

    def next(self, iter, t, dt):

        t.Set(t.Get()+dt.Get())
        self.t.Set(t.Get())

        if isinstance(self.params["dt"], list):
            self.dt.Set(self.params["dt"][iter] )
        elif isinstance(self.params["dt"], np.ndarray):
            self.dt.Set(self.params["dt"][iter])
        elif isinstance(self.params["dt"], Parameter):
            self.dt.Set(self.params["dt"].Get())
        elif isinstance(self.params["dt"], numbers.Number):
            self.dt.Set(self.params["dt"])
        else:
            raise Exception("dt must be either a number or a list of numbers")
        
        if self.dt.Get() <= 0:
            raise Exception("Time-step has been set to negative value")
        
        dt.Set(self.dt.Get())

