# Time-stepping state manager: wraps the simulation clock in NGSolve Parameters so
# time-dependent CoefficientFunctions built from them update without form re-creation.

import logging
logger = logging.getLogger(__name__)
from cosmos.config.parameters import get_config

import numbers
import numpy as np
from ngsolve import *

class SolverTime:
    """Owns the simulation clock (`t`) and time-step size (`dt`) as NGSolve
    Parameters, plus a short history buffer of previous values for multi-step
    time discretizations (e.g. BDF2).
    """

    def __init__(self, dt = Parameter(0.1), initial_t = 0, final_t = 1, t_coef = None):
        """
        Args:
            dt: Fixed time step (number), or a schedule (Parameter/list/ndarray)
                indexed by `self.iter` for variable/adaptive stepping.
            initial_t: Simulation start time.
            final_t: Simulation end time (must be >= initial_t).
            t_coef: Optional extra Parameter tracking a "look-ahead" time value,
                advanced/reset independently of `t` via advance_tcoef()/reset_tcoef().
        """

        self.dynamic = False
        self.iter = 0
        self.buffer = get_config().buffer

        if final_t<initial_t:
            raise Exception("Final time is smaller than initial time!")

        self.t = Parameter(initial_t)
        self.initial_t = initial_t
        self.final_t = final_t
        self.t_coef = t_coef

        # Normalize dt into an internal Parameter; a Parameter/list/ndarray schedule
        # marks the time-stepping as "dynamic" (dt can change between steps).
        if isinstance(dt, Parameter):
            self.dt = Parameter(dt.Get())
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
            raise Exception("dt must be either a number or a list of numbers")

        if self.dt.Get() <= 0:
            raise Exception("Time-step has been set to negative value")

        if isinstance(self.t_coef, Parameter):
            self.t_coef.Set(initial_t + self.dt.Get())
        elif t_coef != None:
            logger.warning("No valid t_coef provided, value of t_coef is ignored")
            self.t_coef = None

        # Keep the raw constructor arguments so postprocess() can re-derive dt at
        # every future iteration (list/array schedules are indexed by `self.iter`).
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
        """Hook called once per time step before the PDE models preprocess;
        currently a no-op, reserved for future use."""

        pass

    def postprocess(self):
        """Advance the clock by one step: increment `iter`, advance `t_coef` (if
        set), move `t` forward by `dt`, refresh `dt` from the configured schedule,
        and push the new (dt, t) pair onto the history buffers."""

        self.iter += 1
        if self.t_coef:
            self.advance_tcoef()
        self.t.Set(self.t.Get() + self.dt.Get())

        if isinstance(self.input_params["dt"], list):
            self.dt.Set( self.input_params["dt"][self.iter] )
        elif isinstance(self.input_params["dt"], np.ndarray):
            self.dt.Set(self.input_params["dt"][self.iter])
        elif isinstance(self.input_params["dt"], Parameter):
            self.dt.Set(self.input_params["dt"].Get())
        else:
            self.dt.Set(self.input_params["dt"])
        if self.dt.Get() <= 0:
                raise Exception("Time-step has been set to negative value")

        # Maintain history buffers of size `buffer` (oldest values dropped first).
        self.prev_dt.append(self.dt.Get())
        if len(self.prev_dt)>self.buffer:
            self.prev_dt.pop(0)
        self.prev_t.append(self.t.Get())
        if len(self.prev_t)>self.buffer:
            self.prev_t.pop(0)

    def get_state(self):
        """Return a dict snapshot of the current time-stepping state."""
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
        """Move `t_coef` to `t + dt`, i.e. the tentative next-step time; used by PDE
        models that assemble on a look-ahead time value inside Solve()."""
        if self.t_coef:
            self.t_coef.Set(self.t.Get() + self.dt.Get())

    def reset_tcoef(self):
        """Reset `t_coef` back to the current (committed) time `t`, undoing
        advance_tcoef()."""
        if self.t_coef:
            self.t_coef.Set(self.t.Get())

    def print_info(self):
        """Print a human-readable summary of the time-stepping parameters."""
        print('Time stepping parameters are: ')
        print(f'\t - initial time: {self.initial_t:.2e}')
        print(f'\t - final time: {self.final_t:.2e}')
        print(f'\t - timestep: {self.dt.Get():.2e}')