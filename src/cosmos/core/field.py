# Field abstraction used by PDE models to accept coefficients (scalars, NGSolve
# CoefficientFunctions/GridFunctions, or Python callables) uniformly, and to expose
# named input/output solution components.

import logging
logger = logging.getLogger(__name__)

from ngsolve import *
import numbers

from ngsolve.webgui import Draw

class Field:
    """Wraps a value that may be a number, a CoefficientFunction/GridFunction, or a
    zero-argument callable returning one of those, and evaluates it uniformly."""

    def __init__(self, coef):

        if isinstance(coef, numbers.Number):
            self._coef = CF(coef)
        elif isinstance(coef, CoefficientFunction) or isinstance(coef, GridFunction):
            self._coef = coef
        elif callable(coef):
            self._coef = coef
        else:
            raise Exception('Coef must be one of these: CoefficientFunction (GridFunction) or callable')

    def _eval(self):
        """Evaluate and return CoefficientFunction."""
        if isinstance(self._coef, GridFunction) or isinstance(self._coef, CoefficientFunction):
            return self._coef
        elif callable(self._coef):
            return self._coef()
        else:
            raise Exception(f"Unsupported type for Field: {type(self._coef)}")

    def __call__(self):
        """Evaluate and return the CoefficientFunction."""
        return self._eval()

    def __repr__(self):
        return f"Field({repr(self._coef)})"
        
class InputField(Field):
    """A named, model-owned input coefficient backed by a GridFunction.

    ``update()`` interpolates the current coefficient (``self()``) into ``gfu`` so that
    downstream bilinear/linear forms, which reference ``gfu`` directly, see the latest
    value. Used for diffusion coefficients, right-hand sides, boundary data, etc.
    """

    def __init__(self, gfu: GridFunction, default: CoefficientFunction, name:str, domain):

        if not isinstance(default, CoefficientFunction):
            raise Exception('InputField default must be a CoefficientFunction or a GridFunction')
        if not isinstance(gfu, GridFunction):
            raise Exception("InputField default must be a CoefficientFunction or a GridFunction")

        super().__init__(default)
        
        self.gfu = gfu
        self.dims = self().dims
        self.name = name
        self.domain = domain

        self.update()

    @property
    def cf(self):
        """The evaluated CoefficientFunction currently backing this input."""
        return self._eval()

    @cf.setter
    def cf(self, new_coef):
        """Replace the underlying coefficient (number, CoefficientFunction,
        GridFunction, or callable); does not itself refresh ``gfu`` (call
        ``update()`` for that)."""
        if isinstance(new_coef, GridFunction) or isinstance(new_coef, CoefficientFunction):
            self._coef = new_coef
        elif isinstance(new_coef, numbers.Number):
            self._coef = CF(new_coef)
        elif callable(new_coef):
            self._coef = new_coef
        else:
            raise Exception("Unsupported type for Field " +  self.name)

    def update(self):
        """Interpolate the current coefficient into the backing GridFunction."""
        self.gfu.Set(self(), definedon = self.domain, dual = True)

class OutputField(Field):
    """A named GridFunction exposed for output/visualization (e.g. VTK export via
    Solver.save_model_solution)."""

    def __init__(self, coef, name, domain):

        if not isinstance(coef, GridFunction):
            raise Exception("Output Field must be a GridFunction")

        super().__init__(coef)
        self.dims = self().dims
        self.name = name
        self.domain = domain
        self.sample_rate = 1

    @property
    def cf(self):
        """The evaluated CoefficientFunction (the underlying GridFunction)."""
        return self._eval()

    