"""``Field``: the type behind every runtime-settable PDE parameter.

Every per-species/per-model parameter that a concrete ``cosmos.pde`` class
exposes through ``set_params`` (diffusion coefficients, boundary values,
right-hand sides, ALE velocities, ...) is stored as a ``Field``, not a raw
NGSolve object -- that is what lets ``BasePDEModel.set_params`` update a
parameter's *value* in place (``field.cf = new_value``) after a model has
already assembled its forms, and what lets a parameter be given as a plain
number, a ``CoefficientFunction``/``GridFunction``, or a zero-argument
callable (re-evaluated fresh on every access, e.g. ``lambda: model.ale.V``)
without every PDE model needing its own type-dispatch logic.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from typing import Callable, Union
from ngsolve import *
import numbers


class Field:
    """Uniform wrapper around an NGSolve coefficient or callable.

    Accepts a plain number, a ``CoefficientFunction``, a ``GridFunction``, or
    any zero-argument callable that returns a ``CoefficientFunction``, and
    exposes them through a consistent ``__call__`` / ``.cf`` interface so that
    PDE parameter fields can be changed transparently at run time.
    """

    def __init__(
        self,
        coef: Union[int, float, CoefficientFunction, GridFunction, Callable],
    ) -> None:
        # Tracked for introspection (e.g. by a caller wanting to know
        # whether re-evaluating `field()` can return something different
        # each time); nothing inside cosmos.core/cosmos.pde currently reads
        # this flag back.
        self.is_callable: bool = False

        if isinstance(coef, numbers.Number):
            self._coef = CF(coef)
        elif isinstance(coef, CoefficientFunction) or isinstance(coef, GridFunction):
            self._coef = coef
        elif callable(coef):
            self._coef = coef
            self.is_callable = True
        else:
            raise TypeError(
                "Coef must be one of these: CoefficientFunction (GridFunction) or callable"
            )

    def _eval(self) -> CoefficientFunction:
        """Evaluate and return CoefficientFunction."""
        if isinstance(self._coef, GridFunction) or isinstance(self._coef, CoefficientFunction):
            return self._coef
        elif callable(self._coef):
            return self._coef()
        else:
            raise TypeError(f"Unsupported type for Field: {type(self._coef)}")

    def __call__(self) -> CoefficientFunction:
        """Evaluate and return the CoefficientFunction."""
        return self._eval()

    def __repr__(self) -> str:
        return f"Field({repr(self._coef)})"

    @property
    def cf(self):
        return self._eval()

    @cf.setter
    def cf(self, new_coef):
        if isinstance(new_coef, GridFunction) or isinstance(new_coef, CoefficientFunction):
            self._coef = new_coef
        elif isinstance(new_coef, numbers.Number):
            self._coef = CF(new_coef)
        elif callable(new_coef):
            self._coef = new_coef
            self.is_callable = True
        else:
            raise TypeError("Unsupported type for new Field")
