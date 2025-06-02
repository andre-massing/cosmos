from ngsolve import *
import numbers

class Field:

    def __init__(self, obj):

        if isinstance(obj,Field):
            self._obj = obj._obj
        else:
            if isinstance(obj, numbers.Number):
                self._obj = CF(obj)
            else:
                self._obj = obj

    def _eval(self):
        """Evaluate and return the actual CoefficientFunction."""
        if isinstance(self._obj, GridFunction) or isinstance(self._obj, CoefficientFunction):
            return self._obj
        elif callable(self._obj):
            return self._obj()
        elif self._obj == None:
            return self._obj
        else:
            raise TypeError(f"Unsupported type for Field: {type(self._obj)}")

    def __call__(self):
        """Evaluate and return the CoefficientFunction."""
        return self._eval()

    def __repr__(self):
        return f"Field({repr(self._obj)})"
    
    @property
    def value(self):
        return self._eval()

    @value.setter
    def value(self, new_value):
        if isinstance(new_value, GridFunction) or isinstance(new_value, CoefficientFunction):
            self._obj = new_value
        elif callable(new_value):
            self._obj = new_value
        else:
            raise TypeError(f"Unsupported type for Field: {type(self._obj)}")