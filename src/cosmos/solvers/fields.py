from ngsolve import *
import numbers

class Field:
    def __init__(self, obj=CF(0)):

        self._obj = obj
        if isinstance(self._obj, numbers.Number):
            self._obj = CF(self._obj)
        self.dim = self._eval().dim

    def _eval(self):
        """Evaluate and return the actual CoefficientFunction."""
        if isinstance(self._obj, GridFunction) or isinstance(self._obj, CoefficientFunction):
            return self._obj
        elif callable(self._obj):
            return CF(self._obj())
        else:
            raise TypeError(f"Unsupported type for Field: {type(self._obj)}")

    def __binary_op(self, other, op):
        if not isinstance(other, Field):
            other = Field(other)
        return Field(lambda: op(self._eval(), other._eval()))
    
    # Special access methods in case of Coefficient Functions
    def __getattr__(self, name): return getattr(self._eval(), name)
    def __getitem__(self, key): return self._eval()[key]
    
    # Arithmetic operations
    def __add__(self, other): return self.__binary_op(other, lambda a, b: a + b)
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other): return self.__binary_op(other, lambda a, b: a - b)
    def __rsub__(self, other): return Field(lambda: other._eval() - self._eval())
    def __mul__(self, other): return self.__binary_op(other, lambda a, b: a * b)
    def __rmul__(self, other): return self.__mul__(other)
    def __pow__(self, other): return self.__binary_op(other, lambda a, b: a ** b)
    def __rpow__(self, other): return self.__pow__(other)
    def __truediv__(self, other): return self.__binary_op(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return Field(lambda: other._eval() / self._eval())
    def __neg__(self): return Field(lambda: -1*self._eval())

    def __call__(self):
        """Evaluate and return the CoefficientFunction."""
        return self._eval()

    def __repr__(self):
        return f"Field({repr(self._obj)})"