from ngsolve import *

class Field:

    def __init__(self, pde = None, attribute = None, funct = None, cf = None,
                 name : str = 'default'):

        self._pde = pde
        self._attribute = attribute
        self._funct = funct
        self._cf = cf

        self.name = name

    def __call__(self):
        return self.evaluate()

    def evaluate(self):

        if self._pde and self._attribute:
            return getattr(self._pde, self._attribute)
        elif self._pde and not self._attribute:
            raise Exception('An attribute has to be associated with a pde and vice-versa')
        elif self._funct:
            return self._funct()
        elif self._cf != None:
            return self._cf
        else:
            return None
    
    def set_to_attribute(self, pde, attribute: str):
        self._pde = pde
        self._attribute = attribute
        self._funct = None
        self._cf = None

    def set_to_cf(self, cf):
        self._pde = None
        self._attribute = None
        self._funct = None
        self._cf = cf

    def set_to_function(self, funct):
        self._pde = None
        self._attribute = None
        self._funct = funct
        self._cf = None