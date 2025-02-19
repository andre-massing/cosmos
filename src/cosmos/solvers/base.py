from ngsolve import *
import types
import numbers

class Base:

    def params_check(self, params = {}, accepted_keys=[], defaults=[]):
        
        wrong = [key for key in params.keys() if key not in accepted_keys]

        if wrong:

            print("The following key are unknown parameters:\n", wrong)

            raise ValueError
        
        for i, key in enumerate(accepted_keys):

            params[key] = params.get(key, defaults[i])

    def params_update(self, params = {}):

        params_copy = params.copy()
        
        for key, value in params.items():

            if isinstance(value, types.FunctionType):

                params_copy[key] = value()

            if isinstance(value, numbers.Number):

                params_copy[key] = CF(value)

        return params_copy