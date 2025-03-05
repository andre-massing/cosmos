from ngsolve import *

class Base:

    def params_check(self, params = {}, accepted_keys=[], defaults=[]):
        
        wrong = [key for key in params.keys() if key not in accepted_keys]

        if wrong:

            print("The following key are unknown parameters:\n", wrong)

            raise ValueError
        
        for i, key in enumerate(accepted_keys):

            params[key] = params.get(key, defaults[i])