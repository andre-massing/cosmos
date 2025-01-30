from glapypack.solvers.base import Base
from ngsolve import *

def test_u_base():

    base = Base()

    params = {'dummy': 1}

    base.params_check(params, ['dummy', 'test'], [0, True])

    assert params['test'] and params['dummy'] == 1

def test_u_base_1():

    base = Base()

    def f():
        return True

    params = {'test': f}

    up_p = base.params_update(params)

    assert up_p['test']