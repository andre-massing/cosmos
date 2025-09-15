from dataclasses import dataclass
from ngsolve import *

@dataclass
class CFG():
    a = 0.1
    b = 0.9
    gamma = 100
    delta = 0.4
    output_folder = '/Users/alesscon/cluster/kovacs/'
    dt = 1e-3
    Tend = 3
    maxh = 0.07