from dataclasses import dataclass
from ngsolve import *

@dataclass
class CFG():
    epsilon = 0.02
    sigma = 3/2/sqrt(2)
    output_folder = '/cluster/work/alesscon//bachini/'
    M = 0.001
    dt = 1e-5
    Tend = 1
    maxh = 0.05