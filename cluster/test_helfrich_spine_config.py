from dataclasses import dataclass
from ngsolve import *

@dataclass
class CFG():
    output_folder = '/Users/alesscon/cluster/spine/'
    dt = 1e-6
    Tend = 5e-2