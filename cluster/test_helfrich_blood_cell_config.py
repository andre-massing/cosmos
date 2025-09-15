from dataclasses import dataclass
from ngsolve import *

@dataclass
class CFG():
    output_folder = '/Users/alesscon/cluster/blood_cell/'
    dt = 1e-2
    maxh = 0.40
    Tend = 1.5