from dataclasses import dataclass
from ngsolve import *

@dataclass
class CFG():
    output_folder = '/cluster/work/alesscon/blood_cell/'
    dt = 1e-4
    maxh = 0.15
    Tend = 1