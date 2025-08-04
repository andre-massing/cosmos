# %%

import sys
import os

# Add the 'tests_adr' folder to Python's search path
sys.path.append(os.path.join(os.path.dirname(__file__), './tests_adr'))
from convergence_sphere import test_sphere
from convergence_half_sphere_dir import test_half_sphere_dir
from convergence_half_sphere_neu import test_half_sphere_neu
from illposed_case import test_illposed

from cosmos.pdes.pde_adr_bnd import BndADR
from cosmos.pdes.pde_adr_bnd_ad import AdBndADR
from cosmos.solvers.time_schemes import BDF1, BDF2, CN

results_folder = "./results_adr"
time_scheme = BDF2()
solvers = [
    # (results_folder ,"BndADR", BndADR, {"time_scheme": time_scheme}),
    # (results_folder ,"AdBndADR", AdBndADR, {"time_scheme": time_scheme}),
    # (results_folder ,"BndADR_BP", BndADR, { "time_scheme": time_scheme, "BP": [-1, 1] }),
    # (results_folder ,"AdBndADR_BP", AdBndADR, { "time_scheme": time_scheme, "BP": [-1, 1] }),
    (results_folder ,"BndADR_MP", BndADR, { "time_scheme": time_scheme, "MP": True }),
    (results_folder ,"AdBndADR_MP", AdBndADR, { "time_scheme": time_scheme, "MP": True }),
    (results_folder ,"BndADR_BPMP", BndADR, { "time_scheme": time_scheme, "BP": [-1, 1], "MP": True }),
    (results_folder ,"AdBndADR_BPMP", AdBndADR, { "time_scheme": time_scheme, "BP": [-1, 1], "MP": True })
]

# Run all tests
for results, name, constructor, kwargs in solvers:
    test_sphere(results, name, constructor, **kwargs)
    test_half_sphere_dir(results, name, constructor, **kwargs)
    test_half_sphere_neu(results, name, constructor, **kwargs)
    test_illposed(results, name, constructor, **kwargs)