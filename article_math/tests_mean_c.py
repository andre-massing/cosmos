# %%

import sys
import os

# Add the 'tests_adr' folder to Python's search path
sys.path.append(os.path.join(os.path.dirname(__file__), './tests_mean_curv'))
from convergence_sphere_mean_c import test_sphere_mean_c


from cosmos.pdes.pde_mc_bgn import MCBGN
from cosmos.pdes.pde_mc_bgn_stab import MCBGNStab
from cosmos.pdes.pde_mc_walker import MCWalker
from cosmos.pdes.pde_mc_walker_dl import MCWalkerDL
from cosmos.solvers.time_schemes import BDF1

results_folder = "./results_mean_curv"
time_scheme = BDF1()
solvers = [
    (results_folder ,"MCBGN", MCBGN, {"time_scheme": time_scheme}),
    (results_folder ,"MCBGNStab", MCBGNStab, {"time_scheme": time_scheme}),
    (results_folder ,"MCWalker", MCWalker, {"time_scheme": time_scheme}),
    (results_folder ,"MCBGNPP", MCBGN, {"time_scheme": time_scheme, "postprocess": True}),
    (results_folder ,"MCBGNStabPP", MCBGNStab, {"time_scheme": time_scheme, "postprocess": True}),
    (results_folder ,"MCWalkerPP", MCWalker, {"time_scheme": time_scheme, "postprocess": True}),
    (results_folder ,"MCWalkerDL", MCWalkerDL, {"time_scheme": time_scheme}),
]

# Run all tests
for results, name, constructor, kwargs in solvers:
    test_sphere_mean_c(results, name, constructor, **kwargs)