#!/bin/bash

# echo "Balàzs simulations"
cd mc_adr
# python3 mc_adr.py

# echo "Herant simulations"
# cd ../herant_et_al
# python3 herant_et_al.py

# echo "Lomakin simulations"
# cd ../lomakin_et_al
# python3 lomakin_et_al_2d.py 
# python3 lomakin_et_al.py 

echo "Bachini-Voigt simulations"
cd ../willmore_cahnhilliard
python3 willmore_cahnhilliard_it.py 
