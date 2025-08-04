#!/bin/bash

# echo "Simulations for non-stabilized Willmore without tangential redistribution"
cd willmore_bgn_notang_nostab
# python3 willmore_bgn_notang_nostab.py

# echo "Simulations for non-stabilized Willmore with tangential redistribution"
# cd ../willmore_bgn_notang_stab
# python3 willmore_bgn_notang_stab.py

echo "Simulations for stabilized Willmore without tangential redistribution"
cd ../willmore_bgn_tang_nostab
python3 willmore_bgn_tang_nostab.py

echo "Simulations for stabilized Willmore with tangential redistribution"
cd ../willmore_bgn_tang_stab
python3 willmore_bgn_tang_stab.py
