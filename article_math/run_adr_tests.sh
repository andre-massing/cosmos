#!/bin/bash

# echo "Simulations for non-stabilized surface ADR"
cd adr_bnd
# python3 adr_bnd.py

# echo "Simulations for stabilized surface ADR"
# cd ../adr_bnd_stab
# python3 adr_bnd_stab.py

echo "Simulations for mass-preserving non-stabilized surface ADR"
cd ../adr_bnd_mp
python3 adr_bnd_mp.py

echo "Simulations for bound-preserving non-stabilized surface ADR"
cd ../adr_bnd_bp
python3 adr_bnd_bp.py

echo "Simulations for bound- and mass-preserving non-stabilized surface ADR"
cd ../adr_bnd_bp_mp
python3 adr_bnd_bp_mp.py

echo "Simulations for mass-preserving stabilized surface ADR"
cd ../adr_bnd_stab_mp
python3 adr_bnd_stab_mp.py

echo "Simulations for bound-preserving stabilized surface ADR"
cd ../adr_bnd_stab_bp
python3 adr_bnd_stab_bp.py

echo "Simulations for bound- and mass-preserving stabilized surface ADR"
cd ../adr_bnd_stab_bp_mp
python3 adr_bnd_stab_bp_mp.py