# %%

import pandas as pd
import numpy as np
import os

# List of input filenames
input_files = ['./article_math/torus/notang_nostab/notang_nostab.dat',
                './article_math/torus/notang_stab/notang_stab.dat',
                './article_math/torus/tang_nostab/tang_nostab.dat',
                './article_math/torus/tang_stab/tang_stab.dat']

# Placeholder for data
columns = []
header = 'Time'
# Loop through each file and extract columns
for i, file in enumerate(input_files):
    df = pd.read_csv(file, delimiter='\t')
    header += '\t' + df.columns[0].replace('_', '-')
    data = np.array(df.values[:])
    columns.append(data)  # Take second column (index 1) from each

n = 100
Tend = 1.4
original_length =  len(columns[0])
energy = np.pi**2*4

columns.append(np.ones(original_length)*energy)
header += '\tExact'

indices = np.linspace(0, original_length - 1, n, dtype=int)
for i, file in enumerate(input_files):
    columns[i] = columns[i][indices]
columns[-1] = columns[-1][indices]
time = indices/original_length*Tend
# Stack all data horizontally: time + 4 selected columns
output_data = np.column_stack([time] + columns)

# Save to a new .dat file
print(header)
np.savetxt('./article_math/torus/torus.dat', output_data, fmt='%.6f', delimiter='\t', header=header, comments='')

# %%

import pandas as pd
import numpy as np
import os

# List of input filenames
input_files = ['./article_math/torus/notang_nostab_2/notang_nostab2.dat',
                './article_math/torus/notang_stab_2/notang_stab2.dat',
                './article_math/torus/tang_nostab_2/tang_nostab2.dat',
                './article_math/torus/tang_stab_2/tang_stab2.dat']

# Placeholder for data
columns = []
header = 'Time'
# Loop through each file and extract columns
for i, file in enumerate(input_files):
    df = pd.read_csv(file, delimiter='\t')
    header += '\t' + df.columns[0].replace('_', '-')
    data = np.array(df.values[:])
    columns.append(data)  # Take second column (index 1) from each

n = 100
Tend = 1.4
original_length =  len(columns[0])
energy = np.pi**2*4

columns.append(np.ones(original_length)*energy)
header += '\tExact'

indices = np.linspace(0, original_length - 1, n, dtype=int)
for i, file in enumerate(input_files):
    columns[i] = columns[i][indices]
columns[-1] = columns[-1][indices]
time = indices/original_length*Tend
# Stack all data horizontally: time + 4 selected columns
output_data = np.column_stack([time] + columns)

# Save to a new .dat file
print(header)
np.savetxt('./article_math/torus/torus2.dat', output_data, fmt='%.6f', delimiter='\t', header=header, comments='')
