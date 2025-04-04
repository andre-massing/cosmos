# %%

from typing import List
import numpy as np
# Write any import statements here

def getMaxAdditionalDinersCount(N: int, K: int, M: int, S: List[int]) -> int:
  # Write your code here
  if N>1e15 or N<1:
    raise Exception
    
  if K<1 or K>N:
    raise Exception
    
  if M>500000 or M<1:
    raise Exception
    
  if M > N:
    raise Exception
    
  S.sort()
  free = 0
  i = 1
  for cnt in S:
    if cnt<1 or cnt>N:
        raise Exception
    if i<cnt - K:
      free += (cnt - i)//(K+1)
    i = cnt + K +1

  if i<N:
    free += (N + K + 1 - i)//(K+1)
  
  return int(free)

# N = 10
# K = 1
# S = [2,6]
# M = 2
N = 15
K = 2
M = 3
S = [11, 6, 14]
getMaxAdditionalDinersCount(N, K, M, S)

#%%

from ngsolve import *

mesh = Mesh(unit_square.GenerateMesh(maxh = 0.2))
fes = VectorH1(mesh)
gfu = GridFunction(fes)

print(gfu.components)

fes2 = fes*fes
gfu2 = GridFunction(fes2)

print(type(gfu2.components[0].components))

print(gfu2[0])

# %%

def prova(x, y, *args):

  if args:
    print(args[0], args[1])

prova(1, 2, 3, 4)

#%%

import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.logspace(0.1, 2, 100)
y = np.logspace(0.1, 2, 100)

# Create a loglog plot
fig, ax = plt.subplots()
ax.loglog(x, y)

# Set custom ticks
tick_positions_x = [1, 10, 100]
tick_positions_y = [1, 10, 100]

# Set the custom tick labels
ax.set_xticks(tick_positions_x)
ax.set_yticks(tick_positions_y)

ax.set_xticklabels(['1', '10', '100'])
ax.set_yticklabels(['1', '10', '100'])

# Optionally, you can also disable the grid lines if they're not needed
ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# Remove the default ticks using tick_params (if necessary)
ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)

# Show the plot
plt.show()
