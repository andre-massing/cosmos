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