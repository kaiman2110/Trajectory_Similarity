import numpy as np
from trajectory_similarity import similarity_calculation as simi


"""
A = np.array([[1, 1],
              [2, 2],
              [3, 3]
              ])

B = np.array([[1, 2],
              [3, 4],
              [5, 6]
              ])
"""

A = np.array([[0,0],
              [0,0]
              ])

B = np.array([[1,0],
              [2,1]
              ])
              
print('DTWの計算')
print(f'A: {A}\nB: {B}')
print(simi.dtw(A,B))