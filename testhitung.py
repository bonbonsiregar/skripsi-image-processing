import numpy as np


a = np.array([1,0,1,0,1,
              1,0,1,1,1,
              1,0,0,0,1,
              1,0,1,1,0,
              1,0,1,0,1])


b = np.sum(a==1)
print(b)