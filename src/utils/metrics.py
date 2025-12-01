import numpy as np

def mae(a,b): return np.mean(np.abs(a-b))
def smape(a,b): 
    denom = (np.abs(a)+np.abs(b))/2 + 1e-8
    return np.mean(np.abs(a-b)/denom)

