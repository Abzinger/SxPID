"""
JxPID
"""
import numpy as np
from itertools import chain, combinations


def sgn(n):
    if n == 0:
        return np.array([+1])
    else:
        return np.concatenate((sgn(n-1), -sgn(n-1)), axis=None)
#^sgn()

def vec(n, diffs):
    """
    Args: 
         n : size of (b_1,...,b_n) the children of a
         diffs : vector of differences (d_i)_i where d_i = p(b_i) - p(a) and d_0 = p(a)  
    """
    if n == 0:
        return np.array([diffs[0]])
    else:
        temp = vec(n-1,diffs) + diffs[n]*np.ones(2**(n-1))
        return np.concatenate((vec(n-1, diffs), temp), axis=None)
#^ vec()

def differs(pdf, alpha, children):
    return 0
#^ differs()

def powerset(iter_set):
    lis = list(iter_set)
    return chain.from_iterable(combination(lis, r) for r in range(len(lis) + 1) )
#^ powerset()

def compare(alpha, beta):
    if beta < alpha:
        return beta
#^ compare()

def comparable(a,b):
    return 0
#^ comparable()
def children(alpha):
    return 0
#^ children()


def ijx_plus(alpha):
    chl = children(alpha) 
    lis_B = powerset(chl)
#^ ijxplus()

#--------
# Test!
#-------

diffs = np.array([0.2,0.3,0.4,0.5])

print(vec(0,diffs))
print(vec(1,diffs))
print(vec(2,diffs))
print(vec(3,diffs))
