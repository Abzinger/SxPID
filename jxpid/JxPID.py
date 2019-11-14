"""
JxPID
"""
from itertools import chain, combinations


def powerset(iter_set):
    lis = list(iter_set)
    return chain.from_iterable(combination(lis, r) for r in range(len(lis) + 1) )
#^ powerset()

def children(alpha):
    ....
#^ children()


def ijx_plus(alpha):
    chl = children(alpha) 
    lis_B = powerset(chl)
    
