# test_gates.py
"""Provide unit tests for SxPID on logic gates."""
import time
import numpy as np
from math import log2
from sxpid import SxPID
from sxpid import lattices as lt 

#--------
# Test!
#-------

# Format of the pdf is 
# dict( (s1,s2,t) : p(s1,s2,t) ) for all s1 in S1, s2 in S2, and t in T if p(s1,s2,t) > 0.


# Read lattices from a file
# Pickled as { n -> [{alpha -> children}, (alpha_1,...) ] }
lattices = lt.lattices

def validate(n, gate, true_values, lattices):
    ptw, avg = SxPID.pid(n, gate, lattices[n][0], lattices[n][1], False)
    for rlz in ptw.keys():
        assert np.allclose(true_values[rlz],
                           np.array([ptw[rlz][((1,),(2,),)][2],
                                     ptw[rlz][((1,),)][2],
                                     ptw[rlz][((2,),)][2],
                                     ptw[rlz][((1,2),)][2]])), (
                                         'pointwise values at ({0},{1},{2}) are not [{3:.8f}, {4:.8f}, {5:.8f}, {6:.8f}]'.format(
                                             rlz[0], rlz[1], rlz[2],
                                             true_values[rlz][0],
                                             true_values[rlz][1],
                                             true_values[rlz][2],
                                             true_values[rlz][3],)
                                     )
    #^ for
#^ validate()


#-----------------
# Bivairate Gates
#-----------------

# Xor
def test_xor_gate():
    """Test SxPID on Xor gate"""
    xorgate = dict()
    xorgate[(0,0,0)] = 0.25
    xorgate[(0,1,1)] = 0.25
    xorgate[(1,0,1)] = 0.25
    xorgate[(1,1,0)] = 0.25

    true_values = dict()
    true_values[(0,0,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(0,1,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(1,0,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(1,1,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
    validate(2, xorgate, true_values, lattices)
#^ test_xor_gate()

# And
def test_and_gate():
    """Test SxPID on And gate"""
    andgate = dict()
    andgate[(0,0,0)] = 0.25
    andgate[(0,1,0)] = 0.25
    andgate[(1,0,0)] = 0.25
    andgate[(1,1,1)] = 0.25

    true_values = dict()
    true_values[(0,0,0)] = np.array([log2(4/3), 0., 0., 0.])
    true_values[(0,1,0)] = np.array([log2(8/9), log2(3/2), log2(3/4), log2(4/3)])
    true_values[(1,0,0)] = np.array([log2(8/9), log2(3/4), log2(3/2), log2(4/3)])
    true_values[(1,1,1)] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    validate(2, andgate, true_values, lattices)
#^ test_and_gate()

# Unq
def test_unq_gate():
    """Test SxPID on Unq gate"""
    unqgate = dict()
    unqgate[(0,0,0)] = 0.25
    unqgate[(0,1,0)] = 0.25
    unqgate[(1,0,1)] = 0.25
    unqgate[(1,1,1)] = 0.25

    true_values = dict()
    true_values[(0,0,0)] = np.array([log2(4/3), log2(3/2), log2(3/4), log2(4/3)])
    true_values[(0,1,0)] = np.array([log2(4/3), log2(3/2), log2(3/4), log2(4/3)])
    true_values[(1,0,1)] = np.array([log2(4/3), log2(3/2), log2(3/4), log2(4/3)])
    true_values[(1,1,1)] = np.array([log2(4/3), log2(3/2), log2(3/4), log2(4/3)])
    validate(2, unqgate, true_values, lattices)
#^ test_unq_gate()

# Rnd
def test_rnd_gate():
    """Test SxPID on Rnd gate"""
    rndgate = dict()
    rndgate[(0,0,0)] = 0.5
    rndgate[(1,1,1)] = 0.5

    true_values = dict()
    true_values[(0,0,0)] = np.array([log2(2), log2(1), log2(1), log2(1)])
    true_values[(1,1,1)] = np.array([log2(2), log2(1), log2(1), log2(1)])
    validate(2, rndgate, true_values, lattices)
#^ test_rnd_gate()

# Copy
def test_copy_gate():
    """Test SxPID on Copy gate"""
    copygate = dict()
    copygate[(0,0,(0,0))] = 0.25
    copygate[(0,1,(0,1))] = 0.25
    copygate[(1,0,(1,0))] = 0.25
    copygate[(1,1,(1,1))] = 0.25

    true_values = dict()
    true_values[(0,0,(0,0))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(0,1,(0,1))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(1,0,(1,0))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(1,1,(1,1))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    validate(2, copygate, true_values, lattices)
#^ test_copy_gate()

# (S1, Xor)
def test_s1xor_gate():
    """Test SxPID on (S1, Xor) gate"""
    s1xorgate = dict()
    s1xorgate[(0,0,(0,0))] = 0.25
    s1xorgate[(0,1,(0,1))] = 0.25
    s1xorgate[(1,0,(1,1))] = 0.25
    s1xorgate[(1,1,(1,0))] = 0.25

    true_values = dict()
    true_values[(0,0,(0,0))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(0,1,(0,1))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(1,0,(1,1))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(1,1,(1,0))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    validate(2, s1xorgate, true_values, lattices)
#^ test_s1xor_gate()

# (S2, Xor)
def test_s2xor_gate():
    """Test SxPID on (S2, Xor) gate"""
    s2xorgate = dict()
    s2xorgate[(0,0,(0,0))] = 0.25
    s2xorgate[(0,1,(1,1))] = 0.25
    s2xorgate[(1,0,(0,1))] = 0.25
    s2xorgate[(1,1,(1,0))] = 0.25

    true_values = dict()
    true_values[(0,0,(0,0))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(0,1,(1,1))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(1,0,(0,1))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    true_values[(1,1,(1,0))] = np.array([log2(4/3), log2(3/2), log2(3/2), log2(4/3)])
    validate(2, s2xorgate, true_values, lattices)
#^ test_s2xor_gate()

# PwUnq
def test_pwunq_gate():
    """Test SxPID on PwUnq gate"""
    pwunqgate = dict()
    pwunqgate[(0,1,1)] = 0.25
    pwunqgate[(1,0,1)] = 0.25
    pwunqgate[(0,2,2)] = 0.25
    pwunqgate[(2,0,2)] = 0.25

    true_values = dict()
    true_values[(0,1,1)] = np.array([log2(1), log2(1), log2(2), log2(1)])
    true_values[(1,0,1)] = np.array([log2(1), log2(2), log2(1), log2(1)])
    true_values[(0,2,2)] = np.array([log2(1), log2(1), log2(2), log2(1)])
    true_values[(2,0,2)] = np.array([log2(1), log2(2), log2(1), log2(1)])
    validate(2, pwunqgate, true_values, lattices)
#^ test_pwunq_gate()

# RndErr
def test_rnderr_gate():
    """Test SxPID on RndErr gate"""
    rnderrgate = dict()
    rnderrgate[(0,0,0)] = 3/8
    rnderrgate[(1,1,1)] = 3/8
    rnderrgate[(0,1,0)] = 1/8
    rnderrgate[(1,0,1)] = 1/8

    true_values = dict()
    true_values[(0,0,0)] = np.array([log2(8/5), log2(5/4), log2(15/16), log2(16/15)])
    true_values[(1,1,1)] = np.array([log2(8/5), log2(5/4), log2(15/16), log2(16/15)])
    true_values[(0,1,0)] = np.array([log2(8/7), log2(7/4), log2(7/16), log2(16/7)])
    true_values[(1,0,1)] = np.array([log2(8/7), log2(7/4), log2(7/16), log2(16/7)])
    validate(2, rnderrgate, true_values, lattices)
#^ test_rnderr_gate()

# # RndUnqXor
# def test_rndunqxor_gate():
#     """Test SxPID on RndUnqXor gate"""
#     rndunqxorgate = dict()
#     rndunqxorgate[(('r','a',0),('r','b',0),('r','a','b',0))] = 1/32
#     rndunqxorgate[(('r','a',0),('r','b',1),('r','a','b',1))] = 1/32
#     rndunqxorgate[(('r','a',1),('r','b',0),('r','a','b',1))] = 1/32
#     rndunqxorgate[(('r','a',1),('r','b',1),('r','a','b',0))] = 1/32
    
#     rndunqxorgate[(('r','a',0),('r','B',0),('r','a','B',0))] = 1/32
#     rndunqxorgate[(('r','a',0),('r','B',1),('r','a','B',1))] = 1/32
#     rndunqxorgate[(('r','a',1),('r','B',0),('r','a','B',1))] = 1/32
#     rndunqxorgate[(('r','a',1),('r','B',1),('r','a','B',0))] = 1/32
    
#     rndunqxorgate[(('r','A',0),('r','b',0),('r','A','b',0))] = 1/32
#     rndunqxorgate[(('r','A',0),('r','b',1),('r','A','b',1))] = 1/32
#     rndunqxorgate[(('r','A',1),('r','b',0),('r','A','b',1))] = 1/32
#     rndunqxorgate[(('r','A',1),('r','b',1),('r','A','b',0))] = 1/32
    
#     rndunqxorgate[(('r','A',0),('r','B',0),('r','A','B',0))] = 1/32
#     rndunqxorgate[(('r','A',0),('r','B',1),('r','A','B',1))] = 1/32
#     rndunqxorgate[(('r','A',1),('r','B',0),('r','A','B',1))] = 1/32
#     rndunqxorgate[(('r','A',1),('r','B',1),('r','A','B',0))] = 1/32

#     rndunqxorgate[(('R','a',0),('R','b',0),('R','a','b',0))] = 1/32
#     rndunqxorgate[(('R','a',0),('R','b',1),('R','a','b',1))] = 1/32
#     rndunqxorgate[(('R','a',1),('R','b',0),('R','a','b',1))] = 1/32
#     rndunqxorgate[(('R','a',1),('R','b',1),('R','a','b',0))] = 1/32

#     rndunqxorgate[(('R','a',0),('R','B',0),('R','a','B',0))] = 1/32
#     rndunqxorgate[(('R','a',0),('R','B',1),('R','a','B',1))] = 1/32
#     rndunqxorgate[(('R','a',1),('R','B',0),('R','a','B',1))] = 1/32
#     rndunqxorgate[(('R','a',1),('R','B',1),('R','a','B',0))] = 1/32

#     rndunqxorgate[(('R','A',0),('R','b',0),('R','A','b',0))] = 1/32
#     rndunqxorgate[(('R','A',0),('R','b',1),('R','A','b',1))] = 1/32
#     rndunqxorgate[(('R','A',1),('R','b',0),('R','A','b',1))] = 1/32
#     rndunqxorgate[(('R','A',1),('R','b',1),('R','A','b',0))] = 1/32
    
#     rndunqxorgate[(('R','A',0),('R','B',0),('R','A','B',0))] = 1/32
#     rndunqxorgate[(('R','A',0),('R','B',1),('R','A','B',1))] = 1/32
#     rndunqxorgate[(('R','A',1),('R','B',0),('R','A','B',1))] = 1/32
#     rndunqxorgate[(('R','A',1),('R','B',1),('R','A','B',0))] = 1/32


#     true_values = dict()
#     true_values[(0,0,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(0,1,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,0,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,1,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     validate(2, rndunqxorgate, true_values, lattices)
# #^ test_rndunqxor_gate()

# # Sum (equivalent to XorAnd)
# def test_sum_gate():
#     """Test SxPID on Sum gate"""
#     sumgate = dict()
#     sumgate[(0,0,0)] = 1/4
#     sumgate[(0,1,1)] = 1/4
#     sumgate[(1,0,1)] = 1/4
#     sumgate[(1,1,2)] = 1/4


#     true_values = dict()
#     true_values[(0,0,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(0,1,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,0,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,1,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     validate(2, sumgate, true_values, lattices)
# #^ test_sum_gate()

# # Reduced or
# def test_reduor_gate():
#     """Test SxPID on ReducedOr gate"""
#     reduorgate = dict()
#     reduorgate[(0,0,0)] = 1/2
#     reduorgate[(0,1,1)] = 1/4
#     reduorgate[(1,0,1)] = 1/4


#     true_values = dict()
#     true_values[(0,0,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(0,1,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,0,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,1,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     validate(2, reduorgate, true_values, lattices)
# #^ test_reduor_gate()

# # RndXor
# def test_rndxor_gate():
#     """Test SxPID on RndXor gate"""
#     rndxorgate = dict()
#     rndxorgate[(('r',0),('r',0),('r',0))] = 1/8
#     rndxorgate[(('r',0),('r',1),('r',1))] = 1/8
#     rndxorgate[(('r',1),('r',0),('r',1))] = 1/8
#     rndxorgate[(('r',1),('r',1),('r',0))] = 1/8

#     rndxorgate[(('R',0),('R',0),('R',0))] = 1/8
#     rndxorgate[(('R',0),('R',1),('R',1))] = 1/8
#     rndxorgate[(('R',1),('R',0),('R',1))] = 1/8
#     rndxorgate[(('R',1),('R',1),('R',0))] = 1/8


#     true_values = dict()
#     true_values[(0,0,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(0,1,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,0,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,1,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     validate(2, rndxorgate, true_values, lattices)
# #^ test_rndxor_gate()

# # Example A (Williams and Beer)
# def test_wbA_gate():
#     """Test SxPID on wbA gate"""
#     wbAgate = dict()
#     wbAgate[(0,0,0)] = 1/3
#     wbAgate[(0,1,1)] = 1/3
#     wbAgate[(1,0,2)] = 1/3


#     true_values = dict()
#     true_values[(0,0,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(0,1,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,0,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,1,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     validate(2, wbAgate, true_values, lattices)
# #^ test_wbA_gate()

# # Example B (Williams and Beer)
# def test_wbB_gate():
#     """Test SxPID on wbB gate"""
#     wbBgate = dict()
#     wbBgate[(0,0,0)] = 1/4
#     wbBgate[(0,1,1)] = 1/4
#     wbBgate[(1,0,2)] = 1/4
#     wbBgate[(1,1,1)] = 1/4


#     true_values = dict()
#     true_values[(0,0,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(0,1,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,0,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,1,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     validate(2, wbBgate, true_values, lattices)
# #^ test_wbB_gate()


# # Example C (Williams and Beer)
# def test_wbC_gate():
#     """Test SxPID on wbC gate"""
#     wbCgate = dict()
#     wbCgate[(0,0,0)] = 1/6
#     wbCgate[(0,1,1)] = 1/6
#     wbCgate[(0,1,2)] = 1/6
#     wbCgate[(1,0,2)] = 1/6
#     wbCgate[(1,1,0)] = 1/6
#     wbCgate[(1,1,1)] = 1/6


#     true_values = dict()
#     true_values[(0,0,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(0,1,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,0,1)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     true_values[(1,1,0)] = np.array([log2(2/3), log2(3/2), log2(3/2), log2(4/3)])
#     validate(2, wbCgate, true_values, lattices)
# #^ test_wbC_gate()

