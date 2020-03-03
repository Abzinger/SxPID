# test_gates.py

from sys import path
path.insert(0,"../sxpid")

import SxPID
 
import time
import pickle 
#--------
# Test!
#-------

# Format of the pdf is 
# dict( (s1,s2,t) : p(s1,s2,t) ) for all s1 in S1, s2 in S2, and t in T if p(s1,s2,t) > 0.


# Read lattices from a file
# Pickled as { n -> [{alpha -> children}, (alpha_1,...) ] }
f = open("../sxpid/lattices.pkl", "rb")
lattices = pickle.load(f)


# Bivariate
n = 2

# Xor
xorgate = dict()
xorgate[(0,0,0)] = 0.25
xorgate[(0,1,1)] = 0.25
xorgate[(1,0,1)] = 0.25
xorgate[(1,1,0)] = 0.25


# And
andgate = dict()
andgate[(0,0,0)] = 0.25
andgate[(0,1,0)] = 0.25
andgate[(1,0,0)] = 0.25
andgate[(1,1,1)] = 0.25

# Unq
unqgate = dict()
unqgate[(0,0,0)] = 0.25
unqgate[(0,1,0)] = 0.25
unqgate[(1,0,1)] = 0.25
unqgate[(1,1,1)] = 0.25

# PwUnq
pwunqgate = dict()
pwunqgate[(0,1,1)] = 0.25
pwunqgate[(1,0,1)] = 0.25
pwunqgate[(0,2,2)] = 0.25
pwunqgate[(2,0,2)] = 0.25

# Rnd
rndgate = dict()
rndgate[(0,0,0)] = 0.5
rndgate[(1,1,1)] = 0.5

# RndErr
rnderrgate = dict()
rnderrgate[(0,0,0)] = 3/8
rnderrgate[(1,1,1)] = 3/8
rnderrgate[(0,1,0)] = 1/8
rnderrgate[(1,0,1)] = 1/8

# Copy
copygate = dict()
copygate[(0,0,(0,0))] = 0.25
copygate[(0,1,(0,1))] = 0.25
copygate[(1,0,(1,0))] = 0.25
copygate[(1,1,(1,1))] = 0.25

# (S1, XOR)
copyxorgate = dict()
copyxorgate[(0,0,(0,0))] = 0.25
copyxorgate[(0,1,(0,1))] = 0.25
copyxorgate[(1,0,(1,1))] = 0.25
copyxorgate[(1,1,(1,0))] = 0.25

# (S2, Xor)
xorcopygate = dict()
xorcopygate[(0,0,(0,0))] = 0.25
xorcopygate[(0,1,(1,1))] = 0.25
xorcopygate[(1,0,(0,1))] = 0.25
xorcopygate[(1,1,(1,0))] = 0.25

# RndUnqXor
rndunqxorgate = dict()
rndunqxorgate[(('r','a',0),('r','b',0),('r','a','b',0))] = 1/32
rndunqxorgate[(('r','a',0),('r','b',1),('r','a','b',1))] = 1/32
rndunqxorgate[(('r','a',1),('r','b',0),('r','a','b',1))] = 1/32
rndunqxorgate[(('r','a',1),('r','b',1),('r','a','b',0))] = 1/32

rndunqxorgate[(('r','a',0),('r','B',0),('r','a','B',0))] = 1/32
rndunqxorgate[(('r','a',0),('r','B',1),('r','a','B',1))] = 1/32
rndunqxorgate[(('r','a',1),('r','B',0),('r','a','B',1))] = 1/32
rndunqxorgate[(('r','a',1),('r','B',1),('r','a','B',0))] = 1/32

rndunqxorgate[(('r','A',0),('r','b',0),('r','A','b',0))] = 1/32
rndunqxorgate[(('r','A',0),('r','b',1),('r','A','b',1))] = 1/32
rndunqxorgate[(('r','A',1),('r','b',0),('r','A','b',1))] = 1/32
rndunqxorgate[(('r','A',1),('r','b',1),('r','A','b',0))] = 1/32

rndunqxorgate[(('r','A',0),('r','B',0),('r','A','B',0))] = 1/32
rndunqxorgate[(('r','A',0),('r','B',1),('r','A','B',1))] = 1/32
rndunqxorgate[(('r','A',1),('r','B',0),('r','A','B',1))] = 1/32
rndunqxorgate[(('r','A',1),('r','B',1),('r','A','B',0))] = 1/32

rndunqxorgate[(('R','a',0),('R','b',0),('R','a','b',0))] = 1/32
rndunqxorgate[(('R','a',0),('R','b',1),('R','a','b',1))] = 1/32
rndunqxorgate[(('R','a',1),('R','b',0),('R','a','b',1))] = 1/32
rndunqxorgate[(('R','a',1),('R','b',1),('R','a','b',0))] = 1/32

rndunqxorgate[(('R','a',0),('R','B',0),('R','a','B',0))] = 1/32
rndunqxorgate[(('R','a',0),('R','B',1),('R','a','B',1))] = 1/32
rndunqxorgate[(('R','a',1),('R','B',0),('R','a','B',1))] = 1/32
rndunqxorgate[(('R','a',1),('R','B',1),('R','a','B',0))] = 1/32

rndunqxorgate[(('R','A',0),('R','b',0),('R','A','b',0))] = 1/32
rndunqxorgate[(('R','A',0),('R','b',1),('R','A','b',1))] = 1/32
rndunqxorgate[(('R','A',1),('R','b',0),('R','A','b',1))] = 1/32
rndunqxorgate[(('R','A',1),('R','b',1),('R','A','b',0))] = 1/32

rndunqxorgate[(('R','A',0),('R','B',0),('R','A','B',0))] = 1/32
rndunqxorgate[(('R','A',0),('R','B',1),('R','A','B',1))] = 1/32
rndunqxorgate[(('R','A',1),('R','B',0),('R','A','B',1))] = 1/32
rndunqxorgate[(('R','A',1),('R','B',1),('R','A','B',0))] = 1/32

# Sum
sumgate = dict()
sumgate[(0,0,0)] = 1/4
sumgate[(0,1,1)] = 1/4
sumgate[(1,0,1)] = 1/4
sumgate[(1,1,2)] = 1/4

# Reduced Or
reduorgate = dict()
reduorgate[(0,0,0)] = 1/2
reduorgate[(0,1,1)] = 1/4
reduorgate[(1,0,1)] = 1/4

# Rnd Xor
rndxorgate = dict()
rndxorgate[(('r',0),('r',0),('r',0))] = 1/8
rndxorgate[(('r',0),('r',1),('r',1))] = 1/8
rndxorgate[(('r',1),('r',0),('r',1))] = 1/8
rndxorgate[(('r',1),('r',1),('r',0))] = 1/8

rndxorgate[(('R',0),('R',0),('R',0))] = 1/8
rndxorgate[(('R',0),('R',1),('R',1))] = 1/8
rndxorgate[(('R',1),('R',0),('R',1))] = 1/8
rndxorgate[(('R',1),('R',1),('R',0))] = 1/8

gates = dict()
gates["Xor"]  = xorgate
gates["And"]  = andgate
gates["Unq"]  = unqgate
gates["PwUnq"]  = pwunqgate
gates["Rnd"]  = rndgate
gates["RndErr"]  = rnderrgate
gates["Copy"] = copygate
gates["(S1,Xor)"] = copyxorgate
gates["(S2,Xor)"] = xorcopygate
gates["RndUnqXor"] = rndunqxorgate
gates["Sum"] = sumgate
gates["ReduOr"] = reduorgate
gates["RndXor"] = rndxorgate

for gate in gates.keys():
    print("***********************************")
    print("The SxPID for the ", gate, " :")
    print("***********************************")
    itic = time.process_time()
    SxPID.pid(n, gates[gate], lattices[n][0], lattices[n][1], True)
    itoc = time.process_time()
    print("time: ", itoc - itic, "secs")

#^ for gate

# # NXor
# for i in range(11):
#     eps = i/40
#     nxorgate = dict()
#     nxorgate[(0,0,0)] = 0.25 - eps
#     nxorgate[(0,1,1)] = 0.25 - eps
#     nxorgate[(1,0,1)] = 0.25 - eps
#     nxorgate[(1,1,0)] = 0.25 - eps
#     nxorgate[(0,0,1)] = eps
#     nxorgate[(0,1,0)] = eps
#     nxorgate[(1,0,0)] = eps
#     nxorgate[(1,1,1)] = eps
#     print("***********************************")
#     print("The JxPID for the Nosiy Xor with pertubabtion", eps, " :")
#     print("***********************************")
#     itic = time.process_time()
#     SxPID.pid(n, nxorgate, lattices[n][0], lattices[n][1], True)
#     itoc = time.process_time()
#     print("time: ", itoc - itic, "secs")
# #^ for i

# Trivariate
n = 3

# Trihash
trihashgate = dict()
trihashgate[(0,0,0,0)] = 0.125
trihashgate[(0,0,1,1)] = 0.125
trihashgate[(0,1,0,1)] = 0.125
trihashgate[(0,1,1,0)] = 0.125
trihashgate[(1,0,0,1)] = 0.125
trihashgate[(1,0,1,0)] = 0.125
trihashgate[(1,1,0,0)] = 0.125
trihashgate[(1,1,1,1)] = 0.125

print("***********************************")
print("The SxPID for the three bit hash:")
print("***********************************")
itic = time.process_time()
SxPID.pid(n, trihashgate, lattices[n][0], lattices[n][1], True)
itoc = time.process_time()
print("time: ", itoc - itic, "secs")


# Quadvariate
n = 4

# Quadhash
quadhashgate = dict()
quadhashgate[(0,0,0,0,0)] = 1/16
quadhashgate[(0,0,0,1,1)] = 1/16
quadhashgate[(0,0,1,0,1)] = 1/16
quadhashgate[(0,0,1,1,0)] = 1/16
quadhashgate[(0,1,0,0,1)] = 1/16
quadhashgate[(0,1,0,1,0)] = 1/16
quadhashgate[(0,1,1,0,0)] = 1/16
quadhashgate[(0,1,1,1,1)] = 1/16
quadhashgate[(1,0,0,0,1)] = 1/16
quadhashgate[(1,0,0,1,0)] = 1/16
quadhashgate[(1,0,1,0,0)] = 1/16
quadhashgate[(1,0,1,1,1)] = 1/16
quadhashgate[(1,1,0,0,0)] = 1/16
quadhashgate[(1,1,0,1,1)] = 1/16
quadhashgate[(1,1,1,0,1)] = 1/16
quadhashgate[(1,1,1,1,0)] = 1/16

print("***********************************")
print("The SxPID for the four bit hash:")
print("***********************************")
itic = time.process_time()
SxPID.pid(n, quadhashgate, lattices[n][0], lattices[n][1], True)
itoc = time.process_time()
print("time: ", itoc - itic, "secs")

# # Bivariate 4-hash
# n = 2
# biquadhashgate = dict()
# biquadhashgate[((0,0),(0,0),0)] = 1/16
# biquadhashgate[((0,0),(0,1),1)] = 1/16
# biquadhashgate[((0,0),(1,0),1)] = 1/16
# biquadhashgate[((0,0),(1,1),0)] = 1/16
# biquadhashgate[((0,1),(0,0),1)] = 1/16
# biquadhashgate[((0,1),(0,1),0)] = 1/16
# biquadhashgate[((0,1),(1,0),0)] = 1/16
# biquadhashgate[((0,1),(1,1),1)] = 1/16
# biquadhashgate[((1,0),(0,0),1)] = 1/16
# biquadhashgate[((1,0),(0,1),0)] = 1/16
# biquadhashgate[((1,0),(1,0),0)] = 1/16
# biquadhashgate[((1,0),(1,1),1)] = 1/16
# biquadhashgate[((1,1),(0,0),0)] = 1/16
# biquadhashgate[((1,1),(0,1),1)] = 1/16
# biquadhashgate[((1,1),(1,0),1)] = 1/16
# biquadhashgate[((1,1),(1,1),0)] = 1/16

# itic = time.process_time()
# SxPID.pid(n, biquadhashgate, lattices[n][0], lattices[n][1], True)
# itoc = time.process_time()
# print("time: ", itoc - itic, "secs")

# # Twist bivariate hash gate 
# tbiquadhashgate = dict()
# tbiquadhashgate[((0,0),(0,0),0)] = 1/8
# tbiquadhashgate[((0,0),(0,1),1)] = 1/8
# tbiquadhashgate[((0,1),(1,0),0)] = 1/8
# tbiquadhashgate[((0,1),(1,1),1)] = 1/8
# tbiquadhashgate[((1,0),(0,0),1)] = 1/8
# tbiquadhashgate[((1,0),(0,1),0)] = 1/8
# tbiquadhashgate[((1,1),(1,0),1)] = 1/8
# tbiquadhashgate[((1,1),(1,1),0)] = 1/8

# itic = time.process_time()
# SxPID.pid(n, tbiquadhashgate, lattices[n][0], lattices[n][1], True)
# itoc = time.process_time()
# print("time: ", itoc - itic, "secs")
