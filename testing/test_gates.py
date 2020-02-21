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


for gate in gates.keys():
    print("***********************************")
    print("The JxPID for the ", gate, " :")
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

itic = time.process_time()
SxPID.pid(n, quadhashgate, lattices[n][0], lattices[n][1], True)
itoc = time.process_time()
print("time: ", itoc - itic, "secs")

# Bivariate 4-hash
n = 2
biquadhashgate = dict()
biquadhashgate[((0,0),(0,0),0)] = 1/16
biquadhashgate[((0,0),(0,1),1)] = 1/16
biquadhashgate[((0,0),(1,0),1)] = 1/16
biquadhashgate[((0,0),(1,1),0)] = 1/16
biquadhashgate[((0,1),(0,0),1)] = 1/16
biquadhashgate[((0,1),(0,1),0)] = 1/16
biquadhashgate[((0,1),(1,0),0)] = 1/16
biquadhashgate[((0,1),(1,1),1)] = 1/16
biquadhashgate[((1,0),(0,0),1)] = 1/16
biquadhashgate[((1,0),(0,1),0)] = 1/16
biquadhashgate[((1,0),(1,0),0)] = 1/16
biquadhashgate[((1,0),(1,1),1)] = 1/16
biquadhashgate[((1,1),(0,0),0)] = 1/16
biquadhashgate[((1,1),(0,1),1)] = 1/16
biquadhashgate[((1,1),(1,0),1)] = 1/16
biquadhashgate[((1,1),(1,1),0)] = 1/16

itic = time.process_time()
SxPID.pid(n, biquadhashgate, lattices[n][0], lattices[n][1], True)
itoc = time.process_time()
print("time: ", itoc - itic, "secs")

# Twist bivariate hash gate 
tbiquadhashgate = dict()
tbiquadhashgate[((0,0),(0,0),0)] = 1/8
tbiquadhashgate[((0,0),(0,1),1)] = 1/8
tbiquadhashgate[((0,1),(1,0),0)] = 1/8
tbiquadhashgate[((0,1),(1,1),1)] = 1/8
tbiquadhashgate[((1,0),(0,0),1)] = 1/8
tbiquadhashgate[((1,0),(0,1),0)] = 1/8
tbiquadhashgate[((1,1),(1,0),1)] = 1/8
tbiquadhashgate[((1,1),(1,1),0)] = 1/8

itic = time.process_time()
SxPID.pid(n, tbiquadhashgate, lattices[n][0], lattices[n][1], True)
itoc = time.process_time()
print("time: ", itoc - itic, "secs")
