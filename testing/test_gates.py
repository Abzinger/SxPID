# test_gates.py

from sys import path
path.insert(0,"../jxpid")

import JxPID

import time
#--------
# Test!
#-------

# Format of the pdf is 
# dict( (s1,s2,t) : p(s1,s2,t) ) for all s1 in S1, s2 in S2, and t in T if p(s1,s2,t) > 0.

# Bivariate
n = 2
lattice = JxPID.Lattice(n)
achain = lattice.antichain()
chld = dict()
for alpha in achain:
    chld[alpha] = lattice.children(alpha, achain)
#^ for


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

# Rnd
rndgate = dict()
rndgate[(0,0,0)] = 0.5
rndgate[(1,1,1)] = 0.5

# Copy
copygate = dict()
copygate[(0,0,(0,0))] = 0.25
copygate[(0,1,(0,1))] = 0.25
copygate[(1,0,(1,0))] = 0.25
copygate[(1,1,(1,1))] = 0.25

gates = dict()
gates["Xor"]  = xorgate
gates["And"]  = andgate
gates["Unq"]  = unqgate
gates["Rnd"]  = rndgate
gates["Copy"] = copygate

for gate in gates.keys():
    print("***********************************")
    print("The JxPID for the ", gate, " :")
    print("***********************************")
    itic = time.process_time()
    JxPID.pid(n, gates[gate], chld, achain, True)
    itoc = time.process_time()
    print("time: ", itoc - itic, "secs")

#^ for gate

# # Trivariate
n = 3
lattice = JxPID.Lattice(n)
achain = lattice.antichain()
chld = dict()
for alpha in achain:
    chld[alpha] = lattice.children(alpha, achain)
#^ for

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
JxPID.pid(n, trihashgate, chld, achain, True)
itoc = time.process_time()
print("time: ", itoc - itic, "secs")


# Quadvariate
n = 4
lattice = JxPID.Lattice(n)
achain = lattice.antichain()
chld = dict()
for alpha in achain:
    chld[alpha] = lattice.children(alpha, achain)
#^ for

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
JxPID.pid(n, quadhashgate, chld, achain, True)
itoc = time.process_time()
print("time: ", itoc - itic, "secs")
