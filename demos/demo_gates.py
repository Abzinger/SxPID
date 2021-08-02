# demo_gates.py
"""Demonstration for SxPID on logic gates."""
import time
from sxpid import SxPID

# Format of the pdf is 
# dict( (s1,s2,t) : p(s1,s2,t) ) for all s1 in S1, s2 in S2, and t in T if p(s1,s2,t) > 0.

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

# PwUnq
pwunqgate = dict()
pwunqgate[(0,1,1)] = 0.25
pwunqgate[(1,0,1)] = 0.25
pwunqgate[(0,2,2)] = 0.25
pwunqgate[(2,0,2)] = 0.25


# RndErr
rnderrgate = dict()
rnderrgate[(0,0,0)] = 3/8
rnderrgate[(1,1,1)] = 3/8
rnderrgate[(0,1,0)] = 1/8
rnderrgate[(1,0,1)] = 1/8


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

# Sum (N.B. equivalent to XorAnd)
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

# RndXor
rndxorgate = dict()
rndxorgate[(('r',0),('r',0),('r',0))] = 1/8
rndxorgate[(('r',0),('r',1),('r',1))] = 1/8
rndxorgate[(('r',1),('r',0),('r',1))] = 1/8
rndxorgate[(('r',1),('r',1),('r',0))] = 1/8

rndxorgate[(('R',0),('R',0),('R',0))] = 1/8
rndxorgate[(('R',0),('R',1),('R',1))] = 1/8
rndxorgate[(('R',1),('R',0),('R',1))] = 1/8
rndxorgate[(('R',1),('R',1),('R',0))] = 1/8

# Williams Beer 2010 examples

# Example A
wbAgate = dict()
wbAgate[(0,0,0)] = 1/3
wbAgate[(0,1,1)] = 1/3
wbAgate[(1,0,2)] = 1/3

# Example B
wbBgate = dict()
wbBgate[(0,0,0)] = 1/4
wbBgate[(0,1,1)] = 1/4
wbBgate[(1,0,2)] = 1/4
wbBgate[(1,1,1)] = 1/4


# Example C
wbCgate = dict()
wbCgate[(0,0,0)] = 1/6
wbCgate[(0,1,1)] = 1/6
wbCgate[(0,1,2)] = 1/6
wbCgate[(1,0,2)] = 1/6
wbCgate[(1,1,0)] = 1/6
wbCgate[(1,1,1)] = 1/6



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
gates["Williams-Beer-A"] = wbAgate
gates["Williams-Beer-B"] = wbBgate
gates["Williams-Beer-C"] = wbCgate

print("+++++++++++++++++++++++++++++++++++++++++++++")
print("The SxPID for Bivariate gates (T: S_1, S_2) :")
print("+++++++++++++++++++++++++++++++++++++++++++++\n\n")
for gate in gates.keys():
    print("**************************************")
    print("The SxPID for the ", gate, " :")
    print("**************************************")
    itic = time.process_time()
    SxPID.pid(gates[gate], verbose=4)
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
trihashgate[(0,0,0,0)] = 1/8
trihashgate[(0,0,1,1)] = 1/8
trihashgate[(0,1,0,1)] = 1/8
trihashgate[(0,1,1,0)] = 1/8
trihashgate[(1,0,0,1)] = 1/8
trihashgate[(1,0,1,0)] = 1/8
trihashgate[(1,1,0,0)] = 1/8
trihashgate[(1,1,1,1)] = 1/8

# TriRnd
trirndgate = dict()
trirndgate[(0,0,0,0)] = 1/2
trirndgate[(1,1,1,1)] = 1/2

# DblXor T = (S_1 Xor S_2 , S_2 Xor S_3)
dblxorgate = dict()
dblxorgate[(0,0,0,(0,0))] = 1/8
dblxorgate[(0,0,1,(0,1))] = 1/8
dblxorgate[(0,1,0,(1,1))] = 1/8
dblxorgate[(0,1,1,(1,0))] = 1/8
dblxorgate[(1,0,0,(1,0))] = 1/8
dblxorgate[(1,0,1,(1,1))] = 1/8
dblxorgate[(1,1,0,(0,1))] = 1/8
dblxorgate[(1,1,1,(0,0))] = 1/8

# Tri XorCopy T = (S_1, S_2, S_1 Xor S_2)
trixorcopygate = dict()
trixorcopygate[(0,0,0,(0,0,0))] = 1/4
trixorcopygate[(0,1,1,(0,1,1))] = 1/4
trixorcopygate[(1,0,1,(1,0,1))] = 1/4
trixorcopygate[(1,1,0,(1,1,0))] = 1/4

# XorLoss T = S_3 = S_1 Xor S_2
xorlossgate = dict()
xorlossgate[(0,0,0,0)] = 1/4
xorlossgate[(0,1,1,1)] = 1/4
xorlossgate[(1,0,1,1)] = 1/4
xorlossgate[(1,1,0,0)] = 1/4

# XorDuplicate T =  S_1 Xor S_2; S_3 = S_1
xordupgate = dict()
xordupgate[(0,0,0,0)] = 1/4
xordupgate[(0,1,0,1)] = 1/4
xordupgate[(1,0,1,1)] = 1/4
xordupgate[(1,1,1,0)] = 1/4

# AndDuplicate T =  S_1 And S_2; S_3 = S_1
anddupgate = dict()
anddupgate[(0,0,0,0)] = 1/4
anddupgate[(0,1,0,0)] = 1/4
anddupgate[(1,0,1,0)] = 1/4
anddupgate[(1,1,1,1)] = 1/4

# XorMultiCoal S_1 = (x_1, x_2); S_2 = (x_1, x_3); S_3 = (x_2, x_3);
# T = x_1 Xor x_2, Xor x_3
xormulticoalgate = dict()
xormulticoalgate[(('a','b'),('a','c'),('b','c'),0)] = 1/8
xormulticoalgate[(('A','B'),('A','c'),('B','c'),0)] = 1/8
xormulticoalgate[(('A','b'),('A','C'),('b','C'),0)] = 1/8
xormulticoalgate[(('a','B'),('a','C'),('B','C'),0)] = 1/8
xormulticoalgate[(('A','b'),('A','c'),('b','c'),1)] = 1/8
xormulticoalgate[(('a','B'),('a','c'),('B','c'),1)] = 1/8
xormulticoalgate[(('a','b'),('a','C'),('b','C'),1)] = 1/8
xormulticoalgate[(('A','B'),('A','C'),('B','C'),1)] = 1/8


# Majority gate T = Sgn(S_1 + S_2 + S_3)
majgate = dict()
majgate[(-1,-1,-1,-1)] = 1/8
majgate[(-1,-1,+1,-1)] = 1/8
majgate[(-1,+1,-1,-1)] = 1/8
majgate[(-1,+1,+1, 1)] = 1/8
majgate[(+1,-1,-1,-1)] = 1/8
majgate[(+1,-1,+1, 1)] = 1/8
majgate[(+1,+1,-1, 1)] = 1/8
majgate[(+1,+1,+1, 1)] = 1/8

gates = dict()
gates["3-bits hash"] = trihashgate
gates["Giant Bit"] = trirndgate
gates["DblXor"] = dblxorgate
gates["XorCopy"] = trixorcopygate
gates["XorLoss"] = xorlossgate
gates["XorDuplicate"] = xordupgate
gates["AndDuplicate"] = anddupgate
gates["XorMultiCoal"] = xormulticoalgate
gates["Majgate"] = majgate

print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
print("The SxPID for Trivariate gates (T: S_1, S_2, S_3) :")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
for gate in gates.keys():
    print("**************************************")
    print("The SxPID for the ", gate, " :")
    print("**************************************")
    itic = time.process_time()
    SxPID.pid(gates[gate], verbose=4)
    itoc = time.process_time()
    print("time: ", itoc - itic, "secs")

#^ for gate


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

# Quad Majority gate T = Sgn(S_1 + S_2 + S_3)
quadmajgate = dict()
quadmajgate[(-1,-1,-1,-1,-1)] = 1/10
quadmajgate[(-1,-1,-1,+1,-1)] = 1/10
quadmajgate[(-1,-1,+1,-1,-1)] = 1/10
quadmajgate[(-1,-1,+1,+1,+0)] = 0.
quadmajgate[(-1,+1,-1,-1,-1)] = 1/10
quadmajgate[(-1,+1,-1,+1,+0)] = 0.
quadmajgate[(-1,+1,+1,-1,+0)] = 0.
quadmajgate[(-1,+1,+1,+1,+1)] = 1/10
quadmajgate[(+1,-1,-1,-1,-1)] = 1/10
quadmajgate[(+1,-1,-1,+1,+0)] = 0.
quadmajgate[(+1,-1,+1,-1,+0)] = 0.
quadmajgate[(+1,-1,+1,+1,+1)] = 1/10
quadmajgate[(+1,+1,-1,-1,+0)] = 0.
quadmajgate[(+1,+1,-1,+1,+1)] = 1/10
quadmajgate[(+1,+1,+1,-1,+1)] = 1/10
quadmajgate[(+1,+1,+1,+1,+1)] = 1/10

gates = dict()
gates["4 Parity"] = quadhashgate
gates["4 Majority"] = quadmajgate

print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
print("The SxPID for Quadvariate gates (T: S_1, S_2, S_3, S_4) :")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
for gate in gates.keys():
    print("**************************************")
    print("The SxPID for the ", gate, " :")
    print("**************************************")
    itic = time.process_time()
    SxPID.pid(gates[gate], verbose=4)
    itoc = time.process_time()
    print("time: ", itoc - itic, "secs")

#^ for gate

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
