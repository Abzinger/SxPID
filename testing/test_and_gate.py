# test_and_gate.py
from sys import path
path.insert(0,"../sxpid")

import SxPID
 
import time
import pickle


# Unpickle the lattices from a file
# Pickled as { n -> [{alpha -> children}, (alpha_1,...) ] }
f = open("../sxpid/lattices.pkl", "rb")
lattices = pickle.load(f)

# Format of the pdf is 
# { (s1,s2,t) -> p(s1,s2,t) } for all s1 in S1, s2 in S2, and t in T.

# And: T = And(S1, S2)
andgate = dict()
andgate[(0,0,0)] = 0.25
andgate[(0,1,0)] = 0.25
andgate[(1,0,0)] = 0.25
andgate[(1,1,1)] = 0.25

# Arguments of pid()
# n : int - number of pid sources                                     
# pdf_orig : dict - the original joint distribution of the inputs and                                                                              
#                   the output (realizations are the keys). It doesn't have 
#                   to be a full support distribution, i.e., it can contain  
#                   realizations with 'zero' mass probability                
# chld :     dict - list of children for each node in the redundancy      
#                   lattice (nodes are the keys)                                 
# achain :   tuple - tuple of all the nodes (antichains) in the         
#                    redundacy lattice
# printing:  Bool - If true prints the results using PrettyTables

# Returns:
#          tuple                                                               
#          (pointwise_decomposition, averaged_decomposition)
#          pointwise_decomposition = { (s1,s2,t) -> { antichain -> (informative_pi, misinformative_pi, informative_pi - misinformative_pi) } }
#          average_decomposition   = { antichain -> (informative_pi, misinformative_pi, informative_pi - misinformative_pi) }

# Compute the pointwise partial information decomposition using SxPID.pid()
print("***********************************")
print("The SxPID for the Logic And(S1,S2):")
print("***********************************")
itic = time.process_time()
n = 2
pdf_orig = andgate
chld = lattices[n][0]
achain = lattices[n][1]
printing = True
ptw, avg = SxPID.pid(n, pdf_orig, chld, achain, printing)
itoc = time.process_time()

print("time: ", itoc - itic, "secs")



# Trivariate And: T = And(S1, S2, S3)

print("***********************************")
print("The SxPID for the Logic And(S1,S2,S3):")
print("***********************************")
triandgate = dict()
triandgate[(0,0,0,0)] = 1/8
triandgate[(0,0,1,0)] = 1/8
triandgate[(0,1,0,0)] = 1/8
triandgate[(0,1,1,0)] = 1/8
triandgate[(1,0,0,0)] = 1/8
triandgate[(1,0,1,0)] = 1/8
triandgate[(1,1,0,0)] = 1/8
triandgate[(1,1,1,1)] = 1/8

itic = time.process_time()
n = 3
pdf_orig = triandgate
chld = lattices[n][0]
achain = lattices[n][1]
printing = True
ptw, avg = SxPID.pid(n, pdf_orig, chld, achain, printing)
itoc = time.process_time()

print("time: ", itoc - itic, "secs")



# Quadvariate And: T = And(S1, S2, S3, S4)

print("***********************************")
print("The SxPID for the Logic And(S1,S2,S3,S4):")
print("***********************************")
quadandgate = dict()
quadandgate[(0,0,0,0,0)] = 1/16
quadandgate[(0,0,0,1,0)] = 1/16
quadandgate[(0,0,1,0,0)] = 1/16
quadandgate[(0,0,1,1,0)] = 1/16
quadandgate[(0,1,0,0,0)] = 1/16
quadandgate[(0,1,0,1,0)] = 1/16
quadandgate[(0,1,1,0,0)] = 1/16
quadandgate[(0,1,1,1,0)] = 1/16
quadandgate[(1,0,0,0,0)] = 1/16
quadandgate[(1,0,0,1,0)] = 1/16
quadandgate[(1,0,1,0,0)] = 1/16
quadandgate[(1,0,1,1,0)] = 1/16
quadandgate[(1,1,0,0,0)] = 1/16
quadandgate[(1,1,0,1,0)] = 1/16
quadandgate[(1,1,1,0,0)] = 1/16
quadandgate[(1,1,1,1,1)] = 1/16

itic = time.process_time()
n = 4
pdf_orig = quadandgate
chld = lattices[n][0]
achain = lattices[n][1]
printing = True
ptw, avg = SxPID.pid(n, pdf_orig, chld, achain, printing)
itoc = time.process_time()

print("time: ", itoc - itic, "secs")


#^EOF
