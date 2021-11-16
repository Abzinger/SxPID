# demo_and_gate.py

from sxpid import SxPID
import time

# Format of the pdf is 
# { (s1,s2,t) -> p(s1,s2,t) } for all s1 in S1, s2 in S2, and t in T.

# And: T = And(S1, S2)
andgate = dict()
andgate[(0,0,0)] = 0.25
andgate[(0,1,0)] = 0.25
andgate[(1,0,0)] = 0.25
andgate[(1,1,1)] = 0.25

# Arguments of pid()                              
# pdf_orig : dict - the original joint distribution of the inputs and                                                                              
#                   the output (realizations are the keys). It doesn't have 
#                   to be a full support distribution, i.e., it can contain  
#                   realizations with 'zero' mass probability                
# verbose:  int bitmask: 1 - Print intermediate steps
#                        2 - Show progress bar (slight performance decrease from the use of imap instead of map)
#                        4 - Print result tables
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
pdf_orig = andgate
ptw, avg = SxPID.pid(pdf_orig, verbose=4)
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
pdf_orig = triandgate
ptw, avg = SxPID.pid(pdf_orig, verbose=4)
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
pdf_orig = quadandgate
ptw, avg = SxPID.pid(pdf_orig, verbose=4)
itoc = time.process_time()

print("time: ", itoc - itic, "secs")


#^EOF
