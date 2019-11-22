"""
JxPID
"""
import numpy as np
import math
import time
from itertools import chain, combinations
from prettytable import PrettyTable

#---------
# Lattice 
#---------
class Lattice():
    def __init__(self, n):
        self.n = n
        self.lis = [i for i in range(1,self.n+1)]
    #^ _init_()
    
    def powerset(self):
        return chain.from_iterable(combinations(self.lis, r) for r in range(1,len(self.lis) + 1) )
    #^ powerset()

    def less_than(self, beta, alpha):
        # compare whether an antichain beta is smaller than antichain alpha
        return all(any(frozenset(b) <= frozenset(a) for b in beta) for a in alpha)
    #^ compare()

    def comparable(self, a,b):
        return a < b or a > b
    #^ comparable()

    def antichain(self):
        # dummy expensive function might use dit or networkx functions
        # assert self.n < 5, "antichain(n): number of sources should be less than 5"
        achain = []
        for r in range(1, math.floor((2**self.n - 1)/2) + 2):
            # enumerate the power set of the powerset
            for alpha in combinations(self.powerset(), r):
                flag = 1
                # check if alpha is an antichain
                for a in list(alpha):
                    for b in list(alpha):
                        if a < b and self.comparable(frozenset(a),frozenset(b)): flag = 0 
                    #^ for b
                #^ for a
                if flag: achain.append(alpha)
            #^ for alpha
        #^ for r 
        return achain
    #^ antichain()

    def children(self, alpha, achain):
        chl = []
        downset = [beta for beta in achain if self.less_than(beta,alpha) and beta != alpha]
        for beta in downset:
            if all(not self.less_than(beta,gamma) for gamma in downset if gamma != beta):
                chl.append(beta)
            #^ if
        #^ for beta
        return chl
    #^ children()

#^ Lattice()

#---------------
# pi^+(t:alpha)
#    and
# pi^-(t:alpha) 
#---------------

def powerset(n):
    lis = [i for i in range(1,n+1)]
    return chain.from_iterable(combinations(lis, r) for r in range(1,len(lis) + 1) )
#^ powerset()

def marg(pdf, rlz, uset):
    idxs = [ idx - 1 for idx in list(uset)]
    summ = 0.
    for k in pdf.keys():
        if all(k[idx] == rlz[idx] for idx in idxs): summ += pdf[k]
    #^ for
    return summ
#^ marg()
    
def prob(n, pdf, rlz, gamma, target=False):
    m = len(gamma)
    pset = powerset(m)
    summ = 0
    for idxs in pset:
        if target:
            uset = frozenset((n+1,))
        else:
            uset = frozenset(())
        #^ if 
        for i in list(idxs):
            uset |= frozenset(gamma[i-1])
        #^ for
        summ += (-1)**(len(idxs) + 1) * marg(pdf, rlz, uset)
    #^ for
    return summ
#^ prob()

def differs(n, pdf, rlz, alpha, chl, target=False):
    if chl == [] and target:
        base = prob(n, pdf, rlz, [()], target)/prob(n, pdf, rlz, alpha, target)
    else:
        base = prob(n, pdf, rlz, alpha, target)
    #^ if bottom
    temp_diffs = [prob(n, pdf, rlz, gamma, target) - base for gamma in chl]
    temp_diffs.sort()
    return [base] + temp_diffs
#^ differs()

def sgn(num_chld):
    if num_chld == 0:
        return np.array([+1])
    else:
        return np.concatenate((sgn(num_chld - 1), -sgn(num_chld - 1)), axis=None)
#^sgn()

def vec(num_chld, diffs):
    """
    Args: 
         num_chld : the number of the children of alpha: (gamma_1,...,gamma_{num_chld}) 
         diffs : vector of probability differences (d_i)_i where d_i = p(gamma_i) - p(alpha) and d_0 = p(alpha)  
    """
    # print(diffs)
    if num_chld == 0:
        return np.array([diffs[0]])
    else:
        temp = vec(num_chld - 1, diffs) + diffs[num_chld]*np.ones(2**(num_chld - 1))
        return np.concatenate((vec(num_chld - 1, diffs), temp), axis=None)
#^ vec()

def pi_plus(n, pdf, rlz, alpha, chld, achain):
    diffs = differs(n, pdf, rlz, alpha, chld[tuple(alpha)], False)
    return np.dot(sgn(len(chld[alpha])), -np.log2(vec(len(chld[alpha]),diffs)))
#^ pi_plus()

def pi_minus(n, pdf, rlz, alpha, chld, achain):
    diffs = differs(n, pdf, rlz, alpha, chld[alpha], True)
    if chld[tuple(alpha)] == []:
        return np.dot(sgn(len(chld[alpha])), np.log2(vec(len(chld[alpha]),diffs)))
    else:
        return np.dot(sgn(len(chld[alpha])), -np.log2(vec(len(chld[alpha]),diffs)))
#^ pi_minus()

def jxpid(n, pdf, chld, achain, printing=True):
    ptw = dict()
    avg = dict()
    for rlz in pdf.keys():
        ptw[rlz] = dict()
        for alpha in achain:
            piplus = pi_plus(n, pdf, rlz, alpha, chld, achain)
            piminus = pi_minus(n, pdf, rlz, alpha, chld, achain)
            ptw[rlz][alpha] = (piplus, piminus, piplus - piminus)
        #^ for
    #^ for
    for alpha in achain:
        avgplus = 0.
        avgminus = 0.
        avgdiff = 0.
        for rlz in pdf.keys():
            avgplus  += pdf[rlz]*ptw[rlz][alpha][0]
            avgminus += pdf[rlz]*ptw[rlz][alpha][1]
            avgdiff  += pdf[rlz]*ptw[rlz][alpha][2]
            avg[alpha] = (avgplus, avgminus, avgdiff)
        #^ for
    #^ for
    if printing:
        table = PrettyTable()
        table.field_names = ["RLZ", "Atom", "pi+", "pi-", "pi"]
        for rlz in pdf.keys():
            count = 0
            for alpha in achain:
                stalpha = ""
                for a in alpha:
                    stalpha += "{"
                    for i in a:
                        stalpha += str(i)
                    #^ for i
                    stalpha += "}" 
                #^ for a
                if count == 0: table.add_row( [str(rlz), stalpha, str(ptw[rlz][alpha][0]), str(ptw[rlz][alpha][1]), str(ptw[rlz][alpha][2])] )
                else:          table.add_row( [" ", stalpha, str(ptw[rlz][alpha][0]), str(ptw[rlz][alpha][1]), str(ptw[rlz][alpha][2])] )
                count += 1 
            #^ for alpha
            table.add_row(["*", "*", "*", "*", "*"])
        #^ for realization

        table.add_row(["-", "-", "-", "-", "-"])
        count = 0
        for alpha in achain:
            stalpha = ""
            for a in alpha:
                stalpha += "{"
                for i in a:
                    stalpha += str(i)
                #^ for i
                stalpha += "}" 
            #^ for a
            if count == 0: table.add_row( ["avg", stalpha, str(avg[alpha][0]), str(avg[alpha][1]), str(avg[alpha][2])] )
            else:          table.add_row( [" ", stalpha, str(avg[alpha][0]), str(avg[alpha][1]), str(avg[alpha][2])] )
            count += 1
        #^ for alpha
        print(table)
    #^ if printing
    
    return ptw, avg
#^ jxpid()

#--------
# Test!
#-------

# Format of the pdf is 
# dict( (s1,s2,t) : p(s1,s2,t) ) for all s1 in S1, s2 in S2, and t in T if p(s1,s2,t) > 0.

# Bivariate
n = 2
lattice = Lattice(n)
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
    jxpid(n, gates[gate], chld, achain, True)
    itoc = time.process_time()
    print("time: ", itoc - itic, "secs")

#^ for gate

# # Trivariate
n = 3
lattice = Lattice(n)
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
jxpid(n, trihashgate, chld, achain, True)
itoc = time.process_time()
print("time: ", itoc - itic, "secs")


# Quadvariate
n = 4
lattice = Lattice(n)
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
jxpid(n, quadhashgate, chld, achain, True)
itoc = time.process_time()
print("time: ", itoc - itic, "secs")
