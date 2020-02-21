# generate_lattices.py


import SxPID
import time
import pickle 


def generate_lattice():
    """
    Pickles this object { n -> [{alpha -> children}, (alpha_1,...) ] }
    into lattices.pkl
    """
    lattices = dict()
    for n in range(2,5):
        lattice = SxPID.Lattice(n)
        achain = lattice.antichain()
        chld = dict()
        for alpha in achain:
            chld[alpha] = lattice.children(alpha, achain)
        #^ for
            
        lattices[n] = [chld, achain]
    #^ for
    f = open("lattices.pkl", "wb")
    pickle.dump(lattices, f)
    f.close()

    return 0
#^ generate_lattice()
