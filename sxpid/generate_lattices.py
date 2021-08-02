# generate_lattices.py


import SxPID
import pickle as pkl
from tqdm import tqdm

def generate_lattice():
    """
    Pickles this object { n -> {alpha -> children}}
    into lattices.pkl
    """
    lattices = dict()
    for n in range(2,6):
        print(n)
        lattice = SxPID.Lattice(n)
        achain = lattice.antichain()
        chld = dict()
        for alpha in tqdm(achain):
            chld[alpha] = lattice.children(alpha, achain)
        #^ for
            
        lattices[n] = chld
    #^ for
    f = open("lattices.pkl", "wb")
    pkl.dump(lattices, f)
    f.close()
#^ generate_lattice()

if __name__=='__main__':
    generate_lattice()