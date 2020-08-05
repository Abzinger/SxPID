"""                                                                            
Shared exclusion partial information decomposition (SxPID)                      
"""

from sxpid import lattices as lt
import numpy as np
import math
import time
from itertools import chain, combinations
from collections import defaultdict
from prettytable import PrettyTable

from tqdm import tqdm

import multiprocessing as mp

from functools import lru_cache
from functools import partial

import logging

#---------
# Lattice 
#---------
class Lattice:
    """Generates the redundancy lattice for 'n' sources                         
    The algerbric structure on which partial information decomposition is       
    build on.                                                                   
    """
    def __init__(self, n):
        self.n = n
        self.lis = [i for i in range(1,self.n+1)]
    #^ _init_()
    
    def powerset(self):
        return chain.from_iterable(combinations(self.lis, r) for r in range(1,len(self.lis) + 1) )
    #^ powerset()

    def less_than(self, beta, alpha):
        """compare whether an antichain beta is smaller than antichain          
        alpha"""
        return all(any(frozenset(b) <= frozenset(a) for b in beta) for a in alpha)
    #^ compare()

    def comparable(self, a,b):
        return a < b or a > b
    #^ comparable()

    def antichain(self):
        """Generates the nodes (antichains) of the lattice"""
        # dummy expensive function might use dit or networkx functions
        assert self.n < 5, "antichain(n): number of sources should be less than 5"
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
        """Enumerates the direct nodes (antichains) ordered by the node         
        (antichain) 'alpha'"""
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

#---------
# PDF 
#---------

class PDF:
    """
    Internal representation of a sparse joint probability mass function of nVar variables.

    Uses COO representation with separate coords and probs numpy array and provides fast numpy vectorized methods for marginalizing.
    """

    def __init__(self, coords, probs, labels=None):
        """
        Create a new PDF from a given coordinate and probability array.

        Args:
            coords: Numpy array of shape (nRlz, nVar) of unsigned integers containing the indices of realizations with non-zero probability. Realizations are assumed to be unique.
                    Use Fortran contiguity for improved performance.
            probs:  Numpy array of shape (nRlz,) of floats containing the corresponding non-zero probabilities
            labels: List of lists, ith list gives the conversion from coordinates to event labels of the ith coordinate
        """

        assert type(coords) is np.ndarray, "Coordinates must be numpy array"
        assert type(probs) is np.ndarray, "Probabilities must be numpy array"

        assert coords.ndim == 2, "Coordinate array must have dimensions (nRealisations, nVariables)"
        assert probs.ndim == 1, "Probability array must have dimension (nRealisations,)"

        assert coords.shape[0] == probs.shape[0], "Dimensions of coordinate array (nRealisations, nVariables) and probability array (nRealisations,) do not match"

        assert np.sum(probs) - 1 < 1e-10, "Probabilities must sum up to 1"
        
        if(not coords.data.f_contiguous):
            print("Coordinate array in PDF is not Fortran contiguous, which is bad for performance!")

        self.coords = coords
        self.probs = probs
        self.labels = labels

        #number of random variables
        self.nVar = self.coords.shape[1]

        #number of realizations with non-zero mass
        self.nRlz = self.coords.shape[0]

    @classmethod
    def from_dict(cls, pdf_dict):
        """
        Creates a PDF from a pdf dictionary {(Rlz tuple):probability}

        Impossible realizations are automatically filtered out. The smallest possible unsigned integer datatype is used for the coordinate array.

        Args:
            pdf_dict: Dictionary of realization tuples and their corresponding probabilities. Labels in realization tuples can be arbitrary comparable (equals) objects.

        Returns:
            A new PDF with the data from the dictionary
        """

        # Remove the impossible realization
        pdf_dict = {k:v for k,v in pdf_dict.items() if v > 1.e-300 }

        nKeys = len(pdf_dict.keys())
        nVar  = len(next(iter(pdf_dict.keys())))

        keys, vals = zip(*pdf_dict.items())

        events = np.empty((nKeys, nVar), dtype=object)
        events[:] = keys
        
        coords = np.empty((nKeys, nVar), order='F', dtype=np.uint64)
        labels = []
        for i in range(nVar):
            labels_column, inverse_column = np.unique(events[:,i], return_inverse=True)
            coords[:,i] = inverse_column
            labels.append(labels_column)

        #Use smallest integer datatype possible for coordinates.
        #Improves memory consumption and execution speeds.
        alphabet_sizes = np.array(list(map(len, labels)))
        max_var = np.max(alphabet_sizes)
        
        uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]
        uint_max   = np.array([np.iinfo(t).max for t in uint_types], dtype=np.uint64)
        dtype = uint_types[np.argmax(max_var <= uint_max)]

        coords = coords.astype(dtype)

        probs   = np.array(vals)

        pdf = PDF(coords, probs, labels)
        
        return pdf

    def get_labels(self):
        """
        Returns list of length nRlz with the original event labels, reconstructed from labels. If no labels are given, returns list of tuples of coordinates.

        Returns:
            List of tuples
        """

        labels_columns = ()
        for i in range(self.nVar):
            coord_column = self.coords[:,i]
            labels_columns += (list(self.labels[i][coord_column] if self.labels else coord_column),)
        labels = list(zip(*labels_columns))

        return labels

    def probu_one_rlz(self, rlz, union_masks):
        """
        Calculate the probability of a union of marginals of sets all including the realization rlz

        Args:
            rlz: Numpy array of shape (nVar,) with coordinates of the realization that is conatined in all sets.
            union_masks: Numpy array of shape (nSets, nVar) with each row of shape (nVar,) indicating which variables to marginalize over.

        Returns:
            summ: Float signifying the probability of a union of marginals
        """

        rlz_mask = rlz == self.coords
        sum_mask = np.all( ~union_masks[0] | rlz_mask, axis=-1)
        for set_mask in union_masks[1:]:
            sum_mask = sum_mask | np.all( ~set_mask | rlz_mask, axis=-1)
        summ = np.sum(self.probs, where=sum_mask)
        return summ


#---------------
# pi^+(t:alpha)
#    and
# pi^-(t:alpha) 
#---------------

def differs(pdf, rlz, alpha, chl, target=False):
    """Compute the probability mass difference                                  
    For a node 'alpha' and any child gamma of alpha it computes p(gamma) -      
    p(alpha) for all gamma"""

    chlt = []
    for c in chl:
        newcolumn = np.full((c.shape[0], 1), target)
        chlt.append(np.append(c, newcolumn, axis=1))

    newcolumn = np.full((alpha.shape[0], 1), target)
    alphat = np.append(alpha, newcolumn, axis=1)

    if chlt == [] and target:
        full = np.full((1, pdf.nVar), False)
        full[:,-1] = target
        base = pdf.probu_one_rlz(rlz, full) / pdf.probu_one_rlz(rlz, alphat)
    else:
        base = pdf.probu_one_rlz(rlz, alphat)

    temp_diffs = [pdf.probu_one_rlz(rlz, gamma) - base for gamma in chlt]
    temp_diffs.sort()
    return [base] + temp_diffs

@lru_cache(32)
def sgn(num_chld):
    """
    Recurrsive function that generates the signs (+ or -) for the            
    inclusion-exculison principle
    """
    if num_chld == 0:
        return np.array([+1])
    else:
        rec = sgn(num_chld - 1)
        return np.concatenate((rec, -rec))
    
def vec(num_chld, diffs):
    """
    Args: 
      num_chld : the number of the children of alpha: (gamma_1,...,gamma_{num_chld}) 
      diffs : vector of probability differences 
              (d_i)_i where d_i = p(gamma_i) - p(alpha) and d_0 = p(alpha)  
    """
    vec = np.empty(2**num_chld)
    vec[0] = diffs[0]
    for i in range(num_chld):
        length = 2**i
        vec[length:2*length] = vec[0:length] + diffs[i+1]
    return vec

def pi_plus(pdf, rlz, alpha, chld):
     """Compute the informative PPID """
     diffs = differs(pdf, rlz, alpha, chld, False)
     return np.dot(sgn(len(chld)), -np.log2(vec(len(chld),diffs)))
#^ pi_plus()

def pi_minus(pdf, rlz, alpha, chld):
    """Compute the misinformative PPID """
    diffs = differs(pdf, rlz, alpha, chld, True)
    if chld == []:
        return np.dot(sgn(len(chld)), np.log2(vec(len(chld),diffs)))
    else:
        return np.dot(sgn(len(chld)), -np.log2(vec(len(chld),diffs)))
    #^ if bottom
#^ pi_minus()

def set_to_bool_mask(n, sett):
    """
    Convert set of sets (i.e. tuple of tuples of integers) to bool mask description of set
    e.g. ((1,), (1, 2, 4)) -> np.array([True, False, False, False],[True, True, False, True]])

    Args:
        n:  Number of variables, length of returned boolean array
        sett:   Tuple of tuples of integers <= n
    """
    ret = np.full((len(sett), n), False)
    for i in range(len(sett)):
        ret[i, np.array(sett[i])-1] = True
    return ret

def bool_mask_to_set(boolmask):
    """
    Inverse to set_to_bool_mask
    """
    setofsets = ()
    for boolset in boolmask:
        sett = tuple(np.nonzero(boolset)[0]+1)
        setofsets += (sett,)
    return setofsets

@lru_cache(4)
def load_achain_dict(n):

    lattices = lt.lattices

    return lattices[n][0]
    
def convert_achain_dict(n, achain_dict):
    """
    Converts antichain-dictionary to internal representation using boolean arrays

    Args:
        n: Number of variables
        achain_dict: Antichain dictionary {achain : [children]}

    Returns:
        achainlist: list of sets (in bool_mask representation, see set_to_bool_mask())
        chldlist: list of list of children sets of corresponding antichains in achainlist
    """
    achainlist = []
    chldlist = []
    for achain in achain_dict.keys():
        achainlist.append(set_to_bool_mask(n, achain))
        chldlist.append([set_to_bool_mask(n, child) for child in achain_dict[achain]])

    return achainlist, chldlist

def compute_atoms(pdf, achain, achain_chld, rlz):
    """
    Computes the pointwise partial information atoms.

    Args:
        pdf: Probability density function
        achain: Antichain, bool array
        achain_chld: Children of antichain
        rlz: Coordinates of current realization

    Returns:
        Dictionary {alpha : pid atom}
    """
    atoms = dict()
    for alpha, alphachl in zip(achain, achain_chld):
        piplus  = pi_plus(pdf, rlz, alpha, alphachl)
        piminus = pi_minus(pdf, rlz, alpha, alphachl)
        atoms[bool_mask_to_set(alpha)] = (piplus, piminus, piplus - piminus)
    return atoms
    
def pid(pdf, achains=None, verbose=2, no_threads=1):
    """Estimate partial information decomposition for 'n' inputs and one output
                                                                                
    Implementation of the partial information decomposition (PID) estimator for
    discrete data. The estimator finds shared information, unique information   
    and synergistic information between the two, three, or four inputs with     
    respect to the output t.                                                    
                                                                                
    P.S. The implementation can be extended to any number 'n' of variables if   
    their corresponding redundancy lattice is provided ( check Lattice() )      
                                                                                
    Args:                                                                       
            pdf: Joint probability density of sources and target. The last variable will be treated as the target, i.e. function performs (pdf.nVar-1)-variable PID.  
            achain : Dictionary {achain -> [children]} with sets encoded as tuples of integers. Will attempt to load from ./sxpid/lattices.pkl if not supplied.
                     Alternatively, list [achain], children are automatically fetched from file.
            verbose: int bitmask: 1 - Print intermediate steps
                                  2 - Show progress bar (slight performance decrease from the use of imap instead of map)
                                  4 - Print result tables
            no_threads: Maximum number of parallel threads (CPU only) for calculation of PID atoms. If None, use all available threads.                                              
    Returns:                                                                    
            tuple                                                               
                ptw_dict: Pointwise partial information decomposition atoms {rlz : {achain : pid atom}}
                avg: Averaged pointwise partial information decomposition atoms {achain : averaged pid atom}
    """                 

    if type(pdf) == dict:
        if verbose & 1: print("Converting PDF dict to internal format...", end='')
        pdf = PDF.from_dict(pdf)
        if verbose & 1: print("[Done]")

    if verbose & 1: print("Loading antichain children...", end='')
    if type(achains) is dict:
        achain_dict = achains
    else:
        achain_dict = load_achain_dict(pdf.nVar-1)

        if type(achains) is list:
            achain_dict = {achain : achain_dict[achain] for achain in achains}
    
    achains, achain_chld = convert_achain_dict(pdf.nVar-1, achain_dict)
    if verbose & 1: print("[Done]")

    # Compute and store the (+, -, +-) atoms
    if verbose & 1: print("Calculating {}-variable PID atoms...".format(pdf.nVar-1, pdf.nRlz), end='\n' if verbose & 2 else '')
    
    if(no_threads > 1):
        #Multi-threaded
        pool = mp.Pool(processes=no_threads)
        if verbose & 2:
            ptw = list(tqdm(pool.imap(partial(compute_atoms, pdf, achains, achain_chld), pdf.coords), total=pdf.nRlz))
        else:
            ptw = pool.map(partial(compute_atoms, pdf, achains, achain_chld), pdf.coords)
        pool.close()
    else:
        #Single-threaded
        rlz_iter = tqdm(pdf.coords) if verbose & 2 else pdf.coords
        ptw = [None] * len(pdf.coords)
        for i, rlz in enumerate(rlz_iter):
            ptw[i] = compute_atoms(pdf, achains, achain_chld, rlz)

    if verbose & 1: print('[Done]')

    # compute and store the average of the (+, -, +-) atoms 
    avg = dict()
    if verbose & 1: print("Computing averages...", end='')
    for alpha in achains:
        alpha_set = bool_mask_to_set(alpha)
        avgplus = 0.
        avgminus = 0.
        avgdiff = 0.
        for rlz in range(pdf.nRlz):
            avgplus  += pdf.probs[rlz]*ptw[rlz][alpha_set][0]
            avgminus += pdf.probs[rlz]*ptw[rlz][alpha_set][1]
            avgdiff  += pdf.probs[rlz]*ptw[rlz][alpha_set][2]
            avg[alpha_set] = (avgplus, avgminus, avgdiff)
        #^ for
    #^ for
    if verbose & 1: print("[Done]")

    #Create ptw_dict from pointwise atom list ptw and event labels
    ptw_dict = dict(zip(pdf.get_labels(), ptw))

    # Print the result if asked
    if verbose & 4:
        table = PrettyTable()
        table.field_names = ["RLZ", "Atom", "pi+", "pi-", "pi"]
        for rlz in ptw_dict:
            count = 0
            for alpha in achains:
                alpha = bool_mask_to_set(alpha)
                stalpha = ""
                for a in alpha:
                    stalpha += "{"
                    for i in a:
                        stalpha += str(i)
                    #^ for i
                    stalpha += "}" 
                #^ for a
                if count == 0: table.add_row( [str(rlz), stalpha, str(ptw_dict[rlz][alpha][0]), str(ptw_dict[rlz][alpha][1]), str(ptw_dict[rlz][alpha][2])] )
                else:          table.add_row( [" ", stalpha, str(ptw_dict[rlz][alpha][0]), str(ptw_dict[rlz][alpha][1]), str(ptw_dict[rlz][alpha][2])] )
                count += 1 
            #^ for alpha
            table.add_row(["*", "*", "*", "*", "*"])
        #^ for realization

        table.add_row(["-", "-", "-", "-", "-"])
        count = 0
        for alpha in achains:
            alpha = bool_mask_to_set(alpha)
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
    
    return ptw_dict, avg
#^ jxpid()
