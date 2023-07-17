"""                                                                            
Shared exclusion partial information decomposition (SxPID)                      
"""

import numpy as np
from itertools import chain, combinations, product
from prettytable import PrettyTable  # type: ignore
from pkg_resources import resource_filename  # type: ignore
from tqdm import tqdm  # type: ignore
import multiprocessing as mp
from functools import partial, lru_cache, reduce
import pickle as pkl

# ---------
# Lattice
# ---------
class Lattice:
    """Generates the redundancy lattice for 'n' sources
    The algerbric structure on which partial information decomposition is
    build.
    """

    def __init__(self, n, achains):
        self.n = n
        self.lis = [i for i in range(1, self.n + 1)]
        self._achains = achains

    def powerset(self):
        return chain.from_iterable(
            combinations(self.lis, r) for r in range(1, len(self.lis) + 1)
        )

    def less_than(self, beta, alpha):
        """compare whether an antichain beta is smaller than antichain
        alpha"""
        return all(any(frozenset(b) <= frozenset(a) for b in beta) for a in alpha)

    def antichain(self):
        """Generates the nodes (antichains) of the lattice"""

        pset = list(self.powerset())[:-1]  # without empty and full set

        implications = [
            [i for (i, a) in enumerate(pset) if frozenset(s) < frozenset(a)]
            for s in pset
        ]

        parthood_dists = []
        for distribution in tqdm(
            product([False, True], repeat=2 ** self.n - 2), total=2 ** (2 ** self.n - 2)
        ):
            if all(
                all(distribution[implication] for implication in implications[index])
                for index in range(len(pset))
                if distribution[index]
            ):
                parthood_dists += [distribution]

        # Construct antichains from parthood distributions
        antichains = []
        for parthood_dist in parthood_dists:
            parthood_dist = list(parthood_dist)
            antichain = ()
            for i in range(len(parthood_dist)):
                if parthood_dist[i]:
                    antichain += (pset[i],)
                    for j in implications[i]:
                        parthood_dist[j] = False
            if len(antichain) == 0:
                antichain = (tuple(i + 1 for i in range(self.n)),)
            antichains += [antichain]

        self._achains = antichains

        return antichains

    def downset(self, alpha):
        """Computes the nodes (antichains) ordered below the node (antichain) 'alpha'"""
        return [
            beta
            for beta in self._achains
            if self.less_than(beta, alpha) and beta != alpha
        ]

    def children(self, alpha):
        """Enumerates the direct nodes (antichains) ordered by the node
        (antichain) 'alpha'"""

        assert self._achains is not None

        chl = []
        downset = self.downset(alpha)

        for beta in downset:
            if all(
                not self.less_than(beta, gamma) for gamma in downset if gamma != beta
            ):
                chl.append(beta)

        return chl

    def compute_moebius_inversion(self):
        # Using linear algebra

        assert self._achains is not None
        # Generate the forward matrix
        print("Generating forward function...")
        forward = [
            [
                1 if beta == alpha or self.less_than(beta, alpha) else 0
                for beta in self._achains
            ]
            for alpha in self._achains
        ]

        print("Inverting matrix")
        inversion = np.around(np.linalg.inv(forward)).astype(np.byte)

        print("Done!")
        return inversion


# ---------
# PDF
# ---------


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

        assert (
            coords.ndim == 2
        ), "Coordinate array must have dimensions (nRealisations, nVariables)"
        assert probs.ndim == 1, "Probability array must have dimension (nRealisations,)"

        assert (
            coords.shape[0] == probs.shape[0]
        ), "Dimensions of coordinate array (nRealisations, nVariables) and probability array (nRealisations,) do not match"

        assert np.sum(probs) - 1 < 1e-10, "Probabilities must sum up to 1"

        if not coords.data.f_contiguous:
            print(
                "Coordinate array in PDF is not Fortran contiguous, which is bad for performance!"
            )

        self.coords = coords
        self.probs = probs
        self.labels = labels

        # number of random variables
        self.nVar = self.coords.shape[1]

        # number of realizations with non-zero mass
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
        pdf_dict = {k: v for k, v in pdf_dict.items() if v > 1.0e-300}

        nKeys = len(pdf_dict.keys())
        nVar = len(next(iter(pdf_dict.keys())))

        keys, vals = zip(*pdf_dict.items())

        events = np.empty((nKeys, nVar), dtype=object)
        events[:] = keys

        coords = np.empty((nKeys, nVar), order="F", dtype=np.uint64)
        labels = []
        for i in range(nVar):
            labels_column, inverse_column = np.unique(events[:, i], return_inverse=True)
            coords[:, i] = inverse_column
            labels.append(labels_column)

        # Use smallest integer datatype possible for coordinates.
        # Improves memory consumption and execution speeds.
        alphabet_sizes = np.array(list(map(len, labels)))
        max_var = np.max(alphabet_sizes)

        uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]
        uint_max = np.array([np.iinfo(t).max for t in uint_types], dtype=np.uint64)
        dtype = uint_types[np.argmax(max_var <= uint_max)]

        coords = coords.astype(dtype)

        probs = np.array(vals)

        pdf = PDF(coords, probs, labels)

        return pdf

    def __getitem__(self, rlz):
        """Returns the probability of a given realization

        Args:
            rlz ([type]): [description]
        """
        idx = np.argmax(np.all(self.coords == rlz, axis=1))
        return self.probs[idx]

    def get_labels(self):
        """
        Returns list of length nRlz with the original event labels, reconstructed from labels. If no labels are given, returns list of tuples of coordinates.

        Returns:
            List of tuples
        """

        labels_columns = ()
        for i in range(self.nVar):
            coord_column = self.coords[:, i]
            labels_columns += (
                list(self.labels[i][coord_column] if self.labels else coord_column),
            )
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
        sum_mask = np.all(~union_masks[0] | rlz_mask, axis=-1)
        for set_mask in union_masks[1:]:
            sum_mask = sum_mask | np.all(~set_mask | rlz_mask, axis=-1)
        summ = np.sum(self.probs, where=sum_mask)
        return summ

    def marginalize(self, *coords):
        """
        Args:
            coord: Coordinate to marginalize out.
        """
        coords = [
            coord if coord >= 0 else self.coords.shape[1] + coord for coord in coords
        ]

        coords_reduced = np.delete(self.coords, obj=coords, axis=1)

        unique, index, inverse = np.unique(
            coords_reduced, return_index=True, return_inverse=True, axis=0
        )

        newprobs = np.bincount(inverse, weights=self.probs)

        return PDF(np.asfortranarray(unique), newprobs)


def get_bool_mask(n, sett, condition_on_target):
    """
    Convert set of sets (i.e. tuple of tuples of integers) to bool mask description of set
    e.g. ((1,), (1, 2, 4)) -> np.array([True, False, False, False],[True, True, False, True]])
    Appends boolean for target variable that dicides whether target is conditioned on

    Args:
        n:  Number of variables, length of returned boolean array
        sett:   Tuple of tuples of integers <= n
        condition_on_target: bool, set last element of mask to this value
    """
    ret = np.full((len(sett), n + 1), False)
    for i in range(len(sett)):
        ret[i, np.array(sett[i]) - 1] = True
        ret[i, -1] = condition_on_target
    return ret


@lru_cache
def load_achains(n):

    lattice_file_name = resource_filename(__name__, "moebius.pkl")

    with open(lattice_file_name, "rb") as lattice_file:
        lattices = pkl.load(lattice_file)

    return lattices[n][0]


@lru_cache
def load_moebius_function(n):

    moebius_file_name = resource_filename(__name__, "moebius.pkl")

    with open(moebius_file_name, "rb") as moebius_file:
        moebius_data = pkl.load(moebius_file)

    return np.array(moebius_data[n][1])


def compute_i_cap_plus(pdf, achains_t, rlz):
    """Computes the local redundancies i_\cap for the given realization

    Args:
        pdf ([type]): [description]
        achains ([type]): [description]
        rlz ([type]): [description]
    """

    # Informative part
    i_cap_plus = -np.log2(
        np.array([pdf.probu_one_rlz(rlz, alphat) for alphat in achains_t])
    )

    return i_cap_plus


@lru_cache
def get_full(n):
    full = np.full((1, n), False)
    full[:, -1] = True
    return full


def compute_i_cap_minus(pdf, achains_t, rlz):
    """Computes the local redundancies i_\cap for the given realization

    Args:
        pdf ([type]): [description]
        achains ([type]): [description]
        part ([type]): "pos" or "neg"
        rlz ([type]): [description]
    """

    full = get_full(pdf.nVar)

    # Misinformative part
    i_cap_minus = np.log2(
        pdf.probu_one_rlz(rlz, full)
        / np.array([pdf.probu_one_rlz(rlz, alphat) for alphat in achains_t])
    )

    return i_cap_minus


def pid(pdf, achains=None, verbose=2, n_threads=1, parts="all", pointwise=True, showProgress=False):
    """Estimate partial information decomposition for 'n' inputs and one output

    Implementation of the partial information decomposition (PID) estimator for
    discrete data. The estimator finds shared information, unique information
    and synergistic information between the two, three, or four inputs with
    respect to the output t.

    P.S. The implementation can be extended to any number 'n' of variables if
    their corresponding redundancy lattice is provided ( check Lattice() )

    Args:
        pdf (PDF | dict[tuple[Any, ...], float]): Joint probability density of sources and target. The last variable will be treated as the target, i.e. function performs (pdf.nVar-1)-variable PID.
        achains (dict[tuple: list[tuples]}, optional): Dictionary {achain: [children]} with sets encoded as tuples of integers. Will attempt to load from ./sxpid/lattices.pkl if not supplied. Defaults to None.
        verbose (int, optional): bitmask:   1 - Print intermediate steps
                                            2 - Show progress bar (slight performance decrease from the use of imap instead of map)
                                            4 - Print result tables.
                                Defaults to 2.
        n_threads (int or 'all', optional): Maximum number of parallel threads (CPU only) for calculation of PID atoms. If None, use all available threads. Defaults to 1.
        parts (str, optional):  'all' - informative and misinformative part
                                'inf' - informative part only
                                'mis' - misinformative part only.
                                Defaults to "all".
        pointwise (bool, optional): Return pointwise decomposition. Disabling improves single-thread memory consumption. Defaults to True.

    Returns:
        tuple:
            ptw_dict: Pointwise partial information decomposition atoms {rlz : {achain : pid atom}} if pointwise is True, else None
            avg: Averaged pointwise partial information decomposition atoms {achain : averaged pid atom}
    """

    assert (
        pointwise or not verbose & 4
    ), "pointwise must be true to print result tables!"

    if type(pdf) == dict:
        if verbose & 1:
            print("Converting PDF dict to internal format...", end="")
        pdf = PDF.from_dict(pdf)
        if verbose & 1:
            print("[Done]")

    n = pdf.nVar - 1

    if verbose & 1:
        print("Loading antichains and moebius function...", end="")

    if type(achains) is not dict:
        achains = load_achains(n)

    moebius_func = load_moebius_function(n)

    if verbose & 1:
        print("[Done]")

    # Compute and store the (+, -, +-) atoms
    if verbose & 1:
        print(
            f"Calculating {n}-variable PID atoms...",
            end="\n" if verbose & 2 else "",
        )

    # Compute the redundancies

    if n_threads == "all":
        n_threads = mp.cpu_count()
    if verbose & 1:
        print(f"Computing on {n_threads} processes.")

    elif n_threads > 1:
        pool = mp.Pool(processes=n_threads)

    imap = pool.imap if n_threads > 1 else map

    if showProgress == "tqdm":
        mapper = lambda f, iter: tqdm(imap(f, iter), total=pdf.nRlz)
    elif showProgress == "print":

        def print_progress(iter, total):
            steps = max(total // 20, 1)
            for i, x in enumerate(iter):
                if i % steps == 0:
                    print(f"{i+1}/{total}")
                yield x

        mapper = lambda f, iter: print_progress(imap(f, iter), pdf.nRlz)
    else:
        mapper = imap

    achain_masks_uncond = [get_bool_mask(n, alpha, False) for alpha in achains]
    achain_masks_cond = [get_bool_mask(n, alpha, True) for alpha in achains]

    if parts == "inf" or parts == "all":
        i_cap_plus = mapper(partial(compute_i_cap_plus, pdf, achain_masks_uncond), pdf.coords)

    if parts == "mis" or parts == "all":
        i_cap_minus = mapper(partial(compute_i_cap_minus, pdf, achain_masks_cond), pdf.coords)

    if verbose & 1:
        print("[Done]")
        print("Computing Moebius inverse...", end="")

    if n_threads > 1:
        pool.close()

    # Compute the Moebius inversion

    if pointwise == True:
        # Collect all pointwise values
        if parts == "inf" or parts == "all":
            i_cap_plus = np.array(list(i_cap_plus))

        if parts == "mis" or parts == "all":
            i_cap_minus = np.array(list(i_cap_minus))

        # Compute Moebius inverse of pointwise
        if parts == "inf" or parts == "all":
            pi_plus = (moebius_func @ i_cap_plus.T).T
        else:
            pi_plus = np.full((pdf.nRlz, len(achains)), np.nan)

        if parts == "mis" or parts == "all":
            pi_minus = (moebius_func @ i_cap_minus.T).T
        else:
            pi_minus = np.full((pdf.nRlz, len(achains)), np.nan)

        pi = pi_plus - pi_minus

        # Convert pointwise values to dictionary format
        ptw = [
            {
                achain: pies
                for (achain, pies) in zip(
                    achains, zip(pi_plus[i], pi_minus[i], pi[i])
                )
            }
            for i in range(pdf.nRlz)
        ]
        ptw_dict = dict(zip(pdf.get_labels(), ptw))

        # Compute average quantities
        Pi_plus = pi_plus.T @ pdf.probs
        Pi_minus = pi_minus.T @ pdf.probs

        Pi = Pi_plus - Pi_minus
    
    elif pointwise == "target":
        # Average over everything except the target variable
        targets = pdf.coords[:, -1]
        unique_targets, unique_target_indices = np.unique(targets, return_index=True)
        unique_target_labels = [pdf.get_labels()[unique_target_index][-1] for unique_target_index in unique_target_indices]
        n_targets = len(unique_targets)

        unique_target_probs = np.array([pdf.probu_one_rlz(tuple(0 for _ in range(n)) + (target,), 
                                                 union_masks=[np.array([False for _ in range(n)] + [True])])
                                                 for target in unique_targets])

        def reduce_targetlocal(cumsum, prob, icap, target):
            cumsum[target] += prob * icap
            return cumsum

        if parts == "inf" or parts == "all":
            i_cap_plus_target = reduce(
                lambda cumsum, arg: reduce_targetlocal(cumsum, *arg),
                zip(pdf.probs, i_cap_plus, targets),
                np.zeros((n_targets, len(achains))),
            )
            i_cap_plus_target = i_cap_plus_target / unique_target_probs[:, None]


        if parts == "mis" or parts == "all":
            i_cap_minus_target = reduce(
                lambda cumsum, arg: reduce_targetlocal(cumsum, *arg),
                zip(pdf.probs, i_cap_minus, targets),
                np.zeros((n_targets, len(achains))),
            )
            i_cap_minus_target = i_cap_minus_target / unique_target_probs[:, None]

        # Compute Moebius inverse
        if parts == "inf" or parts == "all":
            pi_plus_target = (moebius_func @ i_cap_plus_target.T).T
        else:
            pi_plus_target = np.full((len(achains), n_targets), np.nan)

        if parts == "mis" or parts == "all":
            pi_minus_target = (moebius_func @ i_cap_minus_target.T).T
        else:
            pi_minus_target = np.full((len(achains), n_targets), np.nan)

        pi_target = pi_plus_target - pi_minus_target

        # Convert pointwise values to pointwise dictionary format
        ptw_target = [
            {
                achain: pies
                for (achain, pies) in zip(
                    achains, zip(pi_plus_target[i], pi_minus_target[i], pi_target[i])
                )
            }
            for i in range(n_targets)
        ]
        ptw_dict = dict(zip(unique_target_labels, ptw_target))

        # Compute average quantities
        Pi_plus = pi_plus_target.T @ unique_target_probs
        Pi_minus = pi_minus_target.T @ unique_target_probs

        Pi = Pi_plus - Pi_minus

    else:
        # Accelerate computation and reduce memory footprint by first computing average redundancies
        # and then computing Moebius inverse

        # Average i_cap

        if parts == "inf" or parts == "all":
            I_cap_plus = reduce(
                lambda cumsum, arg: cumsum + arg[0] * arg[1],
                zip(i_cap_plus, pdf.probs),
                np.zeros(len(achains)),
            )

        if parts == "mis" or parts == "all":
            I_cap_minus = reduce(
                lambda cumsum, arg: cumsum + arg[0] * arg[1],
                zip(i_cap_minus, pdf.probs),
                np.zeros(len(achains)),
            )

        # Compute Moebius inverse
        if parts == "inf" or parts == "all":
            Pi_plus = moebius_func @ I_cap_plus
        else:
            Pi_plus = np.full(len(achains), np.nan)

        if parts == "mis" or parts == "all":
            Pi_minus = moebius_func @ I_cap_minus
        else:
            Pi_minus = np.full(len(achains), np.nan)

        Pi = Pi_plus - Pi_minus

    # Convert average PID values to dictionary format
    avg = {achain: pies for (achain, pies) in zip(achains, zip(Pi_plus, Pi_minus, Pi))}

    if verbose & 1:
        print("[Done]")

    # Print the result if asked
    if verbose & 4:
        table = PrettyTable()
        table.field_names = ["RLZ", "Atom", "pi+", "pi-", "pi"]
        for rlz in ptw_dict:
            count = 0
            for alpha in achains:
                stalpha = ""
                for a in alpha:
                    stalpha += "{"
                    for i in a:
                        stalpha += str(i)

                    stalpha += "}"

                if count == 0:
                    table.add_row(
                        [
                            str(rlz),
                            stalpha,
                            str(ptw_dict[rlz][alpha][0]),
                            str(ptw_dict[rlz][alpha][1]),
                            str(ptw_dict[rlz][alpha][2]),
                        ]
                    )
                else:
                    table.add_row(
                        [
                            " ",
                            stalpha,
                            str(ptw_dict[rlz][alpha][0]),
                            str(ptw_dict[rlz][alpha][1]),
                            str(ptw_dict[rlz][alpha][2]),
                        ]
                    )
                count += 1

            table.add_row(["*", "*", "*", "*", "*"])

        table.add_row(["-", "-", "-", "-", "-"])
        count = 0
        for alpha in achains:
            stalpha = ""
            for a in alpha:
                stalpha += "{"
                for i in a:
                    stalpha += str(i)

                stalpha += "}"

            if count == 0:
                table.add_row(
                    [
                        "avg",
                        stalpha,
                        str(avg[alpha][0]),
                        str(avg[alpha][1]),
                        str(avg[alpha][2]),
                    ]
                )
            else:
                table.add_row(
                    [
                        " ",
                        stalpha,
                        str(avg[alpha][0]),
                        str(avg[alpha][1]),
                        str(avg[alpha][2]),
                    ]
                )
            count += 1

        print(table)

    return (ptw_dict, avg) if pointwise else avg
