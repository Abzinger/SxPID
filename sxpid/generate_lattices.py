# generate_lattices.py


import sxpid.SxPID
import pickle as pkl
from tqdm import tqdm


def generate_lattice():
    """
    Pickles this object { n -> {alpha -> children}}
    into lattices.pkl
    """
    lattices = dict()
    for n in range(2, 6):
        print(n)
        lattice = sxpid.SxPID.Lattice(n, None)
        achain = lattice.antichain()
        chld = dict()
        for alpha in tqdm(achain):
            chld[alpha] = lattice.children(alpha)
        # ^ for

        lattices[n] = chld
    # ^ for
    f = open("sxpid/lattices.pkl", "wb")
    pkl.dump(lattices, f)
    f.close()


# ^ generate_lattice()


def compute_moebius_inversion():

    moebius = dict()
    for n in range(2, 6):
        print("n=", n)
        f = open("sxpid/lattices.pkl", "rb")
        achains = list(pkl.load(f)[n].keys())
        f.close()

        lattice = sxpid.SxPID.Lattice(n, achains)

        inverse = lattice.compute_moebius_inversion()

        moebius[n] = [achains, inverse]

    with open("sxpid/moebius.pkl", "wb") as moebius_file:
        pkl.dump(moebius, moebius_file)


if __name__ == "__main__":
    generate_lattice()
    compute_moebius_inversion()