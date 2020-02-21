# test_dice.py

from sys import path
path.insert(0, "../../../BROJA_2PID")
import BROJA_2PID
path.insert(0,"../sxpid")
import SxPID
import time
import pickle
import matplotlib.pyplot as plt
import dit


# The dice distribution:
# T = S_1 + alpha*S_2
# P(S_1 = i , S_2 = j) = lambda/36 + (1-lambda)delta_{i,j}/6
def generate_dice_pdfs(num_a, num_ell):
    pdfs = dict()
    for a in range(1, num_a + 1): 
        for lam in range(num_ell + 1):
            ell = lam/num_ell
            pdf = dict()
            for i in range(6):
                for j in range(6):
                    t = i + a*j
                    if i == j: pdf[( i, j, t )] = ell/36 + (1-ell)/6
                    else: pdf[( i, j, t)] = ell/36
                #^ for j
            #^ for i
            pdfs[(a,ell)] = pdf
        #^ for lam
    #^ for a

    return pdfs
#^ generate_dice_pdfs()


# The dice type 2 distribution:
# T = S_1 + 6*S_2 mod alpha
# P(S_1 = i , S_2 = j) = lambda/36 + (1-lambda)delta_{i,j}/6
def generate_dice_type_2_pdfs(num_a, num_ell):
    pdfs = dict()
    for a in range(1, num_a + 1):           
        for lam in range(num_ell + 1):
            ell = lam/num_ell
            pdf = dict()
            for i in range(6):
                for j in range(6):
                    t = (i + a*j) % a
                    if i == j: pdf[( i, j, t )] = ell/36 + (1-ell)/6
                    else: pdf[( i, j, t)] = ell/36
                #^ for j
            #^ for i
            pdfs[(a,ell)] = pdf
        #^ for lam
    #^ for a
    return pdfs
#^ generate_dice_type_2_pdfs()

# Compute SxPID
def compute_sxpid(pdfs, lattices):
    shared   = dict()
    synergy  = dict()
    unique_1 = dict()
    unique_2 = dict()
    n = 2
    for k in pdfs.keys():
        # # Print the distribution in a nice format
        # pts = []
        # values = []
        # for pt, v in pdfs[k].items():
        #     pts.append(pt)
        #     values.append(v)
        # #^ for
        # dpdf = dit.Distribution(pts, values, base='linear', validate=False)
        # print(dpdf)
        # Compute SxPID
        ptw, avg = SxPID.pid(n, pdfs[k], lattices[n][0], lattices[n][1], False)
        shared[k]   = avg[((1,),(2,),)][2]
        synergy[k]  = avg[((1,2,),)][2]
        unique_1[k] = avg[((1,),)][2]
        unique_2[k] = avg[((2,),)][2]
        # print("_sxpid", shared[k] + synergy[k] + unique_1[k] + unique_2[k])
    #^ for pdfs
    return shared, synergy, unique_1, unique_2
#^ compute_sxpid()

# Broja format of pdf 
def sxpid_to_broja2pid_pdf(pdf):
    pdf_broja2pid = dict()
    for k,v in pdf.items():
        pdf_broja2pid[(k[2], k[0], k[1])] = v
    #^ for
    return pdf_broja2pid
#^ to_broja2pid_gate()

# Compute BROJA PID
def compute_broja(pdfs):
    # ECOS parameters 
    parms = dict()
    parms['abstol'] = 1.e-15
    parms['feastol'] = 1.e-15
    #parms['keep_solver_object'] = True
    shared = dict()
    synergy  = dict()
    unique_1 = dict()
    unique_2 = dict()
    for k in pdfs.keys():
        broja2pid_pdf = sxpid_to_broja2pid_pdf(pdfs[k])
        returndata = BROJA_2PID.pid(broja2pid_pdf, cone_solver="ECOS", output=0, **parms)

        shared[k]   = returndata['SI']
        synergy[k]  = returndata['CI']
        unique_1[k] = returndata['UIY']
        unique_2[k] = returndata['UIZ']

        # msg="""Shared information: {SI}
        # Unique information in Y: {UIY}
        # Unique information in Z: {UIZ}
        # Synergistic information: {CI}
        # Primal feasibility: {Num_err[0]}
        # Dual feasibility: {Num_err[1]}
        # Duality Gap: {Num_err[2]}"""
        # print(msg.format(**returndata))
        # print("_broja", shared[k] + synergy[k] + unique_1[k] + unique_2[k])
    #^ for pdfs

    return shared, synergy, unique_1, unique_2
#^ compute_broja()

def compute_fl(pdfs, shared_br, synergy_br, unique_1_br, unique_2_br):
    shared = dict()
    synergy  = dict()
    unique_1 = dict()
    unique_2 = dict()
    for k in pdfs.keys():
        pts = []
        values = []
        for pt, v in pdfs[k].items():
            pts.append(pt)
            values.append(v)
        #^ for
        dpdf = dit.Distribution(pts, values, base='linear', validate=False)
        shared[k] = dit.pid.ipm.PID_PM._measure(dpdf, ((0,), (1,)), (2,))
        unique_1[k] = shared_br[k] + unique_1_br[k] - shared[k] 
        unique_2[k] = shared_br[k] + unique_2_br[k] - shared[k]
        synergy[k]  = shared_br[k] + synergy_br[k] + unique_1_br[k] + unique_2_br[k] - shared[k] - unique_1[k] - unique_2[k]
        # print("_red", shared[k] + synergy[k] + unique_1[k] + unique_2[k])
        print("a", k[0])
        print(dpdf)
    #^ for
    return shared, synergy, unique_1, unique_2
#^ compute_fl()

# def compute_red(pdfs, shared_br, synergy_br, unique_1_br, unique_2_br):
#     shared = dict()
#     synergy  = dict()
#     unique_1 = dict()
#     unique_2 = dict()
#     for k in pdfs.keys():
#         pts = []
#         values = []
#         for pt, v in pdfs[k].items():
#             pts.append(pt)
#             values.append(v)
#         #^ for
#         dpdf = dit.Distribution(pts, values, base='linear', validate=False)
#         shared[k] = dit.pid.iproj.PID_Proj._measure(dpdf, ((0,), (1,)), (2,))
#         unique_1[k] = shared_br[k] + unique_1_br[k] - shared[k] 
#         unique_2[k] = shared_br[k] + unique_2_br[k] - shared[k]
#         synergy[k]  = shared_br[k] + synergy_br[k] + unique_1_br[k] + unique_2_br[k] - shared[k] - unique_1[k] - unique_2[k]
#         # print("_red", shared[k] + synergy[k] + unique_1[k] + unique_2[k])
#         print(dpdf)
#     #^ for
#     return shared, synergy, unique_1, unique_2
# #^ compute_red()


def generate_pid_figure(num_a, num_lam, _sxpid, _broja, _fl, pid_term):
    plt.figure()
    plt.suptitle(pid_term)
    for a in range(1, num_a + 1):
        x = []
        y_sxpid = []
        y_broja = []
        y_fl = []
        for lam in range(num_lam + 1):
            x.append(lam/num_lam)
            y_sxpid.append(_sxpid[(a,lam/num_lam)])
            y_broja.append(_broja[(a,lam/num_lam)])
            y_fl.append(_fl[(a,lam/num_lam)])
        #^ for lam
        plt.subplot(221)
        plt.plot(x,y_sxpid,label=r'$\alpha$ = '+str(a))
        if a == num_a:
            plt.xlabel('$\lambda$')
            plt.ylabel('bits')
            plt.legend()
            plt.title('SxPID')
        #^ if labels

        plt.subplot(222)
        plt.plot(x,y_fl,label=r'$\alpha$ = '+str(a))
        if a == num_a:
            plt.xlabel('$\lambda$')
            plt.ylabel('bits')
            plt.legend()
            plt.title('Finn-Lizier')
        #^ if labels

        plt.subplot(223)
        plt.plot(x,y_broja,label=r'$\alpha$ = '+str(a))
        if a == num_a:
            plt.xlabel('$\lambda$')
            plt.ylabel('bits')
            plt.legend()
            plt.title('BROJA')
        #^ if labels
    #^ for a
    return 0 
#^ generate_pid_figure()


def generate_mi_figure(num_a, num_lam, _shared, _synergy, _unique_1, _unique_2):
    plt.figure()
    plt.suptitle("Mutual Information")
    for a in range(1, num_a + 1):
        x = []
        y_mi_1 = []
        y_mi_2 = []
        y_cmi_1 = []
        y_cmi_2 = []
        y_mi_1_2 = []
        y_mi_1_2_3 = []
        for lam in range(num_lam + 1):
            x.append(lam/num_lam)
            y_mi_1.append(_shared[(a,lam/num_lam)] + _unique_1[(a,lam/num_lam)])
            y_mi_2.append(_shared[(a,lam/num_lam)] + _unique_2[(a,lam/num_lam)])
            y_cmi_1.append(_synergy[(a,lam/num_lam)] + _unique_1[(a,lam/num_lam)])
            y_cmi_2.append(_synergy[(a,lam/num_lam)] + _unique_2[(a,lam/num_lam)])
            y_mi_1_2.append(_shared[(a,lam/num_lam)] + _unique_1[(a,lam/num_lam)] + _synergy[(a,lam/num_lam)] + _unique_2[(a,lam/num_lam)])
            y_mi_1_2_3.append(_shared[(a,lam/num_lam)] - _synergy[(a,lam/num_lam)])
        #^ for lam
        plt.subplot(321)
        plt.plot(x,y_mi_1,label=r'$\alpha$ = '+str(a))
        if a == num_a:
            plt.xlabel('$\lambda$')
            plt.ylabel('bits')
            plt.legend()
            plt.title(r'$I(T:S^{(1)})$')
        #^ if labels

        plt.subplot(322)
        plt.plot(x,y_mi_2,label=r'$\alpha$ = '+str(a))
        if a == num_a:
            plt.xlabel('$\lambda$')
            plt.ylabel('bits')
            plt.legend()
            plt.title('$I(T:S^{(2)})$')
        #^ if labels

        plt.subplot(323)
        plt.plot(x,y_cmi_1,label=r'$\alpha$ = '+str(a))
        if a == num_a:
            plt.xlabel('$\lambda$')
            plt.ylabel('bits')
            plt.legend()
            plt.title('$I(T:S^{(1)}\mid S^{(2)})$')
        #^ if labels
        plt.subplot(324)
        plt.plot(x,y_cmi_2,label=r'$\alpha$ = '+str(a))
        if a == num_a:
            plt.xlabel('$\lambda$')
            plt.ylabel('bits')
            plt.legend()
            plt.title('$I(T:S^{(2)}\mid S^{(1)})$')
        #^ if labels
        plt.subplot(325)
        plt.plot(x,y_mi_1_2,label=r'$\alpha$ = '+str(a))
        if a == num_a:
            plt.xlabel('$\lambda$')
            plt.ylabel('bits')
            plt.legend()
            plt.title('$I(T:S^{(2)},S^{(1)})$')
        #^ if labels
        plt.subplot(326)
        plt.plot(x,y_mi_1_2_3,label=r'$\alpha$ = '+str(a))
        if a == num_a:
            plt.xlabel('$\lambda$')
            plt.ylabel('bits')
            plt.legend()
            plt.title('$I(T:S^{(2)}:S^{(1)})$')
        #^ if labels
    #^ for a
    return 0 
#^ generate_figure()

# Check the cst hypothesis:
def compute_ptw(pdfs, lattices):
    ptw_k   = dict()
    n = 2
    for k in pdfs.keys():
        ptw, avg = SxPID.pid(n, pdfs[k], lattices[n][0], lattices[n][1], False)
        ptw_k[k]   = ptw
    #^ for pdfs
    return ptw_k
#^ compute_ptw()

def check_cst_ptw(num_a, num_lam, ptw_k, frac=False):
    for lam in range(num_lam + 1):
        for a in range(1, num_a + 1):
            for rlz in ptw_k[(a,lam/num_lam)].keys():
                assert abs(ptw_k[(a,lam/num_lam)][rlz][((1,),)][1]) < 1.e-8, "mis-unique S1 fail "
                assert abs(ptw_k[(a,lam/num_lam)][rlz][((2,),)][1]) < 1.e-8, "mis-unique S2 fail "
                assert abs(ptw_k[(a,lam/num_lam)][rlz][((1,2,),)][1]) < 1.e-8, "mis-synergy fail " 
                for b in range(1, num_a + 1):
                    if rlz in ptw_k[(b,lam/num_lam)].keys():
                        assert abs(ptw_k[(a,lam/num_lam)][rlz][((1,),)][0] - ptw_k[(b,lam/num_lam)][rlz][((1,),)][0]) < 1.e-8, "unique S1 fail "
                        assert abs(ptw_k[(a,lam/num_lam)][rlz][((2,),)][0] - ptw_k[(b,lam/num_lam)][rlz][((2,),)][0]) < 1.e-8, "unique S2 fail "
                        assert abs(ptw_k[(a,lam/num_lam)][rlz][((1,2),)][0] - ptw_k[(b,lam/num_lam)][rlz][((1,2),)][0]) < 1.e-8, "synergy fail "
                    #^ if 
                #^ for b
            #^ for rlz
        #^ for a
    #^ for lam
    return 0
#^ check_cst_ptw

def main():
    # Read lattices from a file
    # Pickled as { n -> [{alpha -> children }, (alpha_1, ...)] }
    f = open("../sxpid/lattices.pkl", "rb")
    lattices = pickle.load(f)
    num_a = 6
    num_lam = 100
    # pdfs = generate_dice_pdfs(num_a, num_lam)
    pdfs = generate_dice_type_2_pdfs(num_a, num_lam)
    shared_sxpid, synergy_sxpid, unique_1_sxpid, unique_2_sxpid = compute_sxpid(pdfs,lattices)
    shared_broja, synergy_broja, unique_1_broja, unique_2_broja = compute_broja(pdfs)
    shared_fl, synergy_fl, unique_1_fl, unique_2_fl             = compute_fl(pdfs, shared_broja, synergy_broja, unique_1_broja, unique_2_broja)
    generate_pid_figure(num_a, num_lam, shared_sxpid, shared_broja, shared_fl, 'Shared Information')
    generate_pid_figure(num_a, num_lam, synergy_sxpid, synergy_broja, synergy_fl, 'Synergistic Information')
    generate_pid_figure(num_a, num_lam, unique_1_sxpid, unique_1_broja, unique_1_fl, 'Unique D1 Information')
    generate_pid_figure(num_a, num_lam, unique_2_sxpid, unique_2_broja, unique_2_fl, 'Unique D2 Information')
    generate_mi_figure(num_a, num_lam, shared_broja, synergy_broja, unique_1_broja, unique_2_broja)
    plt.show()
    return 0
#^ main()

#-------
# Run it
#-------

main()
# f = open("../sxpid/lattices.pkl", "rb")
# lattices = pickle.load(f)
# pdfs = generate_dice_pdfs(6,100)
# ptw_k = compute_ptw(pdfs, lattices)
# check_cst_ptw(6, 100, ptw_k)
