from sys import path
path.insert(0, "../../../BROJA_2PID")
import BROJA_2PID
path.insert(0, "../sxpid")
import SxPID
import pickle 

def q_vidx(i): return 3*i+2

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

# PwUnq
pwunqgate = dict()
pwunqgate[(0,1,1)] = 0.25
pwunqgate[(1,0,1)] = 0.25
pwunqgate[(0,2,2)] = 0.25
pwunqgate[(2,0,2)] = 0.25

# Rnd
rndgate = dict()
rndgate[(0,0,0)] = 0.5
rndgate[(1,1,1)] = 0.5

# RndErr
rnderrgate = dict()
rnderrgate[(0,0,0)] = 3/8
rnderrgate[(1,1,1)] = 3/8
rnderrgate[(0,1,0)] = 1/8
rnderrgate[(1,0,1)] = 1/8

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

def sxpid_to_broja2pid_gate(gate):
    gate_broja2pid = dict()
    for k,v in gate.items():
        gate_broja2pid[(k[2], k[0], k[1])] = v
    #^ for
    return gate_broja2pid
#^ to_broja2pid_gate()

def compute_broja(gate):
    # ECOS parameters 
    parms = dict()
    parms['max_iters'] = 100
    parms['keep_solver_object'] = True

    broja2pid_gate = sxpid_to_broja2pid_gate(gate)
    returndata = BROJA_2PID.pid(broja2pid_gate, cone_solver="ECOS", output=2, **parms)

    msg="""Shared information: {SI}
    Unique information in Y: {UIY}
    Unique information in Z: {UIZ}
    Synergistic information: {CI}
    Primal feasibility: {Num_err[0]}
    Dual feasibility: {Num_err[1]}
    Duality Gap: {Num_err[2]}"""
    print(msg.format(**returndata))

    # get the optimal distribution
    solver = returndata["Solver Object"]
    pdf_opt = dict()
    for x in solver.X:
        for y in solver.Y:
            if (x,y) in solver.b_xy.keys():
                for z in solver.Z:
                    if (x,z) in solver.b_xz.keys():
                        i = solver.idx_of_trip[ (x,y,z) ]
                        pdf_opt[ (x,y,z) ] = float(solver.sol_rpq[q_vidx(i)])
                    #^ if
                #^ for z
            #^ if 
        #^ for y
    #^ for x
    print("The End")
    return pdf_opt
#^ compute_broja()


def broja2pid_to_sxpid_gate(gate):
    gate_sxpid = dict()
    for k,v in gate.items():
        gate_sxpid[(k[1], k[2], k[0])] = v
    #^ for
    return gate_sxpid
#^ to_broja2pid_gate()

def compute_sxpid(n, gate, gate_opt, lattices):
    ptw, avg = SxPID.pid(n, gate, lattices[n][0], lattices[n][1], True)
    print("The optimal BROJA distribution")
    gate_opt_sx = broja2pid_to_sxpid_gate(gate_opt)
    pdf = {k:v for k,v in gate_opt_sx.items() if v > 1.e-8 }
    print("Note that if p(s1,s2,t) < 1.e-8 of the optimal BROJA distribution then it is considered impossible")
    for k,v in pdf.items():
        print(k, " : ", v)
    #^ for
    ptw_opt, avg_opt = SxPID.pid(n, pdf, lattices[n][0], lattices[n][1], True)

    return ptw, avg, ptw_opt, avg_opt
#^ compute_sxpid()


#--------
# Run it!
#--------
# Read the lattices:
# pickled as { n -> [{alpha -> children}, (alpha_1,...) ] }
f = open("../sxpid/lattices.pkl", "rb")
lattices = pickle.load(f)
n = 2

for gate in gates.keys():
    print("*********************************")
    print("The BROJA_2PID for the", gate, ":")
    print("*********************************")
    gate_opt = compute_broja(gates[gate])
    print("**************************************************************")
    print("The JxPID for the", gate, "and its optimal BROJA distribution:")
    print("**************************************************************")
    ptw, avg, ptw_opt, avg_opt = compute_sxpid(n, gates[gate], gate_opt, lattices)
#^ for gate
