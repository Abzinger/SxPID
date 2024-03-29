# demo_randompdf_tri.py

from sxpid import SxPID
import numpy as np
import time
from random import random
from sys import argv
import pickle

# test_randompdf_tri.py -- part of SxPID (https://github.com/Abzinger/SxPID/)
# Usage: python3 test_randompdf_tri.py t x y z [iter]
# Where:   t    is the size of the range of T;
#          x    is the size of the range of X;
#          y    is the size of the range of Y;
#          z    is the size of the range of Z;
#          iter is the number of pdfs
#                   (defaults to 250).

def compute_pid(nT, nX, nY, nZ, maxiter):

    # Lists to store Time and Error for boxplotting
    Ti = []

    # List to store detected negative pi_alpha
    Npid = []

    # Files to save Ti, Npid for boxplotting 
    time_file  = open("randompdfs_time.pkl", 'ab')
    Npid_file  = open("randompdfs_negative_pid.pkl", 'ab')
    
    # Main Loop 
    tic = time.time()
    for iter in range(maxiter):

        # Sample the Probability Distribution        
        print("Random PDFs   with |T| =",nT,"|X| =",nX,"|Y| =",nY," |Z| =",nZ)
        print("______________________________________________________________________")
        print("Create pdf #",iter)
        pdf_dict = dict()
        pts = [ random() for j in range(1,nT*nX*nY*nZ) ]
        pts.append(0.)
        pts.sort()
        val = 1.
        for t in range(nT):
            for x in range(nX):
                for y in range(nY):
                    for z in range(nZ):
                        newval = pts.pop()
                        pdf_dict[ (t,x,y,z) ] = val - newval
                        val = newval
                    #^ for z
                #^ for y
            #^ for x
        #^ for t

        # Compute PID
        pdf = SxPID.PDF.from_dict(pdf_dict)

        print("Run SxPID.pid().")
        itic = time.time()
        ptw,avg = SxPID.pid(pdf, n_threads="all")       
        itoc = time.time()

        # Print PID details
        print("_______________________________________")
        print("Time: ",itoc-itic,"secs")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # Store Time for boxplotting
        Ti.append(itoc - itic)
        
        # Store the index of PID if there is a negative pi_alpha
        pdf_clean = {k:v for k,v in pdf_dict.items() if v > 1.e-300 }
        for rlz in pdf_clean:
            for alpha in ptw[rlz]:
                if ptw[rlz][alpha][0] <= -1.e-10 or ptw[rlz][alpha][1] <= -1.e-10:
                    print("Ops! Found a negative inf- or misinf- atom")
                    Npid.append(iter)
                #^ if neg
            #^ for alpha
        #^ for rlz
    #^ for iter
    toc = time.time()
    # Check Average Time
    print("**********************************************************************")
    print("Average time: ",(toc-tic)/maxiter,"secs")

    # Store into times and error file to create boxplots later
    pickle.dump(Ti, time_file)
    pickle.dump(Npid, Npid_file)
    time_file.close()
    Npid_file.close()

    return Npid


def Main(sys_argv):
    print("\ntest_randompdf_tri.py -- part of SxPID (https://github.com/Abzinger/SxPID/)\n")
    if len(sys_argv) < 5 or len(sys_argv) > 6:
    
        msg="""Usage: python3 test_large_randompdf.py t x y z [iter]

Where: t    is the size of the range of T;
       x    is the size of the range of X;
       y    is the size of the range of Y;
       z    is the size of the range of Z;
       iter is the number of pdfs  ;
                        (defaults to 250)."""
        print(msg)
        exit(0)
    #^ if

    try:
        nT = int(sys_argv[1])
        nX = int(sys_argv[2])
        nY = int(sys_argv[3])
        nZ = int(sys_argv[4])
        if len(sys_argv)==6:    maxiter = int(sys_argv[5])
        else:               maxiter = 250
    except:
        print("I couldn't parse one of the arguments (they must all be integers)")
        exit(1)
    #^except

    if min(nT,nX,nY,nZ) < 2:
        print("All sizes of ranges must be at least 2.")
        exit(1)
    #^ if

    if maxiter < 1:
        print("# iterations must be >= 1.")
        exit(1)
    #^ if

    # Compute PID for randomly sampled pdfs
    Npid = compute_pid(nT,nX,nY,nZ,maxiter)
    print("list of pdfs w/ negative PID", Npid)
#^ Main()

#--------
# Run It!
#--------
if __name__=='__main__':
    Main([10, 10, 10, 10, 10, 10])

#EOF
