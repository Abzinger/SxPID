# SxPID
A differentiable measure of shared mutual information via overlapping exclusions in event (measure) spaces for discrete variables.

# What it does?
Computes a pointwise partial information decomposition (PPID) for multiple sources (up to 4 sources) and one target via the I_sx meausre. Pointwise means that every realization (point) in the distribution gets its own PID. In essence, it returns the PID of i(t: s1, s2), the local mutual information, for each realization and then average them to obtain that of I(T : S1, S2). 

For more details, check the preprint:
* A. Makkeh, A. Gutnecht, M. Wibral, *A differentiable measure of pointwise shared information*; [arXiv:2002.03356](https://arxiv.org/abs/2002.03356)


# User Guided Exmaple
The example file testing/test_and_gate.py has detailed explanation on how to run the code, in particular the main function `Sxpid.pid()` to compute the partial information decomposition. 
