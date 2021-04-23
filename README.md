# SxPID
A differentiable measure of shared mutual information via overlapping exclusions in event (measure) spaces for discrete variables.

# What it does?
Computes a pointwise partial information decomposition (PPID) for multiple sources (up to 4 sources) and one target via the I_sx meausre. Pointwise means that every realization (point) in the distribution gets its own PID. In essence, it returns the PPID of <img src="https://render.githubusercontent.com/render/math?math=i(t: s_1, s_2)">, the local mutual information -- for each realization -- and the PID of its average <img src="https://render.githubusercontent.com/render/math?math=I(T : S_1, S_2)">. 


For more details, check the preprint:
* A. Makkeh, A. Gutnecht, M. Wibral, *Introducting A differentiable measure of pointwise shared information*; [Phys Rev E 103, 032149](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.103.032149)

Note that SxPID is also **embbeded in the Information dynamics toolkit xl ([IDTxl](https://github.com/pwollstadt/IDTxl))** where you can use IDTxl's build-in functions to *analyse the node dynamics of networks* from multivariate time series data using SxPID.

# Installation
1. Download or clone the repository from GitHub
2. unpack it 
3. run (from the folder containing SxPID's setup.py file) the following 

``pip install .`` or the editable mode ``pip install -e .``

# User Guided Exmaple
The example `file demo/demo_and_gate.py` has detailed explanation on how to run the code, in particular the main function `Sxpid.pid()` to compute the partial information decomposition. 
