# Hamiltonian Reconstruction (HR) distance as a success metric for the Variational Quantum Eigensolver (VQE)
![Alt text](https://github.com/mcmahon-lab/hamiltonian_reconstruction_metric/blob/master/images/HR_distance.png)
<br/><br/>

*HR distance* is a metric used to evaluate the performance of VQE that only requires a polynomial number of measurements concerning the number of operators. HR distance was introduced in (arxiv link)

In this repository, we provide code to run VQE and HR for two different systems: the 1D Transverse Field Ising model (1D TFIM) and the J1-J2 model (only with ZZ couplings). Directories **1D-TFIM** and **J1-J2** contain all the information to run VQE and HR for 1D TFIM and the J1-J2 model respectively. Example commands for each model are listed here: [1D TFIM](https://github.com/mcmahon-lab/hamiltonian_reconstruction_metric/tree/master/1D-TFIM#example-commands) and [J1-J2 model](https://github.com/mcmahon-lab/hamiltonian_reconstruction_metric/tree/master/J1-J2#example-commands). We let the user decide on the number of qubits, the coupling strength between the neighboring spins (nearest neighbor as well for the J1-J2 model), the number of shots, and the level of depolarization noise. For the quantum circuit ansatz used in VQE, we only provide an arbitrary n-layer ALA, which the user can decide on the number of layers. However, it is not difficult to modify the code to use the ansatz of interest. Lastly, we used python [qiskit](https://qiskit.org/) library to simulate all quantum circuits.

We further provide a *requirements.txt* file in the root directory of the git repo, which can be used to build a [virtual environment](https://docs.python.org/3/library/venv.html) to run all the simulations without package dependency concerns. Please feel free to email jm2239(-at-)cornell.edu, if there are any questions about the git repo.

# How to cite this code
If you use any of our code in your research, please consider citing the following paper:
>L.J.I Moon*, M.M. Sohoni*, M.A. Shimizu, P. Viswanathan, K.Zhang, E.A. Kim, and P.L. McMahon. "Hamiltonian-reconstruction distance as a success metric
for the Variational Quantum Eigensolver." (2024)

# License
The code in this repository is released under the following license:
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of the license is given in this repository as [License.txt](https://github.com/mcmahon-lab/hamiltonian_reconstruction_metric/blob/master/License.txt)



