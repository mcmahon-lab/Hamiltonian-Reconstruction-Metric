# Hamiltonian Reconstruction (HR) distance as a metric for Variational Quantum Eigensolver (VQE)
![Alt text](https://github.com/mcmahon-lab/hamiltonian_reconstruction_metric/blob/master/images/HR_distance.png)
<br/><br/>

*HR distance* is a metric used to evaluate the performance of VQE that only requires a polynomial number of measurements concerning the number of operators. HR distance was introduced in (arxiv link)

In this repository, we provide two examples of using HR distance as a metric in VQE for two different systems: the 1D Transverse Field Ising model (1D TFIM) and the J1-J2 model. We let the user decide on the number of qubits, the coupling strength between neighbors (nearest neighbor for the J1-J2 model), the number of shots, and the level of depolarization noise when simulating VQE and HR. For the circuit ansatz used in VQE, we only provide arbitrary n-layer ALA, where the user can decide on the number of layers. However, it is not difficult to modify the code to use the ansatz of interest for the user.

We used [qiskit](https://qiskit.org/) to simulate all quantum circuits.
