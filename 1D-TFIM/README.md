# VQE and HR for 1D-TFIM
To run VQE for 1D-TFIM, please use the following command to get more information about the parameters
> python ionq_run.py -h

To run HR for 1D-TFIM, please use the following command to get more information about the parameters
> python ionq_run_HR.py -h

## Example commands

The command below runs VQE on 6 qubits 1-D TFIM with coupling strength 0.5, using a 4-layer ALA with no initialization parameter specified.
One-qubit and two-qubit gate depolarization values are set as 0.001 and 0.02 respectively.
All results are stored in OUTPUT_DIR.

> python ionq_run.py --n_qbts 6 --J 0.5 --shots 1000 --n_layers 4 --output_dir OUTPUT_DIR --init_param NONE --p1 0.001 --p2 0.02

The command below runs HR using the same model and ansatz when used to obtain the OUTPUT_DIR from simulating VQE.
One-qubit and two-qubit gate depolarization values are set as 0.001 and 0.02 respectively.
We only measure HR distances for indices specified in *param_idx_l.npy*.
Note that the *--param_idx_l* flag can only be used when there is a *param_idx_l.npy* file in OUTPUT_DIR.
*param_idx_l.npy* file can be created using **create_param_idx_l.py** script.
Without the *--param_idx_l* flag, the HR distances for all lists of parameters from simulated VQE are calculated.

> python ionq_run_HR.py --input_dir OUTPUT_DIR --shots 1000 --backend aer_simulator --param_idx_l --p1 0.001 --p2 0.02 
