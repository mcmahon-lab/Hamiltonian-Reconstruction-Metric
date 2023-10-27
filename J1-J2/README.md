# VQE and HR for J1-J2 model
To run VQE for the J1-J2 model, please use the following command to get more information about the parameters
> python VQE_J1_J2.py -h

To run HR for the J1-J2 model, please use the following command to get more information about the parameters
> python HR_J1_J2.py -h

## Example commands

The command below runs VQE on the 3x2 J1-J2 model with the nearest-neighbor coupling strength as 0.5 and the next-nearest coupling strength as 0.2.
For our quantum ansatz, we use 4 layers ALA and without initialization parameter specified
One-qubit and two-qubit gate depolarization values are set as 0.001 and 0.02 respectively when running quantum circuit simulation.
All results are stored in OUTPUT_DIR.

> python VQE_J1_J2.py --m 3 --n 2 --J1 0.5 --J2 0.2 --shots 1000 --n_layers 4 --output_dir OUTPUT_DIR --init_param NONE --p1 0.001 --p2 0.02

The command below runs HR using the same model and ansatz when used to obtain the OUTPUT_DIR from simulating VQE.
One-qubit and two-qubit gate depolarization values are set as 0.001 and 0.02 respectively when running quantum circuit simulation.
We only measure HR distances for indices specified in *param_idx_l.npy*.
Note that the *--param_idx_l* flag can be only used when there is a *param_idx_l.npy* file in OUTPUT_DIR, which can be created using **create_param_idx_l.py** file.
Without the *--param_idx_l* flag, the code calculates HR distances for all lists of parameters from simulated VQE.

> python HR_J1_j2.py --input_dir OUTPUT_DIR --shots 1000 --backend aer_simulator --param_idx_l --p1 0.001 --p2 0.02 
