# TorsionNet
This repository accompanies the manuscript "TorsionNet: A Deep Neural Network to Predict Small Molecule Torsion Energy Profiles with the Accuracy of Quantum Mechanics."
The code and notebooks in this repository can be used to train a TorsionNet model to predict QM energy profiles and predict the energy profile for a new torsional fragment.

## Environment
Create a conda environment using the `env.yml` file provided.
  
You will need a license to the OpenEye Toolkit.

## Data
SD files containing a set of 500 fragments, the corresponding MM conformers, and the corresponding QM optimized conformers are tracked using Git LFS and are present in the `data` directory.

## Training and using a TorsionNet Model
Follow the jupyter notebooks in this order:

1. `01_preprocess_data`: Combine features obtained from the geometries of MM conformers with the corresponding QM energies and generate training and test splits.
2. `02_TorsionNet_training`: Train a TorsionNet model on the training data.
3. `03_TorsionNet_performance`: Assess the performance of the trained TorsionNet model on training and test splits.
4. `04_TorsionNet_inference`: Use TorsionNet to predict the energy profile of a fragment starting from the SMILES and one particular torsion of interest.

