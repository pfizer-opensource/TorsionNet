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

## Generating TorsionNet torsional profiles for your molecules
Once you have trained a TorsionNet model using the notebooks above you should have
the model (`model.h5`) and the associated standard scaler (`scaler.pkl`) in the 
`notebooks/` folder.

You can use this trained model to predict torsional profiles and estimate the torsional
strain of your molecules. Create an SD file containing your molecules (with explicit 
hydrogens and with 3D coordinates) and then invoke the `calculate_strain.py` script. 
As an example, we will use the SD file containing 2 Pfizer molecules (Lyrica and Detrol) at `data/test_molecules.sdf`:
```
python calculate_strain.py data/test_molecules.sdf test_molecules_out.sdf notebooks/model.h5 notebooks/scaler.pkl
```
The output SD file (`test_molecules_out.sdf` in the invocation above) will contain the total estimated strain for the molecule as well as the torsional energy profiles and individual strains for every rotatable bond in the molecule. For example, for Lyrica, you should see something like:
```
TORSIONNET_STRAIN : 3.1
NUM_TORSIONNET_TORSION_PROFILES : 5
NUM_LOW_CONFIDENCE_TORSIONS : 1
TORSION_1_ATOMS : 1:2:5:9
TORSION_1_TORSIONNET_ENERGY_PROFILE : -180.00:1.14,-165.00:0.78,-150.00:0.75,-135.00:0.44,-120.00:0.18,-105.00:0.00,-90.00:0.02,-75.00:0.10,-60.00:0.24,-45.00:0.59,-30.00:1.24,-15.00:2.07,0.00:2.93,15.00:2.88,30.00:1.95,45.00:1.23,60.00:0.79,75.00:1.03,90.00:2.11,105.00:3.74,120.00:5.09,135.00:4.96,150.00:3.47,165.00:1.90,180.00:1.14
TORSION_1_TORSIONNET_STRAIN : 0.2
TORSION_1_ANGLE : 178.1
TORSION_1_TORSIONNET_PRED_CONFIDENCE : HIGH
TORSION_1_TORSIONNET_PROFILE_OFFSET : 0.58
TORSION_2_ATOMS : 3:1:2:5
TORSION_2_TORSIONNET_ENERGY_PROFILE : -180.00:1.13,-165.00:1.42,-150.00:2.17,-135.00:3.39,-120.00:5.09,-105.00:3.28,-90.00:1.66,-75.00:0.77,-60.00:0.65,-45.00:1.21,-30.00:2.19,-15.00:3.08,0.00:3.98,15.00:3.11,30.00:1.75,45.00:0.73,60.00:0.25,75.00:0.05,90.00:0.00,105.00:0.22,120.00:0.50,135.00:0.91,150.00:1.15,165.00:1.34,180.00:1.13
TORSION_2_TORSIONNET_STRAIN : 1.2
TORSION_2_ANGLE : 177.4
TORSION_2_TORSIONNET_PRED_CONFIDENCE : HIGH
TORSION_2_TORSIONNET_PROFILE_OFFSET : 0.57
TORSION_3_ATOMS : 2:1:3:6
TORSION_3_TORSIONNET_ENERGY_PROFILE : -180.00:1.46,-165.00:1.62,-150.00:1.75,-135.00:1.43,-120.00:0.88,-105.00:0.35,-90.00:0.09,-75.00:0.01,-60.00:0.00,-45.00:0.21,-30.00:1.03,-15.00:2.35,0.00:3.15,15.00:2.84,30.00:1.82,45.00:0.79,60.00:0.19,75.00:0.49,90.00:1.37,105.00:3.48,120.00:4.65,135.00:4.55,150.00:3.15,165.00:1.82,180.00:1.46
TORSION_3_TORSIONNET_STRAIN : 1.6
TORSION_3_ANGLE : -167.2
TORSION_3_TORSIONNET_PRED_CONFIDENCE : HIGH
TORSION_3_TORSIONNET_PROFILE_OFFSET : 0.33
TORSION_4_ATOMS : 1:3:6:11
TORSION_4_TORSIONNET_ENERGY_PROFILE : LOW CONFIDENCE - -180.00:0.56,-165.00:1.07,-150.00:1.73,-135.00:2.46,-120.00:2.77,-105.00:2.18,-90.00:1.13,-75.00:0.79,-60.00:0.88,-45.00:0.74,-30.00:0.41,-15.00:0.14,0.00:0.00,15.00:0.08,30.00:0.35,45.00:0.73,60.00:0.78,75.00:0.78,90.00:1.51,105.00:2.70,120.00:2.98,135.00:2.43,150.00:1.50,165.00:0.81,180.00:0.56
TORSION_4_TORSIONNET_STRAIN : 0.8
TORSION_4_ANGLE : 74.1
TORSION_4_TORSIONNET_PRED_CONFIDENCE : LOW
TORSION_4_TORSIONNET_PROFILE_OFFSET : 3.09
TORSION_5_ATOMS : 2:1:4:7
TORSION_5_TORSIONNET_ENERGY_PROFILE : -180.00:1.16,-165.00:1.20,-150.00:1.65,-135.00:2.27,-120.00:2.73,-105.00:2.16,-90.00:0.82,-75.00:0.13,-60.00:0.03,-45.00:0.23,-30.00:0.85,-15.00:1.78,0.00:2.26,15.00:1.79,30.00:0.78,45.00:0.16,60.00:0.00,75.00:0.03,90.00:0.12,105.00:0.31,120.00:0.58,135.00:0.82,150.00:1.08,165.00:1.19,180.00:1.16
TORSION_5_TORSIONNET_STRAIN : 0.1
TORSION_5_ANGLE : 89.9
TORSION_5_TORSIONNET_PRED_CONFIDENCE : HIGH
TORSION_5_TORSIONNET_PROFILE_OFFSET : 0.39
```

