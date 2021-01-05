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
As an example, we will use the SD file containing 2 approved drug molecules (Salbutamol and Imatinib) at `data/test_molecules.sdf`:
```
python calculate_strain.py \
  --in data/test_molecules.sdf \
  --out test_molecules_out.sdf \
  --model notebooks/model.h5 \
  --scaler notebooks/scaler.pkl
```
The output SD file (`test_molecules_out.sdf` in the invocation above) will contain the total estimated strain for the molecule as well as the torsional energy profiles and individual strains for every rotatable bond in the molecule. For example, for Salbutamol, you should see something like:
```
TORSIONNET_STRAIN : 0.9
NUM_TORSIONNET_TORSION_PROFILES : 8
NUM_LOW_CONFIDENCE_TORSIONS : 3
TORSION_1_ATOMS : 11:13:14:15
TORSION_1_TORSIONNET_ENERGY_PROFILE : -180.00:0.47,-165.00:0.34,-150.00:0.16,-135.00:0.06,-120.00:0.01,-105.00:0.22,-90.00:0.62,-75.00:0.38,-60.00:0.31,-45.00:0.30,-30.00:0.32,-15.00:0.38,0.00:0.54,15.00:0.38,30.00:0.35,45.00:0.33,60.00:0.34,75.00:0.41,90.00:0.53,105.00:0.19,120.00:0.00,135.00:0.00,150.00:0.05,165.00:0.08,180.00:0.47
TORSION_1_TORSIONNET_STRAIN : 0.4
TORSION_1_ANGLE : -76.0
TORSION_1_TORSIONNET_PRED_CONFIDENCE : HIGH
TORSION_1_TORSIONNET_PROFILE_OFFSET : 0.17
TORSION_2_ATOMS : 10:11:13:14
TORSION_2_TORSIONNET_ENERGY_PROFILE : -180.00:0.00,-165.00:0.14,-150.00:0.47,-135.00:4.14,-120.00:7.21,-105.00:9.37,-90.00:7.02,-75.00:5.80,-60.00:3.94,-45.00:2.48,-30.00:1.63,-15.00:0.39,0.00:0.59,15.00:0.96,30.00:1.44,45.00:2.42,60.00:7.64,75.00:9.18,90.00:7.22,105.00:5.30,120.00:2.84,135.00:1.26,150.00:0.39,165.00:0.07,180.00:0.00
TORSION_2_TORSIONNET_STRAIN : 0.3
TORSION_2_ANGLE : -10.2
TORSION_2_TORSIONNET_PRED_CONFIDENCE : HIGH
TORSION_2_TORSIONNET_PROFILE_OFFSET : 0.58
TORSION_3_ATOMS : 19:18:20:21
TORSION_3_TORSIONNET_ENERGY_PROFILE : -180.00:0.00,-165.00:0.02,-150.00:0.13,-135.00:0.93,-120.00:2.98,-105.00:5.05,-90.00:5.45,-75.00:2.89,-60.00:0.95,-45.00:0.37,-30.00:0.25,-15.00:0.24,0.00:0.28,15.00:0.25,30.00:0.24,45.00:0.39,60.00:1.10,75.00:3.14,90.00:5.47,105.00:5.07,120.00:2.87,135.00:0.75,150.00:0.11,165.00:0.01,180.00:0.00
TORSION_3_TORSIONNET_STRAIN : 0.0
TORSION_3_ANGLE : 3.8
TORSION_3_TORSIONNET_PRED_CONFIDENCE : HIGH
TORSION_3_TORSIONNET_PROFILE_OFFSET : 0.17
TORSION_4_ATOMS : 2:3:7:8
TORSION_4_TORSIONNET_ENERGY_PROFILE : -180.00:0.31,-165.00:0.12,-150.00:0.00,-135.00:0.21,-120.00:1.67,-105.00:3.21,-90.00:4.09,-75.00:4.03,-60.00:3.11,-45.00:1.72,-30.00:0.63,-15.00:0.13,0.00:0.04,15.00:0.14,30.00:0.70,45.00:1.82,60.00:3.22,75.00:4.11,90.00:4.04,105.00:3.09,120.00:1.54,135.00:0.22,150.00:0.03,165.00:0.09,180.00:0.31
TORSION_4_TORSIONNET_STRAIN : 0.1
TORSION_4_ANGLE : -12.5
TORSION_4_TORSIONNET_PRED_CONFIDENCE : HIGH
TORSION_4_TORSIONNET_PROFILE_OFFSET : 0.07
TORSION_5_ATOMS : 18:20:21:36
TORSION_5_TORSIONNET_ENERGY_PROFILE : LOW CONFIDENCE - -180.00:0.22,-165.00:0.47,-150.00:0.71,-135.00:1.50,-120.00:2.64,-105.00:3.76,-90.00:3.91,-75.00:2.74,-60.00:1.16,-45.00:0.41,-30.00:0.21,-15.00:0.11,0.00:0.06,15.00:0.08,30.00:0.19,45.00:0.47,60.00:1.38,75.00:2.96,90.00:3.88,105.00:3.63,120.00:2.29,135.00:1.04,150.00:0.25,165.00:0.00,180.00:0.22
TORSION_5_TORSIONNET_STRAIN : 0.1
TORSION_5_ANGLE : -11.7
TORSION_5_TORSIONNET_PRED_CONFIDENCE : LOW
TORSION_5_TORSIONNET_PROFILE_OFFSET : 1.56
TORSION_6_ATOMS : 20:21:22:23
TORSION_6_TORSIONNET_ENERGY_PROFILE : -180.00:0.00,-165.00:0.02,-150.00:0.13,-135.00:0.93,-120.00:2.98,-105.00:5.05,-90.00:5.45,-75.00:2.89,-60.00:0.95,-45.00:0.37,-30.00:0.25,-15.00:0.24,0.00:0.28,15.00:0.25,30.00:0.24,45.00:0.39,60.00:1.10,75.00:3.14,90.00:5.47,105.00:5.07,120.00:2.87,135.00:0.75,150.00:0.11,165.00:0.01,180.00:0.00
TORSION_6_TORSIONNET_STRAIN : 0.0
TORSION_6_ANGLE : -158.6
TORSION_6_TORSIONNET_PRED_CONFIDENCE : HIGH
TORSION_6_TORSIONNET_PROFILE_OFFSET : 0.17
TORSION_7_ATOMS : 24:25:28:29
TORSION_7_TORSIONNET_ENERGY_PROFILE : LOW CONFIDENCE - -180.00:0.13,-165.00:0.14,-150.00:0.15,-135.00:0.19,-120.00:0.30,-105.00:0.33,-90.00:0.37,-75.00:0.37,-60.00:0.36,-45.00:0.30,-30.00:0.24,-15.00:0.21,0.00:0.32,15.00:0.43,30.00:0.42,45.00:0.43,60.00:0.42,75.00:0.39,90.00:0.35,105.00:0.27,120.00:0.15,135.00:0.13,150.00:0.06,165.00:0.00,180.00:0.13
TORSION_7_TORSIONNET_STRAIN : 0.2
TORSION_7_ANGLE : -137.6
TORSION_7_TORSIONNET_PRED_CONFIDENCE : LOW
TORSION_7_TORSIONNET_PROFILE_OFFSET : 3.31
TORSION_8_ATOMS : 25:28:29:35
TORSION_8_TORSIONNET_ENERGY_PROFILE : LOW CONFIDENCE - -180.00:0.07,-165.00:0.24,-150.00:0.47,-135.00:0.75,-120.00:0.98,-105.00:1.03,-90.00:0.90,-75.00:0.76,-60.00:0.73,-45.00:0.82,-30.00:0.99,-15.00:1.10,0.00:1.11,15.00:0.95,30.00:0.70,45.00:0.44,60.00:0.22,75.00:0.13,90.00:0.10,105.00:0.08,120.00:0.07,135.00:0.05,150.00:0.02,165.00:0.00,180.00:0.07
TORSION_8_TORSIONNET_STRAIN : 0.2
TORSION_8_ANGLE : -170.0
TORSION_8_TORSIONNET_PRED_CONFIDENCE : LOW
TORSION_8_TORSIONNET_PROFILE_OFFSET : 6.31
```

