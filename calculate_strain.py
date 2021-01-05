"""Generate torsional profiles for all rotatable bonds.

This script will read in an SD file containing one or more molecules and add an
SD property encoding the torsional profile (generated using TorsionNet) for each
rotatable bond.

This script requires you to have a trained TorsionNet model and the associated
standard scaler. Use the notebooks in the notebooks/ folder to generate these.

Example:
    python generate_profiles.py data/test_molecules.sdf test_molecules_out.sdf notebooks/model.h5 notebooks/scaler.pkl

"""

import time
import pickle
import logging
import argparse
import numpy as np
import tensorflow as tf
from openeye import oechem

from torsion.model import get_sf_elements
from torsion.confs import get_torsional_confs
from torsion.dihedral import (
    extract_torsion_atoms,
    get_canonical_torsions,
    extract_molecule_torsion_data,
    get_molecule_torsion_fragments,
    )
from torsion.inchi_keys import get_specific_dihedral_inchi_key
from torsion.utils.process_sd_data import (
    get_sd_data,
    TOTAL_STRAIN_TAG,
    reorder_sd_props,
    generate_energy_profile_sd_data_1d,
    extract_numeric_data_from_profile_str,
)
from torsion.utils.interpolate import get_global_min_interp1d

SPECIFIC_INCHI_TAG = 'specific_inchi'
NUM_TORSION_PROFILES_TAG = 'NUM_TORSIONNET_TORSION_PROFILES'
PROFILE_TAG = "TORSIONNET_PROFILE"
HAS_PROFILES_TAG = 'has_profiles'
SKIP_TORSION_TAG = 'skip_torsion'
ENERGY_PROFILE_TAG = 'ENERGY_PROFILE'
TORSION_ATOMS_FRAGMENT_TAG = 'TORSION_ATOMS_FRAGMENT'
LOW_PREDICTION_CONFIDENCE_TAG = 'LOW'
NUM_LOW_CONFIDENCE_TORSIONS_TAG = 'NUM_LOW_CONFIDENCE_TORSIONS'
OFFSET_THRESHOLD = 1.0
HIGH_PREDICTION_CONFIDENCE_TAG = 'HIGH'
PROFILE_OFFSET_TAG = 'profile_offset'
STRAIN_TAG = 'TORSIONNET_STRAIN'

def has_undesirable_elements(mol):
    '''
    returns True if molecule contains any element other than
    H, C, N, O, F, S, Cl, or P
    @param mol:
    @type mol: OEGraphMol
    @return: bool
    '''
    atomsHC = oechem.OEOrAtom(oechem.OEIsHydrogen(), oechem.OEIsCarbon())
    atomsNO = oechem.OEOrAtom(oechem.OEIsNitrogen(), oechem.OEIsOxygen())
    atomsFS = oechem.OEOrAtom(oechem.OEHasAtomicNum(9), oechem.OEIsSulfur())
    atomsHCNO = oechem.OEOrAtom(atomsHC, atomsNO)
    atomsHCNOFS = oechem.OEOrAtom(atomsHCNO, atomsFS)
    atomsHCNOFSCl = oechem.OEOrAtom(atomsHCNOFS, oechem.OEHasAtomicNum(17))
    atomsHCNOFSClP = oechem.OEOrAtom(atomsHCNOFSCl, oechem.OEIsPhosphorus())

    undesirable_atom = mol.GetAtom(oechem.OENotAtom(atomsHCNOFSClP))
    if undesirable_atom is not None:
        return True

    return False


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    # Logging arguments
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="increases log verbosity for each occurence.",
    )
    verbosity_group.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        const=-1,
        default=0,
        dest="verbose_count",
        help="quiet output (show errors only)",
    )

    # Program arguments
    parser.add_argument(
        "--in", dest='infile', type=str, help="Input file (.sdf) containing one or more molecules."
    )
    parser.add_argument(
        "--out",
        dest="outfile",
        type=str,
        help="Output file (.sdf) containing molecules annotated with torsional profiles.",
    )
    parser.add_argument("--model", dest="model_file", type=str, help="Path to trained TorsionNet model.")
    parser.add_argument("--scaler", dest="scaler_file", type=str, help="Path to scaler object used to scale input features to TorsionNet.")

    args = parser.parse_args()

    # Setup logging basics
    base_loglevel = 30
    verbosity = max(min(args.verbose_count, 2), -1)
    loglevel = base_loglevel - (verbosity * 10)

    logging.basicConfig(level=loglevel, format="%(message)s")
    logging.getLogger().setLevel(loglevel)

    logging.debug("DEBUG   messages will be shown.")  # 10 verbosity =  2
    logging.info("INFO    messages will be shown.")  # 20 verbosity =  1
    logging.warning("WARNING messages will be shown.")  # 30 verbosity =  0
    logging.error("ERROR   messages will be shown.")  # 40 verbosity = -1

    return args


def generate_torsion_profile(mol_list):
    sf_map = {}
    for graph_mol in mol_list:
        if oechem.OECount(graph_mol, oechem.OEIsRotor()) == 0:
            logging.warning('WARNING: Skipping molecule %s... rotor count is zero', graph_mol.GetTitle())
            continue
        
        frag_mols = get_molecule_torsion_fragments(graph_mol)
        if len(frag_mols) == 0:
            logging.warning('WARNING: Skipping molecule %s... cannot identify torsional fragments', graph_mol.GetTitle())
            continue

        _, torsion_data = extract_molecule_torsion_data(graph_mol, frag_mols)

        for frag_mol in frag_mols:
            if has_undesirable_elements(frag_mol) or oechem.OECount(frag_mol, oechem.OEIsPhosphorus()) > 0:
                logging.warning('WARNING: Skipping a fragment in molecule %s... fragment has undesirable elements', graph_mol.GetTitle())
                continue

            # skip fragments with one or more formal charge
            skip_torsion = False
            if oechem.OECount(frag_mol, oechem.OEHasFormalCharge(1)) > 0 \
                or oechem.OECount(frag_mol, oechem.OEHasFormalCharge(2)) > 0:
                skip_torsion = True

            specific_inchi = get_specific_dihedral_inchi_key(frag_mol)

            if specific_inchi not in sf_map:
                sf_list = get_profile_sf(frag_mol)
                sf_map[specific_inchi] = sf_list

            torsion_data_items = torsion_data[specific_inchi]
            for torsion_data_item in torsion_data_items:
                a_idx, b_idx, c_idx, d_idx, _ = torsion_data_item
                b = graph_mol.GetAtom(oechem.OEHasAtomIdx(b_idx))
                c = graph_mol.GetAtom(oechem.OEHasAtomIdx(c_idx))

                bond = graph_mol.GetBond(b, c)
                if skip_torsion:
                    bond.SetData(SKIP_TORSION_TAG, True)

                tor_atoms_str = ' '.join(list(
                    map(str, [a_idx, b_idx, c_idx, d_idx])))
                if not bond.HasData(TORSION_ATOMS_FRAGMENT_TAG):
                    bond.SetData(TORSION_ATOMS_FRAGMENT_TAG, tor_atoms_str)
                    bond.SetData(SPECIFIC_INCHI_TAG, specific_inchi)
                else:
                    tmp_data = bond.GetData(TORSION_ATOMS_FRAGMENT_TAG)
                    tmp_data = tmp_data + ':' + tor_atoms_str
                    bond.SetData(TORSION_ATOMS_FRAGMENT_TAG, tmp_data)

        graph_mol.SetData(HAS_PROFILES_TAG, False)
        for bond in graph_mol.GetBonds(oechem.OEIsRotor()):
            if bond.HasData(TORSION_ATOMS_FRAGMENT_TAG):
                graph_mol.SetData(HAS_PROFILES_TAG, True)
                break
    
    return mol_list, sf_map




def get_profile_sf(mol):
    '''
    Generates torsional conformations and corresponding
    symmetric functions. Returns list containing
    symmetry functions and angles
    @param mol: OEMol
    @return: list[list, list]
    '''
    torsional_mols = get_torsional_confs(mol)

    sf_list = []
    for torsional_mol in torsional_mols:
        rsf = get_sf_elements(torsional_mol)
        torsion_angle = float(get_sd_data(torsional_mol, 'TORSION_ANGLE'))
        if torsion_angle > 180.0:
            torsion_angle = torsion_angle - 360.0
        sf_list.append((torsion_angle, rsf))

    sf_list.sort()

    return sf_list

def calculate_ml_profiles(mols, sf_map, model, scaler):
    num_mols = len(mols)
    for count, mol in enumerate(mols):
        logging.info("Generating molecule profile %d/%d", (count+1), num_mols)

        if mol.HasData(HAS_PROFILES_TAG) and mol.GetData(HAS_PROFILES_TAG) == True:
            for bond in mol.GetBonds(oechem.OEIsRotor()):
                if bond.HasData(ENERGY_PROFILE_TAG) and bond.HasData(PROFILE_OFFSET_TAG):
                    continue

                if bond.HasData(SPECIFIC_INCHI_TAG):
                    specific_inchi = bond.GetData(SPECIFIC_INCHI_TAG)

                    profile, offset = calculate_sf_ML_profile(model, scaler,
                                                              sf_map[specific_inchi])
                    bond.SetData(ENERGY_PROFILE_TAG, profile)
                    bond.SetData(PROFILE_OFFSET_TAG, float(offset))



def calculate_sf_ML_profile(regr, scaler, sf_list):
    X = np.array([sf for angle, sf in sf_list])
    X = scaler.transform(X)
    y = regr.predict(X).flatten()

    # offset correction
    minE = np.min(y)
    relE = y - minE
    angles = [angle for angle, sf in sf_list]
    profile_str = generate_energy_profile_sd_data_1d(zip(angles, relE))

    return profile_str, minE


def generate_torsional_strain(mol):
    '''
    Calculate strain energy of each rotatable bond using attached ML profiles
    @param mol: OEGraphMol
    @param gen_confs: bool
    @param num_confs: int
    @return: None
    '''
    for bond in mol.GetBonds(oechem.OEIsRotor()):
        if bond.HasData(ENERGY_PROFILE_TAG):
            energy_profile = bond.GetData(ENERGY_PROFILE_TAG)
            x, y = extract_numeric_data_from_profile_str(energy_profile)
            _, f = get_global_min_interp1d(x, y)

            bond_strain = 1e10
            torsion_atoms_list = extract_torsion_atoms(mol, bond)
            for torsion_atoms in torsion_atoms_list:
                a, b, c, d = torsion_atoms
                angle = oechem.OEGetTorsion(mol, a, b, c, d) * oechem.Rad2Deg
                strainE = float(f(angle))
                if strainE < 0.0:
                    strainE = 0.0
                if bond_strain > strainE:
                    bond_strain = strainE

                tor_atoms_str = ' '.join(list(
                    map(str, [a.GetIdx(), b.GetIdx(), c.GetIdx(), d.GetIdx()])))

                bond.SetData(TORSION_ATOMS_FRAGMENT_TAG, tor_atoms_str)

            bond.SetData(STRAIN_TAG, bond_strain)

    if mol.HasData(HAS_PROFILES_TAG) and mol.GetData(HAS_PROFILES_TAG):
        save_profile_as_sd(mol)

    return mol


def generate_ml_profiles(mol_list, model_file, scaler_file):
    out_mols, sf_map = generate_torsion_profile(mol_list)

    logging.info('Loading ML model...')
    model = tf.keras.models.load_model(model_file)

    # load scaler
    logging.info('Loading descriptor transformation data...')
    with open(scaler_file, 'rb') as fptr:
        scaler = pickle.load(fptr)

    # calculate ML profile of each input molecule using attached data
    calculate_ml_profiles(out_mols, sf_map, model, scaler)

    out_mols2 = []
    for out_mol in out_mols:
        out_mols2.append(generate_torsional_strain(out_mol))

    return out_mols2

def save_profile_as_sd(mol:oechem.OEGraphMol):
    oechem.OEDeleteSDData(mol, TOTAL_STRAIN_TAG)
    oechem.OESetSDData(mol, TOTAL_STRAIN_TAG, '') # place holder

    oechem.OEDeleteSDData(mol, NUM_TORSION_PROFILES_TAG)
    oechem.OESetSDData(mol, NUM_TORSION_PROFILES_TAG, '')

    oechem.OEDeleteSDData(mol, NUM_LOW_CONFIDENCE_TORSIONS_TAG)
    oechem.OESetSDData(mol, NUM_LOW_CONFIDENCE_TORSIONS_TAG, '')

    strain_arr = np.zeros(1)
    strain_arr_high_conf_preds = np.zeros(1)

    num_torsion_profiles = 0
    num_low_confidence_torsions = 0

    can_torsions = get_canonical_torsions(mol)
    for num, can_torsion in enumerate(can_torsions):
        bond = mol.GetBond(can_torsion.b, can_torsion.c)
        if bond is not None and bond.HasData(ENERGY_PROFILE_TAG):
            num_torsion_profiles += 1
            bond_strains = bond.GetData(STRAIN_TAG)
            profile_offset = bond.GetData(PROFILE_OFFSET_TAG)
            if profile_offset < OFFSET_THRESHOLD and (not bond.HasData(SKIP_TORSION_TAG)):
                strain_arr_high_conf_preds += np.array(bond_strains)

            strain_arr += np.array(bond_strains)

            offset = bond.GetData(PROFILE_OFFSET_TAG)
            profile_str = bond.GetData(ENERGY_PROFILE_TAG)
            pred_confidence_value = HIGH_PREDICTION_CONFIDENCE_TAG
            if offset > OFFSET_THRESHOLD or bond.HasData(SKIP_TORSION_TAG):
                profile_str = 'LOW CONFIDENCE - ' + profile_str
                pred_confidence_value = LOW_PREDICTION_CONFIDENCE_TAG
                num_low_confidence_torsions += 1

            #tor_atoms_str = bond.GetData(TORSION_ATOMS_FRAGMENT_TAG)
            #tor_atoms_str_list = tor_atoms_str.split(':')
            #a_idx, b_idx, c_idx, d_idx = list(map(int, tor_atoms_str_list[0].split()))

            tor_atoms_str1 = bond.GetData(TORSION_ATOMS_FRAGMENT_TAG)
            ca, cb, cc, cd = list(map(int, tor_atoms_str1.split()))

            apStr = "{}:{}:{}:{}".format(ca+1, cb+1, cc+1, cd+1)

            atom_ca = mol.GetAtom(oechem.OEHasAtomIdx(ca))
            atom_cb = mol.GetAtom(oechem.OEHasAtomIdx(cb))
            atom_cc = mol.GetAtom(oechem.OEHasAtomIdx(cc))
            atom_cd = mol.GetAtom(oechem.OEHasAtomIdx(cd))
            angle_float = oechem.OEGetTorsion(mol, atom_ca, atom_cb,
                                                atom_cc, atom_cd) * oechem.Rad2Deg

            sd_tag1 = 'TORSION_%s_ATOMS'%(num+1)
            sd_tag2 = 'TORSION_%d_TORSIONNET_%s'%(num+1, ENERGY_PROFILE_TAG)
            sd_tag3 = 'TORSION_%d_TORSIONNET_PRED_CONFIDENCE'%(num+1)
            sd_tag4 = 'TORSION_%d_TORSIONNET_PROFILE_OFFSET'%(num+1)

            oechem.OEDeleteSDData(mol, sd_tag1)
            oechem.OEDeleteSDData(mol, sd_tag2)
            oechem.OEDeleteSDData(mol, sd_tag3)
            oechem.OEDeleteSDData(mol, sd_tag4)

            oechem.OESetSDData(mol, sd_tag1, apStr)
            oechem.OESetSDData(mol, sd_tag2, profile_str)

            sd_tag6 = 'TORSION_%d_%s'%((num+1), STRAIN_TAG)
            oechem.OEDeleteSDData(mol, sd_tag6)
            oechem.OESetSDData(mol, sd_tag6, '%.1f'%bond_strains)

            angle = '%.1f'%angle_float
            sd_tag5 = 'TORSION_%d_ANGLE'%(num+1)
            oechem.OEDeleteSDData(mol, sd_tag5)
            oechem.OESetSDData(mol, sd_tag5, angle)
            oechem.OESetSDData(mol, sd_tag3, pred_confidence_value)
            oechem.OESetSDData(mol, sd_tag4, '%.2f'%offset)

    strain_str = '%.1f' % strain_arr_high_conf_preds[0]
    oechem.OESetSDData(mol, TOTAL_STRAIN_TAG, strain_str)
    oechem.OESetSDData(mol, NUM_TORSION_PROFILES_TAG, str(num_torsion_profiles))
    oechem.OESetSDData(mol, NUM_LOW_CONFIDENCE_TORSIONS_TAG, str(num_low_confidence_torsions))

    reorder_sd_props(mol)

    return mol


def main():
    args = parse_args()

    start_time = time.time()
    ifs = oechem.oemolistream()
    if ifs.open(args.infile):
        mols = [oechem.OEMol(mol) for mol in ifs.GetOEGraphMols()]
        ifs.close()
    else:
        oechem.OEThrow.Fatal(f"Unable to open {args.infile}")

    out_mols = generate_ml_profiles(mols, args.model_file, args.scaler_file)
    count = len(out_mols)

    ofs = oechem.oemolostream()
    if ofs.open(args.outfile):
        for mol in out_mols:
            oechem.OEWriteMolecule(ofs, mol)
        ofs.close()
    else:
        oechem.OEThrow.Fatal(f"Unable to open {args.outfile}")

    time_elapsed = time.time() - start_time
    logging.warning('Generated profiles for %d molecules in %.1f seconds', count, time_elapsed)


if __name__ == "__main__":
    main()
