"""
Script to extract torsion fragments from parent molecules.

Torsion fragments are extracted based on a set of rules

"""
import logging
from openeye import oechem
from torsion.utils.process_sd_data import get_sd_data, has_sd_data


def get_torsion_oeatom_list(mol, tag="TORSION_ATOMS_FRAGMENT"):
    if has_sd_data(mol, tag):
        torsion_atoms = get_sd_data(mol, tag)
        try:
            torsion_atoms_idx = list(map(int, torsion_atoms.split()))
            torsion_oeatoms = map(
                lambda idx: mol.GetAtom(oechem.OEHasAtomIdx(idx - 1)), torsion_atoms_idx
            )
            return list(torsion_oeatoms)
        except Exception as e:
            print(e)
            return None


def get_torsion_oebond(mol, tag="TORSION_ATOMS_FRAGMENT"):
    torsion_atoms = get_torsion_oeatom_list(mol, tag)
    try:
        return mol.GetBond(torsion_atoms[1], torsion_atoms[2])
    except Exception as e:
        print(e)
        return None


def get_specific_dihedral_inchi_key(mol, torsion_atoms=None):
    """
    generates unique dihedral inchi key by mutating all four dihedral atoms 
    """
    try:
        if torsion_atoms and len(torsion_atoms) == 4:
            a, b, c, d = torsion_atoms[0], torsion_atoms[1], torsion_atoms[2], torsion_atoms[3]
        else:
            a, b, c, d = get_torsion_oeatom_list(mol)
        modified_inchi = get_modified_inchi_key(mol, [a, b, c, d])
        inchiKey = oechem.OECreateInChIKey(mol) + modified_inchi
        return inchiKey
    except Exception as e:
        logging.warning(e)
        return None


def get_modified_inchi_key(mol, atoms):
    """
    Generates InChIKey for the input molecule.
    Passed atoms of the molecule are mutated (atomic number changed)
    based on the mapping defined in the function.

    @param mol:
    @param atoms:
    @type mol: OEGraphMol
    @type atoms: list[OEAtombase]
    @return: str
    """
    copy_mol = oechem.OEGraphMol(mol)
    atom_map = {
        oechem.OEElemNo_C: oechem.OEElemNo_Pb,
        oechem.OEElemNo_N: oechem.OEElemNo_Bi,
        oechem.OEElemNo_O: oechem.OEElemNo_Po,
        oechem.OEElemNo_F: oechem.OEElemNo_At,
        oechem.OEElemNo_S: oechem.OEElemNo_Te,
        oechem.OEElemNo_Cl: oechem.OEElemNo_I,
        oechem.OEElemNo_P: oechem.OEElemNo_Sb,
        oechem.OEElemNo_Br: 117,
        oechem.OEElemNo_I: 118,
    }
    for ref_atom in atoms:
        copy_atom = copy_mol.GetAtom(oechem.OEHasAtomIdx(ref_atom.GetIdx()))
        if copy_atom is None:
            raise Exception("Null atom found")
        copy_atom.SetAtomicNum(atom_map[copy_atom.GetAtomicNum()])

    return oechem.OECreateInChIKey(copy_mol)


def get_modified_molecule_inchi(mol):
    title = mol.GetTitle().replace(',', '')
    inchi = oechem.OECreateInChIKey(mol)
    return inchi + title