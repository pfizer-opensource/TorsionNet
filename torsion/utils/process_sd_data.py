import numpy as np
from openeye import oechem

TOTAL_STRAIN_TAG = 'TORSIONNET_STRAIN'

def has_sd_data(mol, tag):
    if oechem.OEHasSDData(mol, tag):
        return True
    if oechem.OEHasSDData(mol.GetActive(), tag):
        return True
    return False


def get_sd_data(mol, tag):
    try:
        if oechem.OEHasSDData(mol, tag):
            return oechem.OEGetSDData(mol, tag)
        if oechem.OEHasSDData(mol.GetActive(), tag):
            return oechem.OEGetSDData(mol.GetActive(), tag)
    except AttributeError as e:
        print(e)
        return ""


def delete_sd_data(mol, tag, locator_tag):
    if oechem.OEHasSDData(mol, locator_tag):
        return oechem.OEDeleteSDData(mol, tag)
    elif oechem.OEHasSDData(mol.GetActive(), locator_tag):
        return oechem.OEDeleteSDData(mol.GetActive(), tag)
    return False


def dump_sd_data(mol):
    print("Data Attached at the molecule level:")
    for dp in oechem.OEGetSDDataPairs(mol):
        print(dp.GetTag(), ":", dp.GetValue())
    if type(mol) == oechem.OEMol:
        print("\n\n" + 10 * "-" + "Data Attached to Conformers:")
        for conf_id, conf in enumerate(mol.GetConfs()):
            print("Data attached to conformer {}:".format(conf_id))
            for dp in oechem.OEGetSDDataPairs(conf):
                print(dp.GetTag(), ":", dp.GetValue())
    print()


def generate_energy_profile_sd_data_1d(data):
    angles, energies = zip(*data)

    angles = list(angles)
    energies = list(energies)
    angles.insert(0, -180)
    energies.insert(0, energies[-1])

    min_energy = min(energies)
    rel_energies = [energy - min_energy for energy in energies]

    # Generate sddata string
    sddata = ','.join(["{:.2f}:{:.2f}".format(angle, energy)
                       for angle, energy in zip(angles, rel_energies)])

    return sddata

def extract_numeric_data_from_profile_str(profile):
    xyData = [map(float, item.split(':')) for item in profile.split(',')]
    x, y = zip(*xyData)

    return np.array(x), np.array(y)


def reorder_sd_props(mol:oechem.OEGraphMol):
    strain1 = oechem.OEGetSDData(mol, TOTAL_STRAIN_TAG)

    data_pairs = [(TOTAL_STRAIN_TAG, strain1)]
    for dp in oechem.OEGetSDDataPairs(mol):
        if dp.GetTag() == TOTAL_STRAIN_TAG:
            pass
        else:
            data_pairs.append((dp.GetTag(), dp.GetValue()))

    oechem.OEClearSDData(mol)

    for k, v in data_pairs:
        oechem.OESetSDData(mol, k, v)

    data_pairs = []
    for dp in oechem.OEGetSDDataPairs(mol):
        data_pairs.append((dp.GetTag(), dp.GetValue()))
    oechem.OEClearSDData(mol)

    for k, v in data_pairs:
        oechem.OESetSDData(mol, k, v)