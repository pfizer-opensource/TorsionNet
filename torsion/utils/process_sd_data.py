from openeye import oechem

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

