from openeye import oechem


def get_dihedral(mol, dihedralAtomIndices):
    """Get the dihedral corresponding to the indices in dihedralAtomIndices.

    Note that the indices are zero-indexed. The torsion of interest is the bond
    between atoms with indices 1 and 2.
    """
    # dihedralAtomIndices = [int(x) for x in get_sd_data(mol, 'TORSION_ATOMS_FRAGMENT').split()]
    dih = oechem.OEAtomBondSet()
    tor = oechem.OEAtomBondSet()
    for i in range(3):
        srcIdx = dihedralAtomIndices[i]
        destIdx = dihedralAtomIndices[i + 1]
        src = mol.GetAtom(oechem.OEHasAtomIdx(srcIdx))
        dest = mol.GetAtom(oechem.OEHasAtomIdx(destIdx))
        dih.AddAtom(src)
        bond = mol.GetBond(src, dest)
        dih.AddBond(bond)
        if i == 1:
            tor.AddBond(bond)
    dih.AddAtom(dest)

    return dih, tor
