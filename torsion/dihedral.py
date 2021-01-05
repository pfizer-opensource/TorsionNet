import operator
import logging
import collections
from openeye import oechem, oemedchem
from torsion.utils.process_sd_data import has_sd_data, get_sd_data
from torsion.inchi_keys import get_specific_dihedral_inchi_key, get_modified_molecule_inchi

TORSION_ATOMS_FRAG_KEY = "TORSION_ATOMS_FRAGMENT"
TORSION_ATOMS_FRAGMENT_TAG = 'TORSION_ATOMS_FRAGMENT'
FRAGMENT_TO_PARENT_ATOMS_KEY = 'FRAGMENT_TO_PARENT_ATOMS'
TORSION_ATOMS_PARENT_MOL_KEY = "TORSION_ATOMS_ParentMol"


def extract_torsion_atoms(mol, bond):
    torsion_atoms_str_list = bond.GetData(TORSION_ATOMS_FRAGMENT_TAG).split(':')
    torsion_atoms_list = []
    for torsion_atoms_str in torsion_atoms_str_list:
        torsion_atoms_idx = list(map(int, torsion_atoms_str.split()))
        torsion_oeatoms \
            = map(lambda idx: mol.GetAtom(oechem.OEHasAtomIdx(idx)),
                  torsion_atoms_idx)
        a, b, c, d = list(torsion_oeatoms)
        torsion_atoms_list.append((a, b, c, d))
    return torsion_atoms_list


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

class TorsionGenerator():
    @staticmethod
    def IsOrtho(atom, torsion):
        '''
        @param atom:
        @param torsion:
        @type atom: OEAtombase
        @type torsion: OETorsion
        @return: bool
        '''
        numRingAtoms = 0
        for a, b in zip(oechem.OEShortestPath(atom, torsion.b),
                        oechem.OEShortestPath(atom, torsion.c)):
            if a.IsInRing(): numRingAtoms += 1
            if b.IsInRing(): numRingAtoms += 1

        return numRingAtoms <= 4


    @staticmethod
    def GetNbrs(atomSet):
        '''
        Returns atoms connected to atomSet atoms,
        excluding those from atomSet
        @param atomSet: OEAtomBondSet
        @type atomSet: OEAtomBondSet
        @return: None
        '''
        allNbrs = []
        for atom in atomSet.GetAtoms():
            nbrs = atom.GetAtoms()
            for nbr in nbrs:
                if not atomSet.HasAtom(nbr):
                    allNbrs.append(nbr)

        return allNbrs

    @staticmethod
    def GetSameRingAtoms(mol, atomSet):
        '''
        @type mol: OEGraphMol
        @type atomSet: OEAtomBondSet
        @return list[OEAtombase]
        '''
        oechem.OEFindRingAtomsAndBonds(mol)
        _, ringIdxPerAtom = oechem.OEDetermineRingSystems(mol)
        toKeepRings = {}
        for atom in atomSet.GetAtoms():
            if ringIdxPerAtom[atom.GetIdx()] > 0:
                toKeepRings[ringIdxPerAtom[atom.GetIdx()]] = True

        ringAtoms = []
        for i in range(0, len(ringIdxPerAtom)):
            if ringIdxPerAtom[i] in toKeepRings:
                ringAtoms.append(mol.GetAtom(oechem.OEHasAtomIdx(i)))

        return ringAtoms

    @staticmethod
    def GetFuncGroups(mol):
        '''
        :param mol:
        :return:
        '''
        funcGrps = []
        for funcGrp in oemedchem.OEGetFuncGroupFragments(mol):
            if oechem.OECount(funcGrp, oechem.OEIsHeavy()) > 5:
                continue
            if oechem.OECount(funcGrp, oechem.OEIsHetero()) == 0:
                continue
            if oechem.OECount(funcGrp, oechem.OEAtomIsInRing()) > 0:
                continue

            funcGrps.append(oechem.OEAtomBondSet(funcGrp))

        return funcGrps


    @staticmethod
    def GetTorsions(mol):
        '''
        Goes through each rotatable bond in the molecule
        and extracts torsion atoms (a-b-c-d)
        Core torsion atoms are extended by one bond
        If core or extended atoms are part of a ring,
        then entire ring is kept
        Keep ortho substitution
        Keep functional groups that have at least one atom overlap
        with the core/extended torsion atoms
        Functional group inclusion criteria:
        - <= 5 heavy atoms
        - must contain at least one hetero atom
        - non-ring
        Add methyl cap if bond involving hetero atom is broken
        @param mol: OEGraphMol
        @type mol: OEGraphMol
        @return: list[OEGraphMol]
        '''
        # mol = OEGraphMol(input_mol)

        oechem.OEAssignHybridization(mol)
        funcGrps = TorsionGenerator.GetFuncGroups(mol)
        includedTorsions = oechem.OEAtomBondSet()
        torsionMols = []
        for atom in mol.GetAtoms():
            atom.SetData("idx", atom.GetIdx() + 1)

        torsions = get_canonical_torsions(mol)
        if torsions is None:
            torsions = oechem.OEGetTorsions(mol, oechem.OEIsRotor())

        for torsion in torsions:
            if torsion.a.IsHydrogen() or torsion.b.IsHydrogen() or \
                torsion.c.IsHydrogen() or torsion.d.IsHydrogen():
                continue

            torsion_bond = mol.GetBond(torsion.b, torsion.c)
            if includedTorsions.HasBond(torsion_bond):
                continue
            # if includedTorsions.HasAtom(torsion.b) and \
            #     includedTorsions.HasAtom(torsion.c):
            #     continue

            # revert map idx to zero in original mol
            for atom in mol.GetAtoms():
                atom.SetMapIdx(0)

            # includedTorsions.AddAtom(torsion.b)
            # includedTorsions.AddAtom(torsion.c)
            includedTorsions.AddBond(torsion_bond)

            torsionSet = oechem.OEAtomBondSet(mol.GetBonds())
            torsionSet.AddAtoms([torsion.a, torsion.b, torsion.c, torsion.d])
            for atom in torsionSet.GetAtoms():
                atom.SetMapIdx(1)

            # extend core torsion atoms by one bond
            nbrs = TorsionGenerator.GetNbrs(torsionSet)
            torsionSet.AddAtoms(nbrs)

            # include ring atoms
            ringAtoms = TorsionGenerator.GetSameRingAtoms(mol, torsionSet)
            torsionSet.AddAtoms(ringAtoms)

            for atom in torsionSet.GetAtoms():
                if not atom.GetMapIdx() == 1:
                    atom.SetMapIdx(2)

            # add functional groups that overlap with torsion set
            TorsionGenerator.AddFuncGroupAtoms(funcGrps, torsionSet)

            # add relevant ring atoms (ortho substituents and ring H)
            TorsionGenerator.AddRelevantRingAtoms(mol, torsion, torsionSet)

            # special treatment for C=O
            for atom in torsionSet.GetAtoms(oechem.OEAndAtom(
                    oechem.OEIsOxygen(), oechem.OEIsAtomHybridization(oechem.OEHybridization_sp2))):
                for nbr in atom.GetAtoms():
                    if torsionSet.HasAtom(nbr):
                        for nbr2 in nbr.GetAtoms(oechem.OEIsHeavy()):
                            if not torsionSet.HasAtom(nbr2):
                                nbr2.SetMapIdx(2)
                                torsionSet.AddAtom(nbr2)

            # mark bridging atom and cap if needed
            BRIDGE_ATOM_IDX = 4
            TorsionGenerator.MarkBridgingAtoms(BRIDGE_ATOM_IDX, mol, torsionSet)

            A_IDX = 11
            B_IDX = 12
            C_IDX = 13
            D_IDX = 14
            torsion.a.SetMapIdx(A_IDX)
            torsion.b.SetMapIdx(B_IDX)
            torsion.c.SetMapIdx(C_IDX)
            torsion.d.SetMapIdx(D_IDX)

            torsionMol = oechem.OEGraphMol()
            oechem.OESubsetMol(torsionMol, mol, torsionSet, True)
            torsionMol.Sweep()
            torsionMols.append(torsionMol)

            # change bridge atom to Carbon
            for atom in torsionMol.GetAtoms(oechem.OEHasMapIdx(BRIDGE_ATOM_IDX)):
                atom.SetAtomicNum(oechem.OEElemNo_C)
                explicit_valence = atom.GetExplicitValence()
                if explicit_valence < 4:
                    atom.SetImplicitHCount(4 - explicit_valence)

            TorsionGenerator.SetSDData(A_IDX, B_IDX, C_IDX, D_IDX,
                                       torsion, torsionMol)

            # set map idx to zero in torsion mol
            for atom in torsionMol.GetAtoms():
                atom.SetMapIdx(0)

        # revert map idx to zero in original mol
        for atom in mol.GetAtoms():
            atom.SetMapIdx(0)

        return torsionMols

    @staticmethod
    def SetSDData(A_IDX, B_IDX, C_IDX, D_IDX, torsion, torsionMol):
        taIdx = torsionMol.GetAtom(oechem.OEHasMapIdx(A_IDX)).GetIdx() + 1
        tbIdx = torsionMol.GetAtom(oechem.OEHasMapIdx(B_IDX)).GetIdx() + 1
        tcIdx = torsionMol.GetAtom(oechem.OEHasMapIdx(C_IDX)).GetIdx() + 1
        tdIdx = torsionMol.GetAtom(oechem.OEHasMapIdx(D_IDX)).GetIdx() + 1
        apStr = "cs1:0:1;1%{}:1%{}:1%{}:1%{}".format(taIdx, tbIdx, tcIdx, tdIdx)
        oechem.OESetSDData(torsionMol, "TORSION_ATOMPROP", apStr)
        fragTorAtoms = "{} {} {} {}".format(taIdx, tbIdx, tcIdx, tdIdx)
        oechem.OESetSDData(torsionMol, TORSION_ATOMS_FRAG_KEY, fragTorAtoms)
        parentTorAtoms = "{} {} {} {}".format(torsion.a.GetIdx() + 1,
                                              torsion.b.GetIdx() + 1,
                                              torsion.c.GetIdx() + 1,
                                              torsion.d.GetIdx() + 1)
        oechem.OESetSDData(torsionMol, TORSION_ATOMS_PARENT_MOL_KEY, parentTorAtoms)

        atom_map = ""
        for atom in torsionMol.GetAtoms():
            atom_map += str(atom.GetIdx() + 1) + "_" + str(atom.GetData("idx")) + "-"
        atom_map = atom_map[:-1]
        oechem.OESetSDData(torsionMol, FRAGMENT_TO_PARENT_ATOMS_KEY, atom_map)

    @staticmethod
    def MarkBridgingAtoms(BRIDGE_ATOM_IDX, mol, torsionSet):
        NorOorS = oechem.OEOrAtom(oechem.OEOrAtom(oechem.OEIsNitrogen(), oechem.OEIsOxygen()), oechem.OEIsSulfur())
        for atom in mol.GetAtoms(oechem.OEAndAtom(oechem.OEHasMapIdx(2), NorOorS)):
            for nbr in atom.GetAtoms(oechem.OEIsHeavy()):
                if not torsionSet.HasAtom(nbr):
                    if nbr.GetMapIdx() == 0:
                        torsionSet.AddAtom(nbr)
                        if nbr.GetHvyDegree() == 1:
                            nbr.SetMapIdx(3)
                            continue

                        nbr.SetMapIdx(BRIDGE_ATOM_IDX)

    @staticmethod
    def AddRelevantRingAtoms(mol, torsion, torsionSet):
        atom1or2 = oechem.OEOrAtom(oechem.OEHasMapIdx(1), oechem.OEHasMapIdx(2))
        ringNbrs = []
        for atom in mol.GetAtoms(oechem.OEAndAtom(oechem.OEAtomIsInRing(), atom1or2)):
            for nbr in atom.GetAtoms(oechem.OEAndAtom(
                    oechem.OENotAtom(atom1or2), oechem.OENotAtom(oechem.OEAtomIsInRing()))):
                if nbr.IsHydrogen():
                    ringNbrs.append(nbr)
                    continue

                if nbr.IsOxygen() and mol.GetBond(atom, nbr).GetOrder() == 2:
                    ringNbrs.append(nbr)
                    continue

                if TorsionGenerator.IsOrtho(nbr, torsion):
                    ringNbrs.append(nbr)
        for nbr in ringNbrs:
            if not torsionSet.HasAtom(nbr):
                nbr.SetMapIdx(2)
                torsionSet.AddAtom(nbr)

    @staticmethod
    def AddFuncGroupAtoms(funcGrps, torsionSet):
        addGrps = []
        for funcGrp in funcGrps:
            for atom in funcGrp.GetAtoms():
                if torsionSet.HasAtom(atom):
                    addGrps.append(funcGrp)
                    break
        for grp in addGrps:
            for atom in grp.GetAtoms():
                if not torsionSet.HasAtom(atom):
                    atom.SetMapIdx(2)
                    torsionSet.AddAtom(atom)


    @staticmethod
    def GetMinPathLength(refTorsion, adjTorsion):
        '''
        Returns path length between the two torsions
        @param refTorsion: OETorsion
        @param adjTorsion: OETorsion
        @return: int
        '''
        minPathLen = 1000
        for refAtom in [refTorsion.b, refTorsion.c]:
            for torAtom in [adjTorsion.b, adjTorsion.c]:
                pathLen = oechem.OEGetPathLength(refAtom, torAtom)
                if pathLen < minPathLen:
                    minPathLen = pathLen

        return minPathLen

    @staticmethod
    def GetAdjacentTorsions(mol, refTorsion):
        '''
        Returns all torsions that are 0 or 1 path length away from
        the reference torsion
        @param mol: OEGraphMol
        @param refTorsion: OETorsion
        @return: int
        '''
        adjTorsions = []
        PATH_LENGTH_THRESHOLD = 1
        torset = {str(refTorsion.b.GetIdx()) + "_" + str(refTorsion.c.GetIdx()):True}
        torset[str(refTorsion.c.GetIdx()) + "_" + str(refTorsion.b.GetIdx())] = True
        pred = oechem.OEAndBond(oechem.OEHasOrder(1), oechem.OENotBond(oechem.OEBondIsInRing()))
        for adjTorsion in oechem.OEGetTorsions(mol, pred):
            # skip nitrile
            order_ab = adjTorsion.a.GetBond(adjTorsion.b).GetOrder()
            order_cd = adjTorsion.c.GetBond(adjTorsion.d).GetOrder()
            if order_ab == 3 or order_cd == 3:
                continue

            # skip torsions involving terminal -N-H
            if adjTorsion.a.IsHydrogen() and adjTorsion.b.IsNitrogen():
                continue
            if adjTorsion.d.IsHydrogen() and adjTorsion.c.IsNitrogen():
                continue

            key1 = str(adjTorsion.b.GetIdx()) + "_" + str(adjTorsion.c.GetIdx())
            key2 = str(adjTorsion.c.GetIdx()) + "_" + str(adjTorsion.b.GetIdx())
            if key1 in torset or key2 in torset:
                continue

            pathLen = TorsionGenerator.GetMinPathLength(refTorsion, adjTorsion)
            if pathLen <= PATH_LENGTH_THRESHOLD:
                adjTorsions.append(adjTorsion)
                torset[key1] = True
                torset[key2] = True

        return adjTorsions


def get_molecule_torsion_fragments(mol):
    torgen = TorsionGenerator()
    tormols = torgen.GetTorsions(mol)

    ## Add missing hydrogens
    for tormol in tormols:
        oechem.OEAddExplicitHydrogens(tormol, False, True)

    return tormols


def get_canonical_torsions(mol):
    '''
    Return unique torsions in canonical order.
    Only one torsion containing the same central two atoms are return
    Cannonical ordering is determined using the order of atoms
    in canonical smiles representation
    1. generate a canonical smiles representation from the input molecule
    2. create a list of (min(b_idx, c_idx), min(a_idx, d_idx), max(a_idx, d_idx), OETorsion)
    3. sort the list in #2, extract subset with unique rotatable bonds
    :param mol: OEGraphMol
    :return: list[OEGraphMol]
    '''
    CANONICAL_IDX_TAG = 'can_idx'
    def assign_canonical_idx(mol):
        for atom in mol.GetAtoms():
            atom.SetMapIdx(0)
        for map_idx, atom in enumerate(mol.GetAtoms(oechem.OEIsHeavy())):
            atom.SetMapIdx(map_idx+1)

        can_smiles = oechem.OEMolToSmiles(mol)

        can_mol = oechem.OEGraphMol()
        # smiles_opt = OEParseSmilesOptions(canon=True)
        # OEParseSmiles(can_mol, can_smiles, smiles_opt)
        oechem.OESmilesToMol(can_mol, can_smiles)

        for can_atom in can_mol.GetAtoms(oechem.OEIsHeavy()):
            atom = mol.GetAtom(oechem.OEHasMapIdx(can_atom.GetMapIdx()))
            atom.SetData(CANONICAL_IDX_TAG, can_atom.GetIdx())


    try:
        assign_canonical_idx(mol)
    except Exception as e:
        print('Error GetCanonicalizedTorsions. ', e)
        return None

    torsions = []
    for torsion in oechem.OEGetTorsions(mol, oechem.OEIsRotor()):
        if torsion.a.IsHydrogen() or torsion.b.IsHydrogen() or \
            torsion.c.IsHydrogen() or torsion.d.IsHydrogen():
            continue

        sum_bc = torsion.b.GetData(CANONICAL_IDX_TAG) + torsion.c.GetData(CANONICAL_IDX_TAG)
        min_bc = min(torsion.b.GetData(CANONICAL_IDX_TAG),
                     torsion.c.GetData(CANONICAL_IDX_TAG))
        max_bc = max(torsion.b.GetData(CANONICAL_IDX_TAG),
                     torsion.c.GetData(CANONICAL_IDX_TAG))
        min_ad = min(torsion.a.GetData(CANONICAL_IDX_TAG),
                     torsion.d.GetData(CANONICAL_IDX_TAG))
        max_ad = max(torsion.a.GetData(CANONICAL_IDX_TAG),
                     torsion.d.GetData(CANONICAL_IDX_TAG))

        torsions.append((sum_bc, min_bc, max_bc, min_ad, max_ad, torsion))

    # sort
    torsions.sort(key = operator.itemgetter(0, 1, 2, 3, 4))

    seen = {}
    unique_torsions = []
    for _, _, _, _, _, torsion in torsions:
        bond = mol.GetBond(torsion.b, torsion.c)
        if bond is not None and bond.GetIdx() not in seen:
            unique_torsions.append(torsion)
            seen[bond.GetIdx()] = True

    # revert mol to original state
    for atom in mol.GetAtoms(oechem.OEIsHeavy()):
        atom.SetMapIdx(0)
        atom.DeleteData(CANONICAL_IDX_TAG)

    return unique_torsions


def extract_molecule_torsion_data(parent_mol, frag_mols = None):
    '''
    extract dihedral angle associated with each torsion motif in the input molecule
    Torsion motifs are represented using generic modified inchi (central two atoms)
    and specific modified inchi (4 torsion atoms)
    @param parent_mol:
    @type parent_mol: oechem.OEGraphMol
    @return: tuple(str, dict[str, list[float]])
    '''
    if frag_mols is None:
        frag_mols = get_molecule_torsion_fragments(parent_mol)

    torsion_data = collections.defaultdict(list)
    for frag_mol in frag_mols:
        #inchi_key = oechem.OECreateInChIKey(frag_mol)
        atom_map = get_fragment_to_parent_atom_mapping(parent_mol, frag_mol)

        try:
            _, b, c, _ = get_torsion_oeatom_list(frag_mol)

            for a in b.GetAtoms(oechem.OEIsHeavy()):
                for d in c.GetAtoms(oechem.OEIsHeavy()):
                    if a.GetIdx() == c.GetIdx() or d.GetIdx() == b.GetIdx():
                        continue

                    ap = atom_map[a]
                    bp = atom_map[b]
                    cp = atom_map[c]
                    dp = atom_map[d]

                    if a.GetAtomicNum() == ap.GetAtomicNum() \
                            and b.GetAtomicNum() == bp.GetAtomicNum() \
                            and c.GetAtomicNum() == cp.GetAtomicNum() \
                            and d.GetAtomicNum() == dp.GetAtomicNum():
                        angle = oechem.OEGetTorsion(parent_mol, ap, bp, cp, dp)*oechem.Rad2Deg
                        torsion_inchi = get_specific_dihedral_inchi_key(frag_mol, [a, b, c, d])

                        torsion_data[torsion_inchi].append((ap.GetIdx(), bp.GetIdx(), cp.GetIdx(), dp.GetIdx(), angle))

        except Exception as e:
            logging.warning(e)
            continue

    parent_inchi = get_modified_molecule_inchi(parent_mol)

    return (parent_inchi, torsion_data)


def get_fragment_to_parent_atom_mapping(parent_mol, frag_mol):
    try:
        mapping_data = oechem.OEGetSDData(frag_mol, FRAGMENT_TO_PARENT_ATOMS_KEY)
        idx_map = dict(map(int, idx_pair.split('_'))
                       for idx_pair in mapping_data.split('-'))
        atom_map = {}
        for frag_idx, parent_idx in idx_map.items():
            frag_atom = frag_mol.GetAtom(oechem.OEHasAtomIdx(frag_idx-1))
            parent_atom = parent_mol.GetAtom(oechem.OEHasAtomIdx(parent_idx-1))
            if frag_atom is not None and parent_atom is not None:
                atom_map[frag_atom] = parent_atom

        return atom_map

    except Exception as e:
        logging.warning(e)
        return {}


def get_torsion_oeatom_list(mol, tag=TORSION_ATOMS_FRAG_KEY):
    if has_sd_data(mol, tag):
        torsion_atoms = get_sd_data(mol, tag)
        try:
            torsion_atoms_idx = list(map(int, torsion_atoms.split()))
            torsion_oeatoms \
                = map(lambda idx: mol.GetAtom(oechem.OEHasAtomIdx(idx - 1)),
                      torsion_atoms_idx)
            return list(torsion_oeatoms)
        except Exception as e:
            print(e)
            return None