import logging
from openeye import oechem, oeszybki, oeomega
from torsion.utils.process_sd_data import get_sd_data, has_sd_data
from torsion.dihedral import get_dihedral


TORSION_LIBRARY = [
            '[C,N,c:1][NX3:2][C:3](=[O])[C,N,c,O:4] 0 180',   # amides are flipped cis and trans
            '[#1:1][NX3H:2][C:3](=[O])[C,N,c,O:4] 0',  # primary amides are NOT flipped
            '[*:1][C,c:2][OX2:3][*:4] 0 180',           # hydroxyls and ethers are rotated 180 degrees
            '[H:1][CH3:2]-!@[!#1:3][*:4] 0.1 180',     # methyls are rotated 180 degrees
            '[H:1][CH3:2]-!@[!#1:3]=[*:4] 0.1 180'
            ]

MAX_CONFS = 100

class hasDoubleBondO(oechem.OEUnaryAtomPred):
    def __call__(self, atom):
        for bond in atom.GetBonds():
            if bond.GetOrder() == 2 and bond.GetNbr(atom).IsOxygen():
                return True
        return False

def isAmideRotor(bond):
    if bond.GetOrder() != 1:
        return False
    atomB = bond.GetBgn()
    atomE = bond.GetEnd()
    pred = hasDoubleBondO()
    if atomB.IsCarbon() and atomE.IsNitrogen() and pred(atomB):
        return True
    if atomB.IsNitrogen() and atomE.IsCarbon() and pred(atomE):
        return True
    return False

def isMethylRotor(bond):
    if bond.GetOrder() != 1:
        return False
    atomB = bond.GetBgn()
    atomE = bond.GetEnd()

    if atomB.IsHydrogen() or atomE.IsHydrogen():
        return False

    def isMethylCarbon(atom):
        return atom.GetAtomicNum() == oechem.OEElemNo_C and \
               atom.GetHvyDegree() == 1 and \
               atom.GetTotalHCount() == 3

    return isMethylCarbon(atomB) or isMethylCarbon(atomE)

def isEtherRotor(bond):
    if bond.GetOrder() != 1:
        return False
    atomB = bond.GetBgn()
    atomE = bond.GetEnd()
    
    isEtherOxygen = oechem.OEMatchAtom("[OX2][C,c]")
    return (atomB.IsCarbon() and isEtherOxygen(atomE)) or (atomE.IsCarbon() and isEtherOxygen(atomB))


def isRotatableBond(bond):
    inRing = oechem.OEBondIsInRing()

    return (not inRing(bond)) and (
            isAmideRotor(bond) or \
            isMethylRotor(bond) or \
            isEtherRotor(bond)
            )


class distance_predicate(oechem.OEUnaryBondPred):
    def __init__(self, atom1_idx, atom2_idx):
        oechem.OEUnaryBondPred.__init__(self)
        self.atom1_idx = atom1_idx
        self.atom2_idx = atom2_idx

    def __call__(self, bond):
        atomB = bond.GetBgn()
        atomE = bond.GetEnd()
        mol = bond.GetParent()
        atom1 = mol.GetAtom(oechem.OEHasAtomIdx(self.atom1_idx))
        atom2 = mol.GetAtom(oechem.OEHasAtomIdx(self.atom2_idx))
        return max(oechem.OEGetPathLength(atomB, atom1),
                   oechem.OEGetPathLength(atomE, atom1),
                   oechem.OEGetPathLength(atomB, atom2),
                   oechem.OEGetPathLength(atomE, atom2)) <= 3

def configure_omega(library, rotor_predicate, rms_cutoff, energy_window, num_conformers=MAX_CONFS):
    opts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
    opts.SetEnumRing(False)
    opts.SetEnumNitrogen(oeomega.OENitrogenEnumeration_Off)
    opts.SetSampleHydrogens(True)
    opts.SetRotorPredicate(rotor_predicate)
    opts.SetIncludeInput(False)

    opts.SetEnergyWindow(energy_window)
    opts.SetMaxConfs(num_conformers)
    opts.SetRMSThreshold(rms_cutoff)
    
    conf_sampler = oeomega.OEOmega(opts)
    conf_sampler.SetCanonOrder(False)
    torlib = conf_sampler.GetTorLib()

# torlib.ClearTorsionLibrary()
    for rule in library:
        if not torlib.AddTorsionRule(rule):
            oechem.OEThrow.Fatal('Failed to add torsion rule: {}'.format(rule))
    conf_sampler.SetTorLib(torlib)

    return conf_sampler

def gen_starting_confs(mol, torsion_library, num_conformers=MAX_CONFS, rms_cutoff=0.0, energy_window=25):

    # Identify the atoms in the dihedral
    TAGNAME = 'TORSION_ATOMS_FRAGMENT'
    if not has_sd_data(mol, TAGNAME):
        raise ValueError("Molecule does not have the SD Data Tag '{}'.".format(TAGNAME))

    dihedralAtomIndices = [int(x)-1 for x in get_sd_data(mol, TAGNAME).split()]
    inDih = \
    oechem.OEOrAtom(oechem.OEOrAtom(oechem.OEHasAtomIdx(dihedralAtomIndices[0]), 
                                    oechem.OEHasAtomIdx(dihedralAtomIndices[1])),
                    oechem.OEOrAtom(oechem.OEHasAtomIdx(dihedralAtomIndices[2]), 
                                    oechem.OEHasAtomIdx(dihedralAtomIndices[3]))
                   )

    mol1 = mol.CreateCopy()
    mc_mol = oechem.OEMol(mol1)

    if num_conformers > 1:
        rotor_predicate = oechem.OEOrBond(oechem.OEIsRotor(),
                                            oechem.PyBondPredicate(isRotatableBond))
        
        #Initialize conformer generator and multi-conformer library
        conf_generator = configure_omega(torsion_library, rotor_predicate,
                                         rms_cutoff, energy_window, num_conformers)

        # Generator conformers
        if not conf_generator(mc_mol, inDih):
            raise ValueError("Conformers cannot be generated.")
        logging.debug("Generated a total of %d conformers for %s.", mc_mol.NumConfs(),
                                                                    mol.GetTitle())

    for conf_no, conf in enumerate(mc_mol.GetConfs()):
        conformer_label = mol.GetTitle()+'_' +\
                         '_'.join(get_sd_data(mol, 'TORSION_ATOMS_ParentMol').split()) +\
                         '_{:02d}'.format(conf_no)
        oechem.OESetSDData(conf, "CONFORMER_LABEL", conformer_label)
        conf.SetTitle(conformer_label)

    return mc_mol


def get_best_conf(mol, dih, num_points):
    """Drive the primary torsion in the molecule and select the lowest 
       energy conformer to represent each dihedral angle
    """
    delta = 360.0/num_points
    angle_list = [2*i*oechem.Pi/num_points for i in range(num_points)]

    dih_atoms = [x for x in dih.GetAtoms()]

    # Create new output OEMol
    title = mol.GetTitle()
    tor_mol = oechem.OEMol()

    opts = oeszybki.OETorsionScanOptions()
    opts.SetDelta(delta)
    opts.SetForceFieldType(oeszybki.OEForceFieldType_MMFF94)
    opts.SetSolvationType(oeszybki.OESolventModel_NoSolv)
    tmp_angle = 0.0
    tor = oechem.OETorsion(dih_atoms[0], dih_atoms[1], dih_atoms[2], dih_atoms[3], tmp_angle)

    oeszybki.OETorsionScan(tor_mol, mol, tor, opts)
    oechem.OECopySDData(tor_mol, mol)

    # if 0 and 360 sampled because of rounding 
    if tor_mol.NumConfs() > num_points:
        for conf in tor_mol.GetConfs():
            continue
        tor_mol.DeleteConf(conf)

    for angle, conf in zip(angle_list, tor_mol.GetConfs()):
        angle_deg = int(round(angle*oechem.Rad2Deg))
        tor_mol.SetActive(conf)
        oechem.OESetTorsion(conf, dih_atoms[0], dih_atoms[1], dih_atoms[2], dih_atoms[3], angle)

        conf_name = title + '_{:02d}'.format(conf.GetIdx())
        oechem.OESetSDData(conf, 'CONFORMER_LABEL', conf_name)
        oechem.OESetSDData(conf, 'TORSION_ANGLE', "{:.0f}".format(angle_deg))
        conf.SetDoubleData('TORSION_ANGLE', angle_deg)
        conf.SetTitle('{}: Angle {:.0f}'.format(conf_name, angle_deg))

    return tor_mol 


def get_torsional_confs(mol):
    mc_mol = gen_starting_confs(mol, TORSION_LIBRARY, True, 20)
    torsion_tag = 'TORSION_ATOMS_FRAGMENT'
    torsion_atoms_in_fragment = get_sd_data(mol, torsion_tag).split()
    dihedral_atom_indices = [int(x) - 1 for x in torsion_atoms_in_fragment]
    dih, _ = get_dihedral(mc_mol, dihedral_atom_indices)
    torsional_confs = get_best_conf(mc_mol, dih, 24)
    torsional_mols = []
    for conf in torsional_confs.GetConfs():
        new_mol = oechem.OEMol(conf)
        oechem.OECopySDData(new_mol, mol)
        torsional_mols.append(new_mol)
    return torsional_mols
