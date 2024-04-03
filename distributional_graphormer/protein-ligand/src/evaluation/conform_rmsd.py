from pymol import cmd


def com(selection, state=None, mass=None, object=None, quiet=1, **kwargs):
    quiet = int(quiet)
    if (object == None):
        try:
            object = cmd.get_legal_name(selection)
            object = cmd.get_unused_name(object + "_COM", 0)
        except AttributeError:
            object = 'COM'
    cmd.delete(object)

    if (state != None):
        x, y, z = get_com(selection, mass=mass, quiet=quiet)
        if not quiet:
            print("%f %f %f" % (x, y, z))
        cmd.pseudoatom(object, pos=[x, y, z], **kwargs)
        cmd.show("spheres", object)
        return [x, y, z]
    else:
        for i in range(cmd.count_states()):
            x, y, z = get_com(selection, mass=mass, state=i + 1, quiet=quiet)
            if not quiet:
                print("State %d:%f %f %f" % (i + 1, x, y, z))
            cmd.pseudoatom(object, pos=[x, y, z], state=i + 1, **kwargs)
            cmd.show("spheres", 'last ' + object)
        return None


def get_com(selection, state=1, mass=None, quiet=1):
    quiet = int(quiet)

    totmass = 0.0
    if mass != None and not quiet:
        print("Calculating mass-weighted COM")

    state = int(state)
    model = cmd.get_model(selection, state)
    x, y, z = 0, 0, 0
    for a in model.atom:
        if (mass != None):
            m = a.get_mass()
            x += a.coord[0] * m
            y += a.coord[1] * m
            z += a.coord[2] * m
            totmass += m
        else:
            x += a.coord[0]
            y += a.coord[1]
            z += a.coord[2]

    if (mass != None):
        return x / totmass, y / totmass, z / totmass
    else:
        return x / len(model.atom), y / len(model.atom), z / len(model.atom)


def get_full_rmsd(mol_pdb1,mol_pdb2):
    cmd.reinitialize()

    cmd.load(f"{str(mol_pdb1)}", "mol1")
    cmd.load(f"{str(mol_pdb2)}", "mol2")
    # remove hydrogens
    cmd.remove("hydrogens")
    mol1_center = com("mol1", state=1, object='COG')
    mol2_center = com("mol2", state=1, object='COG')

    # move mol2 to the mol1_ceter using the center of geometry
    cmd.translate([mol1_center[0] - mol2_center[0], mol1_center[1] - mol2_center[1], mol1_center[2] - mol2_center[2]], "mol2")

    # align mol1, mol2, cycles=0, transform=0, object=aln
    cmd.align("mol1", "mol2", cycles=0, transform=0, object='aln')
    # rms_cur mol1 & aln, mol2 & aln, matchmaker=-1
    rmsd = cmd.rms_cur("mol1 & aln", "mol2 & aln", matchmaker=-1)
    return rmsd

def get_conf_rmsd(lig_pdb1,lig_pdb2):
    cmd.reinitialize()
    cmd.load(f"{str(lig_pdb1)}", "pro")
    cmd.load(f"{str(lig_pdb2)}", "lig")
    return cmd.align("pro", "lig", cycles=0)[0]