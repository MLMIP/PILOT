import os
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import get_surface, residue_depth

def calc_depth(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('PDB', pdb_file)
    model = struct[0]
    chain = model[chain_id]
    depth_dict = {}
    surface = get_surface(chain)
    for res in chain:
        res_id = res.get_id()
        pos = str(res_id[1]).strip() + str(res_id[2]).strip()
        depth = residue_depth(res, surface)
        depth_dict[pos] = depth
    return depth_dict