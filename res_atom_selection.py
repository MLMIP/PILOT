import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools
from Bio.PDB.Polypeptide import three_to_one

pdb_dir = '/storage3/database/datasets/dataset_ddG/5.0/all_pdb'



NON_STANDARD_SUBSTITUTIONS = {
    'GLH': 'GLU', 'ASH': 'ASP', 'CYX': 'CYS', 'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS', '2AS': 'ASP', '3AH': 'HIS',
    '5HP': 'GLU', 'ACL': 'ARG', 'AGM': 'ARG', 'AIB': 'ALA', 'ALM': 'ALA', 'ALO': 'THR', 'ALY': 'LYS', 'ARM': 'ARG',
    'ASA': 'ASP', 'ASB': 'ASP', 'ASK': 'ASP', 'ASL': 'ASP', 'ASQ': 'ASP', 'AYA': 'ALA', 'BCS': 'CYS', 'BHD': 'ASP',
    'BMT': 'THR', 'BNN': 'ALA', 'BUC': 'CYS', 'BUG': 'LEU', 'C5C': 'CYS', 'C6C': 'CYS', 'CAS': 'CYS', 'CCS': 'CYS',
    'CEA': 'CYS', 'CGU': 'GLU', 'CHG': 'ALA', 'CLE': 'LEU', 'CME': 'CYS', 'CSD': 'ALA', 'CSO': 'CYS', 'CSP': 'CYS',
    'CSS': 'CYS', 'CSW': 'CYS', 'CSX': 'CYS', 'CXM': 'MET', 'CY1': 'CYS', 'CY3': 'CYS', 'CYG': 'CYS', 'CYM': 'CYS',
    'CYQ': 'CYS', 'DAH': 'PHE', 'DAL': 'ALA', 'DAR': 'ARG', 'DAS': 'ASP', 'DCY': 'CYS', 'DGL': 'GLU', 'DGN': 'GLN',
    'DHA': 'ALA', 'DHI': 'HIS', 'DIL': 'ILE', 'DIV': 'VAL', 'DLE': 'LEU', 'DLY': 'LYS', 'DNP': 'ALA', 'DPN': 'PHE',
    'DPR': 'PRO', 'DSN': 'SER', 'DSP': 'ASP', 'DTH': 'THR', 'DTR': 'TRP', 'DTY': 'TYR', 'DVA': 'VAL', 'EFC': 'CYS',
    'FLA': 'ALA', 'FME': 'MET', 'GGL': 'GLU', 'GL3': 'GLY', 'GLZ': 'GLY', 'GMA': 'GLU', 'GSC': 'GLY', 'HAC': 'ALA',
    'HAR': 'ARG', 'HIC': 'HIS', 'HIP': 'HIS', 'HMR': 'ARG', 'HPQ': 'PHE', 'HTR': 'TRP', 'HYP': 'PRO', 'IAS': 'ASP',
    'IIL': 'ILE', 'IYR': 'TYR', 'KCX': 'LYS', 'LLP': 'LYS', 'LLY': 'LYS', 'LTR': 'TRP', 'LYM': 'LYS', 'LYZ': 'LYS',
    'MAA': 'ALA', 'MEN': 'ASN', 'MHS': 'HIS', 'MIS': 'SER', 'MLE': 'LEU', 'MPQ': 'GLY', 'MSA': 'GLY', 'MSE': 'MET',
    'MVA': 'VAL', 'NEM': 'HIS', 'NEP': 'HIS', 'NLE': 'LEU', 'NLN': 'LEU', 'NLP': 'LEU', 'NMC': 'GLY', 'OAS': 'SER',
    'OCS': 'CYS', 'OMT': 'MET', 'PAQ': 'TYR', 'PCA': 'GLU', 'PEC': 'CYS', 'PHI': 'PHE', 'PHL': 'PHE', 'PR3': 'CYS',
    'PRR': 'ALA', 'PTR': 'TYR', 'PYX': 'CYS', 'SAC': 'SER', 'SAR': 'GLY', 'SCH': 'CYS', 'SCS': 'CYS', 'SCY': 'CYS',
    'SEL': 'SER', 'SEP': 'SER', 'SET': 'SER', 'SHC': 'CYS', 'SHR': 'LYS', 'SMC': 'CYS', 'SOC': 'CYS', 'STY': 'TYR',
    'SVA': 'SER', 'TIH': 'ALA', 'TPL': 'TRP', 'TPO': 'THR', 'TPQ': 'ALA', 'TRG': 'LYS', 'TRO': 'TRP', 'TYB': 'TYR',
    'TYI': 'TYR', 'TYQ': 'TYR', 'TYS': 'TYR', 'TYY': 'TYR'
}


def get_pdb_array(pdb_file, chain_id):
    pdb_array = []
    with open(pdb_file, 'r') as pdbfile:
        for line in pdbfile:
            if line[0:4] == 'ATOM' or line[0:4] == 'HETA':
                line_list = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27],
                                line[30:38],
                                line[38:46], line[46:54]]
                line_list = [i.strip() for i in line_list]
                if (line_list[0] == 'ATOM' or line_list[0] == 'HETAT') and line_list[5] == chain_id and line_list[4] != 'HOH':
                    pdb_array.append(line_list)
    return np.array(pdb_array, dtype='str')



def get_residue_info(pdb_array):
    atom_res_array = pdb_array[:, 6]  # 每一个原子对应的氨基酸编号
    # print(atom_res_array)
    boundary_list = []  # 列表中代表每一个氨基酸的起始原子和终止原子的位置
    pdb_pos_list = []
    start_pointer = 0
    curr_pointer = 0
    curr_atom = atom_res_array[0]

    # One pass through the list of residue numbers and record row number boundaries. Both sides inclusive.
    while (curr_pointer < atom_res_array.shape[0] - 1):
        curr_pointer += 1
        if atom_res_array[curr_pointer] != curr_atom:
            pdb_pos_list.append(curr_atom)
            boundary_list.append([start_pointer, curr_pointer - 1])
            start_pointer = curr_pointer
            curr_atom = atom_res_array[curr_pointer]
    boundary_list.append([start_pointer, atom_res_array.shape[0] - 1])
    pdb_pos_list.append(curr_atom)
    return np.array(boundary_list), pdb_pos_list


def get_residue_distance_matrix(pdb_array, residue_index, distance_type):
    if distance_type == 'c_alpha':
        coord_array = np.empty((residue_index.shape[0], 3))
        for i in range(residue_index.shape[0]):
            # 获取每一个氨基酸中α碳原子的位置，若该氨基酸中有α碳原子，则取其位置，否则，则取该氨基酸中所有原子的平均值
            res_start, res_end = residue_index[i]
            flag = False
            res_array = pdb_array[res_start:res_end + 1]
            for j in range(res_array.shape[0]):
                if res_array[j][2] == 'CA':
                    coord_array[i] = res_array[:, 7:10][j].astype(np.float64)
                    flag = True
                    break
            if not flag:
                coord_i = pdb_array[:, 7:10][res_start:res_end + 1].astype(np.float64)
                coord_array[i] = np.mean(coord_i, axis=0)
        residue_dm = squareform(pdist(coord_array))
    elif distance_type == 'centroid':
        coord_array = np.empty((residue_index.shape[0], 3))
        for i in range(residue_index.shape[0]):
            res_start, res_end = residue_index[i]
            coord_i = pdb_array[:, 7:10][res_start:res_end + 1].astype(np.float64)
            coord_array[i] = np.mean(coord_i, axis=0)
        residue_dm = squareform(pdist(coord_array))
    elif distance_type == 'atoms_average':
        full_atom_dist = squareform(pdist(pdb_array[:, 7:10].astype(float)))
        residue_dm = np.zeros((residue_index.shape[0], residue_index.shape[0]))
        for i, j in itertools.combinations(range(residue_index.shape[0]), 2):
            index_i = residue_index[i]
            index_j = residue_index[j]
            distance_ij = np.mean(full_atom_dist[index_i[0]:index_i[1] + 1, index_j[0]:index_j[1] + 1])
            residue_dm[i][j] = distance_ij
            residue_dm[j][i] = distance_ij
    else:
        raise ValueError('Invalid distance type: %s' % distance_type)
    return residue_dm


def get_nearest_resindex(pdb_file, chain_id, mut_pos, aa_num=16):
    selectes_pdb_pos = []
    atompos2index = {}
    new_pdb_array = []
    pdb_array = get_pdb_array(pdb_file, chain_id)
    residue_index, pdb_pos_list = get_residue_info(pdb_array)
    residue_dm = get_residue_distance_matrix(pdb_array, residue_index, 'c_alpha')
    index = residue_dm.argsort()[pdb_pos_list.index(mut_pos), :aa_num]

    for i in index:
        selectes_pdb_pos.append(pdb_pos_list[i])

    index = 0
    for atom_info in pdb_array:
        if atom_info[2][0] == 'C' or atom_info[2][0] == 'N' or atom_info[2][0] == 'O' or atom_info[2][0] == 'S':
            if atom_info[6] in selectes_pdb_pos:
                atompos2index[atom_info[1]] = index
                new_pdb_array.append(list(atom_info))
                index += 1

    return selectes_pdb_pos, atompos2index, np.array(new_pdb_array, dtype='str')


def get_res_atom_dict(pdb_array):
    res_dict, atom_dict = {}, {}
    for atom_info in pdb_array:
        res_name = atom_info[4]
        if atom_info[4] in NON_STANDARD_SUBSTITUTIONS:
            res_name = NON_STANDARD_SUBSTITUTIONS[res_name]
        try:
            aa = three_to_one(res_name)
        except KeyError:
            continue
        res_dict[atom_info[6]] = aa
        atom_dict[atom_info[1]] = atom_info[2]

    return res_dict, atom_dict