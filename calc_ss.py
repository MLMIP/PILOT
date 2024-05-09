from Bio.PDB.DSSP import dssp_dict_from_pdb_file


def ss8to3(ss):
    if ss == 'H' or ss == 'G' or ss == 'I':
        return [1, 0, 0]
    elif ss == 'B' or ss == 'E':
        return [0, 1, 0]
    else:
        assert ss == 'T' or ss == 'S' or ss == '-'
        return [0, 0, 1]


def calc_ss(pdb_file, dssp_path):
    try:
        dssp_tuple = dssp_dict_from_pdb_file(pdb_file, DSSP=dssp_path)
    except Exception:
        return False
    dssp_dict = dssp_tuple[0]
    ss_dict = {}
    for (key, value) in dssp_dict.items():
        index = (str(key[1][1]) + str(key[1][2])).strip()
        ss = ss8to3(value[1])
        ss_dict[index] = ss
    # print(ss_dict)
    return ss_dict