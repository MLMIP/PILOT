
aa2property = {'A':[1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
               'G':[0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
               'V':[3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
               'L':[2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
               'I':[4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
               'F':[2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
               'Y':[2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
               'W':[3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
               'T':[3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
               'S':[1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
               'R':[2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
               'K':[1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
               'H':[2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
               'D':[1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
               'E':[1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
               'N':[1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
               'Q':[1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
               'M':[2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
               'P':[2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
               'C':[1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41]}

atom2code_dist = {'C':[1, 0, 0, 0], 'N':[0, 1, 0, 0], 'O':[0, 0, 1, 0], 'S':[0, 0, 0, 1]}


def aa2code():
    aa2code = {}
    aa_name = ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H']
    for i in range(20):
        code = []
        for j in range(20):
            if i == j:
                code.append(1)
            else:
                code.append(0)
        aa2code[aa_name[i]] = code
    return aa2code


def feature_alignment(selected_res, selected_atom, feature_data, res_dict, atom_dict, pdb2uniprot_posdict,
                      mut_pos):


    res_feature_dict, atom_feature_dict = {}, {}

    sasa_res = feature_data['sasa_res']
    sasa_atom = feature_data['sasa_atom']
    ss_dict = feature_data['ss_dict']
    depth_dict = feature_data['depth_dict']
    pssm_dict = feature_data['pssm_dict']
    # res_dict = feature_data['res_dict']
    hhm_score = feature_data['hhm_score']
    conservation_dict = feature_data['conservation_dict']
    conservation_score = feature_data['conservation_score']

    aa2code_dict = aa2code()

    ############################### res features ###############################
    for pos in selected_res:
        try:
            res_name = res_dict[pos]

            aa_code = aa2code_dict[res_name]
            ss = ss_dict[pos]
            depth = [depth_dict[pos]]
            properties = aa2property[res_name]
            sasa = [sasa_res[pos]]

            uniprot_pos = str(pdb2uniprot_posdict[pos])
            pssm = pssm_dict[uniprot_pos]
            hhm = hhm_score[int(uniprot_pos) - 1]
            cs1 = conservation_dict[int(uniprot_pos)]
            cs2 = conservation_score[int(uniprot_pos) - 1]

            if pos == mut_pos:
                is_mut = [1]
            else:
                is_mut = [0]
        except:
            continue

        res_feature_dict[pos] = aa_code + ss + depth + properties + sasa + pssm + list(hhm) + list(cs1) + [
            cs2] +  is_mut

    ############################### atom features ###############################
    for atom_index in selected_atom.keys():
        try:
            atom_type = atom2code_dist[atom_dict[atom_index][0]]
            sasa = [sasa_atom[atom_index]]
        except:
            continue
        atom_feature_dict[atom_index] = atom_type + sasa

    return res_feature_dict, atom_feature_dict