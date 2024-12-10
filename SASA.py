import os
import subprocess



def use_naccess(pdb_file, sasa_path, naccess_path):
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    if not os.path.exists(os.path.join(sasa_path, pdb_name +'.rsa')):
        os.chdir(sasa_path)
        _, _ = subprocess.Popen([naccess_path, pdb_file, '-h'], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE).communicate()
        os.system(f'mv {sasa_path}/5.rsa {pdb_name}.rsa')
        os.system(f'mv {sasa_path}/5.asa {pdb_name}.asa')

    return f'{sasa_path}/{pdb_name}.rsa', f'{sasa_path}/{pdb_name}.asa'

def calc_SASA(rsa_file, asa_file):
    res_naccess_output, atom_naccess_output = [], []
    res_sasa_dict, atom_sasa_dict = {}, {}

    res_naccess_output += open(rsa_file, 'r').readlines()
    atom_naccess_output += open(asa_file, 'r').readlines()

    for res_info in res_naccess_output:
        if res_info[0:3] == 'RES':
            residue_index = res_info[9:14].strip()
            relative_perc_accessible = float(res_info[22:28])
            res_sasa_dict[residue_index] = relative_perc_accessible

    # 注意原子的位置也要对应起来
    for atom_info in atom_naccess_output:
        if atom_info[0:4] == 'ATOM':
            atom_index = atom_info[6:11].strip()
            relative_perc_accessible = atom_info[54:62].strip()
            atom_sasa_dict[atom_index] = relative_perc_accessible

    return res_sasa_dict, atom_sasa_dict