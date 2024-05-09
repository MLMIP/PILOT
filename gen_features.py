import os
import numpy as np
import torch
from feature_generators.gen_pdb import gen_all_pdb
from feature_generators.gen_fasta import gen_all_fasta
from feature_generators.SASA import use_naccess, calc_SASA
from feature_generators.evol_info import use_psiblast, use_hhblits, gen_msa, get_pssm, process_hhm, calc_res_freq
from feature_generators.use_esm2 import use_esm2
from feature_generators.calc_ss import calc_ss
from feature_generators.calc_depth import calc_depth
from feature_generators.calc_cs import calculate_conservation_score, blosum_background_distr
from feature_generators.res_atom_selection import get_nearest_resindex, get_res_atom_dict
from feature_generators.feature_alignment import feature_alignment
from feature_generators.get_edge import get_edge



foldx_path = './software/foldx'
naccess_path = './software/naccess2.1.1/naccess'
dssp_path = '../anaconda3/envs/pilot/bin/mkdssp'
psi_path = './software/ncbi-blast-2.13.0+/bin/psiblast'
clustalo_path = './software/clustalo'
hhblits_path = './software/hh-suite/build/bin/hhblits'
uniref90_path = './software/database/uniref90/uniref90'
uniRef30_path = './software/database/uniref30/UniRef30_2022_02'


def get_new_pdb_array(pdb_array, res_pdbpos, atom_indexpos):
    new_pdb_array = []
    res_index_pos_dict, atom_index_pos = {}, {}
    res_pos = -10
    res_index, atom_index = 0, 0
    for atom_info in pdb_array:
        if (atom_info[6] in res_pdbpos) and (atom_info[1] in atom_indexpos):
            new_pdb_array.append(atom_info)
            atom_index_pos[atom_index] = atom_info[1]
            atom_index += 1
            if atom_info[6] != res_pos:
                res_index_pos_dict[res_index] = atom_info[6]
                res_index += 1
                res_pos = atom_info[6]
    return np.array(new_pdb_array, dtype='str'), res_index_pos_dict, atom_index_pos


def generate_node_feature(feature_dict, index_pos_dict, feature_num):
    residue_node_feature = np.empty((len(index_pos_dict), feature_num), dtype=np.float64)
    all_pdbpos = index_pos_dict.values()
    for i, pdbpos in enumerate(all_pdbpos):
        residue_node_feature[i] = [float(feature) for feature in feature_dict[pdbpos]]
    return residue_node_feature


def get_esm2(esm_name, index_pos_dict, pdb2uniprot_posdict, esm_dir):
    extra_feature = np.empty((len(index_pos_dict), 1280), dtype=np.float64)
    esm2_dir = f'{esm_dir}/{esm_name}.pt'
    esm2 = torch.load(esm2_dir)
    all_pdbpos = index_pos_dict.values()
    for i, pdbpos in enumerate(all_pdbpos):
        uniprot_pos = pdb2uniprot_posdict[pdbpos]
        extra_feature[i] = [float(feature) for feature in esm2[int(uniprot_pos)-1]]
    return extra_feature




def gen_features(pdb_id, chain_id, mut_pos, wild_type, mutant, dir, seq_dict):

    # Create folder
    row_pdb_dir = f'{dir}/row_pdb'
    cleaned_pdb_dir = f'{dir}/cleaned_pdb'
    fasta_dir = f'{dir}/fasta'
    pssm_dir = f'{dir}/rawmsa_pssm'
    msa_dir = f'{dir}/msa'
    hhm_dir = f'{dir}/hhm'
    sasa_dir = f'{dir}/sasa'
    esm_dir = f'{dir}/esm'
    input_dir = f'{dir}/input'
    if not os.path.exists(row_pdb_dir):
        os.mkdir(row_pdb_dir)
    if not os.path.exists(cleaned_pdb_dir):
        os.mkdir(cleaned_pdb_dir)
    if not os.path.exists(fasta_dir):
        os.mkdir(fasta_dir)
    if not os.path.exists(pssm_dir):
        os.mkdir(pssm_dir)
    if not os.path.exists(msa_dir):
        os.mkdir(msa_dir)
    if not os.path.exists(hhm_dir):
        os.mkdir(hhm_dir)
    if not os.path.exists(sasa_dir):
        os.mkdir(sasa_dir)
    if not os.path.exists(esm_dir):
        os.mkdir(esm_dir)
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    pdb_chain = pdb_id+'_'+chain_id
    mut_id = pdb_id+'_'+chain_id+'_'+wild_type+mut_pos+mutant
    print('--------------------------------------')
    print(f'Predicting mutation {mut_id}.')
    print('--------------------------------------')

    # Generate all PDB and fasta files
    print(f'Generating PDB and fasta files ...')
    wild_pdb, mut_pdb = gen_all_pdb(pdb_id, chain_id, mut_pos, wild_type, mutant, row_pdb_dir, cleaned_pdb_dir, foldx_path)
    wild_fasta, mut_fasta, wild_seq, mut_seq, pdb_positions, pdbpos2uniprotpos_dict = \
        gen_all_fasta(pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir, fasta_dir)

    # Generate all basic feature files
    print(f'Using naccess ...')
    wild_frsa, wild_fasa = use_naccess(wild_pdb, sasa_dir, naccess_path)
    mut_frsa, mut_fasa = use_naccess(mut_pdb, sasa_dir, naccess_path)

    print(f'Using psiblast ...')
    wild_frawmsa, wild_fpssm = use_psiblast(wild_fasta, pssm_dir, psi_path, uniref90_path)
    mut_frawmsa, mut_fpssm = use_psiblast(mut_fasta, pssm_dir, psi_path, uniref90_path)

    print(f'Generating MSA files ...')
    wild_fmsa = gen_msa(pdb_chain, wild_seq, wild_frawmsa, seq_dict, msa_dir, clustalo_path)
    mut_fmsa = gen_msa(mut_id, mut_seq, mut_frawmsa, seq_dict, msa_dir, clustalo_path)

    print(f'Using hhblits ...')
    wild_fhhm = use_hhblits(pdb_chain, wild_fasta, hhblits_path, uniRef30_path, hhm_dir)
    mut_fhhm = use_hhblits(mut_id, mut_fasta, hhblits_path, uniRef30_path, hhm_dir)

    print(f'Using esm2 ...')
    use_esm2(wild_fasta, pdb_chain, esm_dir)
    use_esm2(mut_fasta, mut_id, esm_dir)


    # Calculation of all features
    print('Calculating all features ...')
    wild_rsasa, wild_asasa = calc_SASA(wild_frsa, wild_fasa)
    mut_rsasa, mut_asasa = calc_SASA(mut_frsa, mut_fasa)

    wild_ss = calc_ss(wild_pdb, dssp_path)
    mut_ss = calc_ss(mut_pdb, dssp_path)

    wild_depth = calc_depth(wild_pdb, chain_id)
    mut_depth = calc_depth(mut_pdb, chain_id)

    wild_pssm, wild_res_dict = get_pssm(wild_fpssm)
    mut_pssm, mut_res_dict = get_pssm(mut_fpssm)

    wild_hhm = process_hhm(wild_fhhm)
    mut_hhm = process_hhm(mut_fhhm)

    wild_res_freq = calc_res_freq(wild_fmsa)
    mut_res_freq = calc_res_freq(mut_fmsa)

    wild_cs = calculate_conservation_score().calculate_js_div_from_msa(wild_fmsa, blosum_background_distr)
    mut_cs = calculate_conservation_score().calculate_js_div_from_msa(mut_fmsa, blosum_background_distr)

    wild_data = {
        'sasa_res':wild_rsasa,
        'sasa_atom':wild_asasa,
        'ss_dict':wild_ss,
        'depth_dict':wild_depth,
        'pssm_dict':wild_pssm,
        'res_dict':wild_res_dict,
        'hhm_score':wild_hhm,
        'conservation_dict':wild_res_freq,
        'conservation_score':wild_cs
    }

    mut_data = {
        'sasa_res':mut_rsasa,
        'sasa_atom':mut_asasa,
        'ss_dict':mut_ss,
        'depth_dict':mut_depth,
        'pssm_dict':mut_pssm,
        'res_dict':mut_res_dict,
        'hhm_score':mut_hhm,
        'conservation_dict':mut_res_freq,
        'conservation_score':mut_cs
    }

    # Select the atom and amino acid nearest to the mutation site
    wild_selected_res, wild_selected_atom, wild_pdb_array = get_nearest_resindex(wild_pdb, chain_id, mut_pos,
                                                                           aa_num=16)
    mut_selected_res, mut_selected_atom, mut_pdb_array = get_nearest_resindex(mut_pdb, chain_id, mut_pos,
                                                                           aa_num=16)
    wild_res_dict, wild_atom_dict = get_res_atom_dict(wild_pdb_array)
    mut_res_dict, mut_atom_dict = get_res_atom_dict(mut_pdb_array)

    # Obtain the features corresponding to selected atoms and amino acids
    wild_res_feat_dict, wild_atom_feat_dict = feature_alignment(wild_selected_res, wild_selected_atom,
                                                             wild_data, wild_res_dict, wild_atom_dict,
                                                             pdbpos2uniprotpos_dict, mut_pos)
    mut_res_feat_dict, mut_atom_feat_dict = feature_alignment(mut_selected_res, mut_selected_atom,
                                                             mut_data, mut_res_dict, mut_atom_dict,
                                                             pdbpos2uniprotpos_dict, mut_pos)

    wild_pdb_array, wild_res_index_pos_dict, wild_atom_index_pos = get_new_pdb_array(wild_pdb_array,
                                                                               wild_res_feat_dict.keys(),
                                                                               wild_atom_feat_dict.keys())
    mut_pdb_array, mut_res_index_pos_dict, mut_atom_index_pos = get_new_pdb_array(mut_pdb_array,
                                                                               mut_res_feat_dict.keys(),
                                                                               mut_atom_feat_dict.keys())

    # Calculate the features of the edge
    wild_res_edge_index, wild_res_edge_feature, wild_atom_res_index = get_edge().generate_res_edge_feature_postmut(
        mut_pos, wild_pdb_array)
    mut_res_edge_index, mut_res_edge_feature, mut_atom_res_index = get_edge().generate_res_edge_feature_postmut(
        mut_pos, mut_pdb_array)

    wild_atom_edge_index, wild_atom_edge_feature = get_edge().generate_atom_edge_feature(wild_pdb_array)
    mut_atom_edge_index, mut_atom_edge_feature = get_edge().generate_atom_edge_feature(mut_pdb_array)

    # The final input matrix is obtained
    wild_res_node_feat = generate_node_feature(wild_res_feat_dict, wild_res_index_pos_dict, 105)
    mut_res_node_feat = generate_node_feature(mut_res_feat_dict, mut_res_index_pos_dict, 105)

    wild_atom_node_feat = generate_node_feature(wild_atom_feat_dict, wild_atom_index_pos, 5)
    mut_atom_node_feat = generate_node_feature(mut_atom_feat_dict, mut_atom_index_pos, 5)

    wild_mb_data = get_esm2(pdb_chain, wild_res_index_pos_dict, pdbpos2uniprotpos_dict, esm_dir)
    mut_mb_data = get_esm2(mut_id, mut_res_index_pos_dict, pdbpos2uniprotpos_dict, esm_dir)

    # save input matrix
    res_node_wt_path = f'{input_dir}/{mut_id}_RN_wt.npy'
    res_edge_wt_path = f'{input_dir}/{mut_id}_RE_wt.npy'
    res_edge_index_wt_path = f'{input_dir}/{mut_id}_REI_wt.npy'
    atom_node_wt_path = f'{input_dir}/{mut_id}_AN_wt.npy'
    atom_edge_wt_path = f'{input_dir}/{mut_id}_AE_wt.npy'
    atoms_edge_index_wt_path = f'{input_dir}/{mut_id}_AEI_wt.npy'
    atom2res_index_wt_path = f'{input_dir}/{mut_id}_I_wt.npy'
    extra_feat_wt_path = f'{input_dir}/{mut_id}_EF_wt.npy'

    res_node_mt_path = f'{input_dir}/{mut_id}_RN_mt.npy'
    res_edge_mt_path = f'{input_dir}/{mut_id}_RE_mt.npy'
    res_edge_index_mt_path = f'{input_dir}/{mut_id}_REI_mt.npy'
    atom_node_mt_path = f'{input_dir}/{mut_id}_AN_mt.npy'
    atom_edge_mt_path = f'{input_dir}/{mut_id}_AE_mt.npy'
    atom_edge_index_mt_path = f'{input_dir}/{mut_id}_AEI_mt.npy'
    atom2res_index_mt_path = f'{input_dir}/{mut_id}_I_mt.npy'
    extra_feat_mt_path = f'{input_dir}/{mut_id}_EF_mt.npy'

    np.save(res_node_wt_path, wild_res_node_feat)
    np.save(res_edge_wt_path, wild_res_edge_feature)
    np.save(res_edge_index_wt_path, wild_res_edge_index)
    np.save(atom_node_wt_path, wild_atom_node_feat)
    np.save(atom_edge_wt_path, wild_atom_edge_feature)
    np.save(atoms_edge_index_wt_path, wild_atom_edge_index)
    np.save(atom2res_index_wt_path, wild_atom_res_index)
    np.save(extra_feat_wt_path, wild_mb_data)

    np.save(res_node_mt_path, mut_res_node_feat)
    np.save(res_edge_mt_path, mut_res_edge_feature)
    np.save(res_edge_index_mt_path, mut_res_edge_index)
    np.save(atom_node_mt_path, mut_atom_node_feat)
    np.save(atom_edge_mt_path, mut_atom_edge_feature)
    np.save(atom_edge_index_mt_path, mut_atom_edge_index)
    np.save(atom2res_index_mt_path, mut_atom_res_index)
    np.save(extra_feat_mt_path, mut_mb_data)
















