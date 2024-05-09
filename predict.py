import argparse
from gen_features import gen_features
import numpy as np
import torch


def Normalization(x, scop=1, start=0):
    return scop * (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0) + 1e-8) + start


def Standardization(x):
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)


def get_data(batchs):
    target = [data.y for data in batchs]
    x = [data.x for data in batchs]
    e = [data.edge_attr for data in batchs]
    ei = [data.edge_index for data in batchs]
    pos = [data.pos for data in batchs]

    x = torch.cat(x)
    e = torch.cat(e)
    pos = torch.cat(pos)
    ei = torch.cat(ei)
    return x, ei, e, pos, target


def predict(res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_feat_wt, index_wt,
                   res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_feat_mt, index_mt, rand):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    PATH = './PILOT_model.pkl'

    model = torch.load(PATH)
    model.to(device)

    model.eval()

    rand = torch.tensor(rand).view(1, 1)

    res_x_wt, res_ei_wt, res_e_wt, extra_feat_wt, rand = res_x_wt.to(device), res_ei_wt.to(device), res_e_wt.to(
        device), extra_feat_wt.to(device), rand.to(device)
    atom_x_wt, atom_ei_wt, atom_e_wt = atom_x_wt.to(device), atom_ei_wt.to(device), atom_e_wt.to(device)
    res_x_mt, res_ei_mt, res_e_mt, extra_feat_mt = res_x_mt.to(device), res_ei_mt.to(device), res_e_mt.to(
        device), extra_feat_mt.to(device)
    atom_x_mt, atom_ei_mt, atom_e_mt = atom_x_mt.to(device), atom_ei_mt.to(device), atom_e_mt.to(device)

    pred_y = model(res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_feat_wt, index_wt,
                   res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_feat_mt, index_mt, rand)
    return pred_y


def gen_seq_dict():
    uniref90_path = './database/uniref90/uniref90.fasta'
    seq_dict = {}
    print('Reading uniref90.fasta ...')
    with open(uniref90_path, 'r') as f_r:
        seq = ''
        for line in f_r:
            if line.startswith('>'):
                if seq:
                    seq_dict[identifier] = seq
                identifier = line.strip().split()[0].split('_')[1]
                seq = ''
            else:
                seq += line.strip()
        seq_dict[identifier] = seq
    print('finished reading!')
    return seq_dict

seq_dict = gen_seq_dict()


def get_pred_result(pdb_id, chain_id, mut_pos, wild_type, mutant, feature_dir):
    mut_info = [mut_pos, wild_type, mutant]
    mut_id = pdb_id + '_'+ chain_id + '_' + mut_info[1] + mut_info[0] + mut_info[2]
    res_node_wt_path = f'{feature_dir}/{mut_id}_RN_wt.npy'
    res_edge_wt_path = f'{feature_dir}/{mut_id}_RE_wt.npy'
    res_edge_index_wt_path = f'{feature_dir}/{mut_id}_REI_wt.npy'
    atom_node_wt_path = f'{feature_dir}/{mut_id}_AN_wt.npy'
    atom_edge_wt_path = f'{feature_dir}/{mut_id}_AE_wt.npy'
    atom_edge_index_wt_path = f'{feature_dir}/{mut_id}_AEI_wt.npy'
    atom2res_index_wt_path = f'{feature_dir}/{mut_id}_I_wt.npy'
    extra_feat_wt_path = f'{feature_dir}/{mut_id}_EF_wt.npy'

    res_node_mt_path = f'{feature_dir}/{mut_id}_RN_mt.npy'
    res_edge_mt_path = f'{feature_dir}/{mut_id}_RE_mt.npy'
    res_edge_index_mt_path = f'{feature_dir}/{mut_id}_REI_mt.npy'
    atom_node_mt_path = f'{feature_dir}/{mut_id}_AN_mt.npy'
    atom_edge_mt_path = f'{feature_dir}/{mut_id}_AE_mt.npy'
    atoms_edge_index_mt_path = f'{feature_dir}/{mut_id}_AEI_mt.npy'
    atom2res_index_mt_path = f'{feature_dir}/{mut_id}_I_mt.npy'
    extra_feat_mt_path = f'{feature_dir}/{mut_id}_EF_mt.npy'


    res_node_wt = np.load(res_node_wt_path).astype(float)
    res_node_wt[:, :-1] = Standardization(res_node_wt[:, :-1])
    res_edge_wt = np.load(res_edge_wt_path).astype(float)
    res_edge_wt = Normalization(res_edge_wt)
    res_edge_index_wt = np.load(res_edge_index_wt_path).astype(int)
    atom_node_wt = np.load(atom_node_wt_path).astype(float)
    atom_edge_wt = np.load(atom_edge_wt_path).astype(float)
    atom_edge_wt = Normalization(atom_edge_wt)
    atom_edge_index_wt = np.load(atom_edge_index_wt_path).astype(int)
    atom2res_index_wt = np.load(atom2res_index_wt_path).astype(int)
    extra_feat_wt = np.load(extra_feat_wt_path).astype(float)

    res_node_mt = np.load(res_node_mt_path).astype(float)
    res_node_mt[:, :-1] = Standardization(res_node_mt[:, :-1])
    res_edge_mt = np.load(res_edge_mt_path).astype(float)
    res_edge_mt = Normalization(res_edge_mt)
    res_edge_index_mt = np.load(res_edge_index_mt_path).astype(int)
    atom_node_mt = np.load(atom_node_mt_path).astype(float)
    atom_edge_mt = np.load(atom_edge_mt_path).astype(float)
    atom_edge_mt = Normalization(atom_edge_mt)
    atoms_edge_index_mt = np.load(atoms_edge_index_mt_path).astype(int)
    atom2res_index_mt = np.load(atom2res_index_mt_path).astype(int)
    extra_feat_mt = np.load(extra_feat_mt_path).astype(float)


    res_node_wt = torch.tensor(res_node_wt, dtype=torch.float)
    res_edge_wt = torch.tensor(res_edge_wt, dtype=torch.float)
    res_edge_index_wt = torch.tensor(res_edge_index_wt, dtype=torch.int64)
    atom_node_wt = torch.tensor(atom_node_wt, dtype=torch.float)
    atom_edge_wt = torch.tensor(atom_edge_wt, dtype=torch.float)
    atom_edge_index_wt = torch.tensor(atom_edge_index_wt, dtype=torch.int64)
    atom2res_index_wt = torch.tensor(atom2res_index_wt, dtype=torch.int64)
    extra_feat_wt = torch.tensor(extra_feat_wt, dtype=torch.float)

    res_node_mt = torch.tensor(res_node_mt, dtype=torch.float)
    res_edge_mt = torch.tensor(res_edge_mt, dtype=torch.float)
    res_edge_index_mt = torch.tensor(res_edge_index_mt, dtype=torch.int64)
    atom_node_mt = torch.tensor(atom_node_mt, dtype=torch.float)
    atom_edge_mt = torch.tensor(atom_edge_mt, dtype=torch.float)
    atom_edge_index_mt = torch.tensor(atoms_edge_index_mt, dtype=torch.int64)
    atom2res_index_mt = torch.tensor(atom2res_index_mt, dtype=torch.int64)
    extra_feat_mt = torch.tensor(extra_feat_mt, dtype=torch.float)

    pred_y = predict(res_node_wt, res_edge_index_wt, res_edge_wt, atom_node_wt, atom_edge_index_wt, atom_edge_wt,
                     extra_feat_wt, atom2res_index_wt,
                     res_node_mt, res_edge_index_mt, res_edge_mt, atom_node_mt, atom_edge_index_mt, atom_edge_mt,
                     extra_feat_mt, atom2res_index_mt, 0)
    pred_y = pred_y.detach().cpu().numpy()[0][0]
    return pred_y



def main():
    parser = argparse.ArgumentParser(description="Use PILOT to predict ddG from ")
    parser.add_argument('-i', '--mutant-list', dest='mutant_list', type=str,
                        required=True, help='The file storing the information of the mutations.')
    parser.add_argument('-o', '--output-file', dest='outfile', type=str,
                        required=True, help='The path of the result.')
    parser.add_argument('-d', '--feature-dir', dest='dir', type=str,
                        required=True, help='The path to store intermediate features and model inputs.')
    args = parser.parse_args()
    infile = args.mutant_list
    outfile = args.outfile
    feature_dir = args.dir

    with open(infile, 'r') as f_r, open(outfile, 'a') as f_w:
        for line in f_r:
            mut_info = line.strip().split()
            pdb_id = mut_info[0]
            chain_id = mut_info[1]
            mut_pos = mut_info[2]
            wild_type = mut_info[3][0]
            mutant = mut_info[3][-1]
            mut_id = pdb_id + '_' + chain_id + '_' + wild_type + mut_pos + mutant
            # gen_features(pdb_id, chain_id, mut_pos, wild_type, mutant, feature_dir, seq_dict)
            input_dir = f'{feature_dir}/input'
            pred_ddG = get_pred_result(pdb_id, chain_id, mut_pos, wild_type, mutant, input_dir)
            f_w.write(f'{pdb_id}\t{chain_id}\t{mut_pos}\t{wild_type}/{mutant}\t{pred_ddG}\n')
            print('======================================')
            print(f'The result of {mut_id}: {pred_ddG}.')
            print('======================================')


if __name__ == "__main__":
    main()
