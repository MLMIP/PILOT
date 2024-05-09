import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools
import os

class get_edge(object):
    def __init__(self):
        super(get_edge, self).__init__()
        self.pydca_dir = '/storage3/database/datasets/dataset_ddG/5.0/MSA_and_plmdca_mfdca'

    def get_pydca_dict(self, uniprot_id, pdb_id, chain_id, pdb2uniprot_pos):
        plmdca_dict, mfdca_dict = {}, {}
        uniprot_pos = pdb2uniprot_pos.values()
        if uniprot_id == '-':
            pdb_chain = pdb_id + '_' + chain_id
            plmdca_path = os.path.join(self.pydca_dir, pdb_chain+'_plmdca', 'PLMDCA_apc_fn_scores_'+pdb_chain+'.txt')
            mfdca_path = os.path.join(self.pydca_dir, pdb_chain+'_mfdca', 'MFDCA_apc_fn_scores_'+pdb_chain+'.txt')
        else:
            plmdca_path = os.path.join(self.pydca_dir, uniprot_id+'_plmdca', 'PLMDCA_apc_fn_scores_'+uniprot_id+'.txt')
            mfdca_path = os.path.join(self.pydca_dir, uniprot_id+'_mfdca', 'MFDCA_apc_fn_scores_'+uniprot_id+'.txt')
        with open(plmdca_path) as f_r_1:
            for line in f_r_1:
                if line[0] != '#':
                    info = line.split()
                    if (info[0] in uniprot_pos) and (info[1] in uniprot_pos):
                        if int(info[0]) <= int(info[1]): pos1, pos2 = info[0], info[1]
                        else: pos1, pos2 = info[1], info[0]
                        plmdca = info[2]
                        plmdca_dict[pos1+'_'+pos2] = plmdca
        with open(mfdca_path) as f_r_1:
            for line in f_r_1:
                if line[0] != '#':
                    info = line.split()
                    if (info[0] in uniprot_pos) and (info[1] in uniprot_pos):
                        if int(info[0]) <= int(info[1]): pos1, pos2 = info[0], info[1]
                        else: pos1, pos2 = info[1], info[0]
                        mfdca = info[2]
                        mfdca_dict[pos1+'_'+pos2] = mfdca
        return plmdca_dict, mfdca_dict

    def get_residue_info(self, pdb_array):
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

    def get_atom_distance_matrix(self, pdb_array):
        coord_array = np.empty((pdb_array.shape[0], 3))
        for i, res_array in enumerate(pdb_array):
            coord_array[i] = res_array[7:10].astype(np.float64)
        atom_dm = squareform(pdist(coord_array))
        return coord_array, atom_dm

    def get_residue_distance_matrix(self, pdb_array, residue_index, distance_type):
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

    def get_atom_neighbor_index(self, atom_dm, threshold=3):
        source, target, distance = [], [], []
        for i in range(atom_dm.shape[0]):
            for j in range(atom_dm.shape[1]):
                if atom_dm[i, j] <= threshold:
                    source.append(i)
                    target.append(j)
                    distance.append(atom_dm[i, j])
        return source, target, distance

    def vector_dot(self, digit):
        if digit > 1:
            digit = 1
        elif digit < -1:
            digit = -1
        return digit

    def get_atom_neighbor_angle(self, coord_array, source, target):
        polar_angle, azimuthal_angle = [], []
        for (pos1, pos2) in zip(source, target):
            if pos1 == pos2:
                polar_angle.append(0.0)
                azimuthal_angle.append(0.0)
            else:
                coord_array_1 = coord_array[pos1]
                coord_array_2 = coord_array[pos2]
                direction_vector = coord_array_1 - coord_array_2
                unit_vector = (direction_vector) / np.linalg.norm(direction_vector, 2)
                direction_vector[2] = 0.0
                projection_vector = (direction_vector) / np.linalg.norm(direction_vector, 2)
                z_axis = np.array([0.0, 0.0, 1.0])
                x_axis = np.array([1.0, 0.0, 0.0])
                p_angle = np.arccos(get_edge().vector_dot(np.sum(unit_vector * z_axis))) / np.pi
                a_angle = np.arccos(get_edge().vector_dot(np.sum(projection_vector * x_axis))) / np.pi
                polar_angle.append(p_angle)
                azimuthal_angle.append(a_angle)
        return polar_angle, azimuthal_angle

    def get_residue_neighbor_angle(self, pdb_array, residue_index, neighbor_index):
        normal_vector_array = np.empty((neighbor_index.shape[0], 3), dtype=np.float64)
        for i, (res_start, res_end) in enumerate(residue_index):
            res_info = pdb_array[res_start:res_end + 1]
            res_acid_plane_index = np.where(
                np.logical_and(np.isin(res_info[:, 2], ['CA', 'C', 'O']), np.isin(res_info[:, 3], ['', 'A'])))
            res_acid_plane = res_info[res_acid_plane_index][:, 7:10].astype(np.float64)
            if res_acid_plane.shape[0] != 3:
                normal_vector_array[i] = np.array([np.nan] * 3)
                continue
            normal_vector = get_edge().get_normal(res_acid_plane)
            if np.all(np.isnan(normal_vector)):
                normal_vector_array[i] = np.array([np.nan] * 3)
            else:
                normal_vector_array[i] = normal_vector

        pairwise_normal_dot = normal_vector_array.dot(normal_vector_array.T)

        # Correct floating point precision error
        pairwise_normal_dot[pairwise_normal_dot > 1] = 1
        pairwise_normal_dot[pairwise_normal_dot < -1] = -1

        pairwise_angle = np.arccos(pairwise_normal_dot) / np.pi

        # angle_matrix = np.empty_like(neighbor_index, dtype=np.float64)
        # for i, index in enumerate(neighbor_index):
        #     angle_matrix[i] = pairwise_angle[i, index]
        #
        # angle_matrix = get_edge().fill_nan_mean(angle_matrix, axis=0)
        return pairwise_angle

    def get_normal(self, acid_plane):
        cp = np.cross(acid_plane[2] - acid_plane[1], acid_plane[0] - acid_plane[1])
        if np.all(cp == 0):
            return np.array([np.nan] * 3)
        normal = cp / np.linalg.norm(cp, 2)
        return normal

    def fill_nan_mean(self, array, axis=0):
        if axis not in [0, 1]:
            raise ValueError('Invalid axis: %s' % axis)
        mean_array = np.nanmean(array, axis=axis)
        inds = np.where(np.isnan(array))
        array[inds] = np.take(mean_array, inds[1 - axis])
        if np.any(np.isnan(array)):
            full_array_mean = np.nanmean(array)
            inds = np.unique(np.where(np.isnan(array))[1 - axis])
            if axis == 0:
                array[:, inds] = full_array_mean
            else:
                array[inds] = full_array_mean
        return array

    def get_residue_edge_data(self, residue_dm, neighbor_index, neighbor_angle):
        edge_matrix = np.zeros((neighbor_index.shape[0], neighbor_index.shape[1], 2))
        for i, dist in enumerate(residue_dm):
            edge_matrix[i][:, 0] = dist[neighbor_index[i]]
            edge_matrix[i][:, 1] = neighbor_angle[i]
        return edge_matrix

    def get_neighbor_index(self, residue_dm, num_neighbors=16):
        return residue_dm.argsort()[:, :num_neighbors]

    def add_pydca(self, edge_index, edge_feature, plmdca_dict, mfdca_dict, res_index_pos_dict_premut, pdb2uniprot_pos):
        edge_feature = edge_feature.tolist()
        for (i, pos) in enumerate(zip(edge_index[0], edge_index[1])):
            uniprot_pos1 = pdb2uniprot_pos[res_index_pos_dict_premut[pos[0]]]
            uniprot_pos2 = pdb2uniprot_pos[res_index_pos_dict_premut[pos[1]]]
            if uniprot_pos1 == uniprot_pos2:
                edge_feature[i].append(0.00)
                edge_feature[i].append(0.00)
                continue
            else:
                if int(uniprot_pos1) < int(uniprot_pos2): index = uniprot_pos1+'_'+uniprot_pos2
                else: index = uniprot_pos2+'_'+uniprot_pos1
                plmdca, mfdca = plmdca_dict[index], mfdca_dict[index]
                edge_feature[i].append(float(plmdca))
                edge_feature[i].append(float(mfdca))
        # print(edge_feature)
        edge_feature = np.array(edge_feature)
        return edge_feature


    def generate_res_edge_feature_postmut(self, mut_pos, pdb_array, num_neighbors=16, distance_type='c_alpha'):
        residue_index, pdb_pos_list = get_edge().get_residue_info(pdb_array)
        residue_dm = get_edge().get_residue_distance_matrix(pdb_array, residue_index, distance_type)
        neighbor_index = get_edge().get_neighbor_index(residue_dm, num_neighbors)
        neighbor_angle = get_edge().get_residue_neighbor_angle(pdb_array, residue_index, neighbor_index)

        mut_index = pdb_pos_list.index(mut_pos)
        edge_num = len(pdb_pos_list)
        source = np.array([mut_index for i in range(edge_num)])
        target = np.array([i for i in range(edge_num)])

        edge_data = np.empty((len(pdb_pos_list), 2), dtype=np.float64)
        for s, t in zip(source, target):
            edge_data[t][0] = residue_dm[s][t]
            edge_data[t][1] = neighbor_angle[s][t]


        edge_index = [source, target]
        return edge_index, edge_data, residue_index

    def generate_atom_edge_feature(self, pdb_array):
        coord_array, atom_dm = get_edge().get_atom_distance_matrix(pdb_array)
        source, target, distance = get_edge().get_atom_neighbor_index(atom_dm, 3)
        polar_angle, azimuthal_angle = get_edge().get_atom_neighbor_angle(coord_array, source, target)
        edge_feature = np.empty((len(source), 3))
        for (i, edge_info) in enumerate(zip(distance, polar_angle, azimuthal_angle)):
            edge_feature[i][0] = edge_info[0]
            edge_feature[i][1] = edge_info[1]
            edge_feature[i][2] = edge_info[2]
        edge_index = [np.array(source), np.array(target)]
        return edge_index, edge_feature

    def get_redisue_neighbor_index(self, residue_dm, angle_matrix, threshold=8):
        source, target, distance, angle = [], [], [], []
        for i in range(residue_dm.shape[0]):
            for j in range(residue_dm.shape[1]):
                if residue_dm[i, j] <= threshold:
                    source.append(i)
                    target.append(j)
                    distance.append(residue_dm[i, j])
                    angle.append(angle_matrix[i, j])
        return source, target, distance, angle


    # 边修改后的，也就是把氨基酸之间距离小于8埃的视作有边
    def generate_residue_edge_feature(self, pdb_array, num_neighbors=4, distance_type='c_alpha'):
        residue_index, pdb_pos_list = get_edge().get_residue_info(pdb_array)
        residue_dm = get_edge().get_residue_distance_matrix(pdb_array, residue_index, distance_type)
        neighbor_index = get_edge().get_neighbor_index(residue_dm, num_neighbors)
        angle_matrix = get_edge().get_residue_neighbor_angle(pdb_array, residue_index, neighbor_index)
        source, target, distance, angle = get_edge().get_redisue_neighbor_index(residue_dm, angle_matrix, 8)
        edge_index = [np.array(source), np.array(target)]
        edge_feature = np.empty((len(source), 2))
        for (i, edge_info) in enumerate(zip(distance, angle)):
            edge_feature[i][0] = edge_info[0]
            edge_feature[i][1] = edge_info[1]

        return edge_index, edge_feature, residue_index


