aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
aadict = {aa[k]: k for k in range(len(aa))}

import copy
import numpy as np
# from protein_physical_chemistry import aa, aadict

iupac_alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                  "W", "Y", "Z", "X", "*", "-"]
# BLOSUM62 background distribution
blosum_background_distr = [0.078, 0.051, 0.041, 0.052, 0.024, 0.034, 0.059, 0.083, 0.025, 0.062, 0.092, 0.056, 0.024,
                           0.044, 0.043, 0.059, 0.055, 0.014, 0.034, 0.072]

class calculate_conservation_score(object):
    def __init__(self):
        super(calculate_conservation_score, self).__init__()
        self.pdb_clean_dir =  '/storage3/database/PDB/foldx'
        self.main_dir = '/storage3/database/datasets/dataset_ddG/5.0/all_data'
        self.mut_pdb_dir = '/storage3/database/PDB/foldx/ddG_clean_output'

    def read_fasta_alignment(self, msa_file):
        alignment = []
        with open(msa_file, 'r') as f:
            msa_content = f.read()
        for to_process in msa_content.split('>')[1:]:
            to_process_list = to_process.split('\n')
            sequence = ''.join(to_process_list[1:]).upper().replace('B', 'D').replace('Z', 'Q').replace('X', '-')
            sequence = ''.join([c if c in iupac_alphabet else '-' for c in sequence])
            alignment.append(list(sequence.replace('U', '-')))

        return alignment


    def calculate_sequence_weights(self, alignment):
        seq_weights = np.zeros(len(alignment), dtype=float)   # 序列数量
        for i in range(len(alignment[0])):  # all positions
            freq_counts = np.zeros(21, dtype=float)    # 在i个位置上各个氨基酸出现的频数
            for j in range(len(alignment)):  # all sequences
                if alignment[j][i] != '-':  # ignore gaps
                    freq_counts[aadict[alignment[j][i]]] += 1
            num_observed_types = np.nonzero(freq_counts)[0].shape[0]   # 在i位置上存在过的氨基酸的类别数
            for j in range(len(alignment)):  # all sequences
                d = freq_counts[aadict[alignment[j][i]]] * num_observed_types
                if d > 0:
                    seq_weights[j] += 1 / d
        seq_weights = seq_weights / len(alignment)
        return seq_weights


    def weighted_freq_count_pseudocount(self, col, seq_weights, pseudocount):
        # If the weights do not match, use equal weight.
        if len(seq_weights) != len(col):
            seq_weights = np.ones(len(col), dtype=float)

        freq_counts = np.array([pseudocount] * 21)  # For each AA
        for j in range(len(col)):
            freq_counts[aadict[col[j]]] += 1 * seq_weights[j]  # 该位置上每种种氨基酸所在序列的分数之和
        freq_counts = freq_counts / (np.sum(seq_weights) + 21 * pseudocount)   # 归一化
        return freq_counts


    def weighted_gap_penalty(self, col, seq_weights):
        # If the weights do not match, use equal weight.
        if len(seq_weights) != len(col):
            seq_weights = np.ones(len(col), dtype=float)

        gap_sum = np.sum(seq_weights[np.where(np.array(col) == '-')[0]])
        return 1 - gap_sum / np.sum(seq_weights)


    def js_divergence(self, col, bg_distr, seq_weights, pseudocount, gap_penalty=1):
        fc = calculate_conservation_score().weighted_freq_count_pseudocount(col, seq_weights, pseudocount)

        # If background distribution lacks a gap count, remove fc gap count.
        if len(bg_distr) == 20:
            fc = fc[:-1]
            fc = fc / np.sum(fc)

        # Make r distribution
        r = 0.5 * fc + 0.5 * np.array(bg_distr)
        d = 0
        for i in range(r.shape[0]):
            if r[i] != 0:
                if fc[i] == 0:
                    d += bg_distr[i] * np.log2(bg_distr[i] / r[i])
                elif bg_distr[i] == 0:
                    d += fc[i] * np.log2(fc[i] / r[i])
                else:
                    d += fc[i] * np.log2(fc[i] / r[i]) + bg_distr[i] * np.log2(bg_distr[i] / r[i])
        d /= 2
        if gap_penalty == 1:
            return d * calculate_conservation_score().weighted_gap_penalty(col, seq_weights)
        else:
            return d


    def window_score(self, scores, window_len, lam):
        w_scores = copy.deepcopy(scores)
        for i in range(window_len, len(scores) - window_len):
            if scores[i] < 0:
                continue
            score_sum = 0
            num_terms = 0
            for j in range(i - window_len, i + window_len + 1):
                if i != j and scores[j] >= 0:
                    num_terms += 1
                    score_sum += scores[j]
            if num_terms > 0:
                w_scores[i] = (1 - lam) * (score_sum / num_terms) + lam * scores[i]
        return w_scores


    def calculate_js_div_from_msa(self, msa_file, bg_distr, pseudocount=0.0000001, window_size=3, lam=0.5):
        alignment = calculate_conservation_score().read_fasta_alignment(msa_file)
        seq_weights = calculate_conservation_score().calculate_sequence_weights(alignment)

        a = sum(seq_weights)
        scores = []
        for i in range(len(alignment[0])):
            col = [alignment[j][i] for j in range(len(alignment))]
            scores.append(calculate_conservation_score().js_divergence(col, bg_distr, seq_weights, pseudocount, 1))

        if window_size > 0:
            scores = calculate_conservation_score().window_score(scores, window_size, lam)
        return np.array(scores)

