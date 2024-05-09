import os
import numpy as np
from Bio import SeqIO



def use_psiblast(fasta_file, rawmsa_dir, psi_path, uniref90_path):
    fasta_name = os.path.basename(fasta_file).split('.')[0]
    rawmsa_file = os.path.join(rawmsa_dir, fasta_name+'.rawmsa')
    pssm_file = os.path.join(rawmsa_dir, fasta_name + '.pssm')
    if not os.path.exists(pssm_file):
        cmd = psi_path + ' -query ' + \
            fasta_file + ' -db ' + uniref90_path + ' -out ' + rawmsa_file + \
            ' -evalue 0.001 -matrix BLOSUM62 -num_iterations 3 -num_threads 16 -out_ascii_pssm ' + pssm_file
        os.system(cmd)
    return rawmsa_file, pssm_file



def format_rawmsa(prot_id, rawmsa_file, seq_dict, formatted_output_file):
    identifiers_to_align = set()
    with open(rawmsa_file, 'r') as infile:
        for line in infile:
            if line.startswith('>'):
                identifier = line.strip().split()[0]
                if identifier.split('_')[1] != prot_id:
                    identifiers_to_align.add(identifier)

    if len(identifiers_to_align) > 0:
        with open(formatted_output_file, 'w') as outfile:
            for identifier in sorted(identifiers_to_align):
                if identifier.split('_')[1] in seq_dict:
                    outfile.write(identifier + '\n' + seq_dict[identifier.split('_')[1]] + '\n')


'''
使用clustalo对clustal_input文件进行处理，按最长的序列对齐
'''
def run_clustal(clustal_input_file, clustal_output_file, clustalo_path, num_threads=6):
    with open(clustal_input_file, 'r') as f:
        numseqs = len(f.readlines())
    numseqs /= 2

    if numseqs > 1:
        os.system('%s -i %s -o %s --force --threads %s' % (
        clustalo_path, clustal_input_file, clustal_output_file, str(num_threads)))
    else:
        os.system('cp %s %s' % (clustal_input_file, clustal_output_file))

'''
将clustalo生成的clustal按照参考序列进行对齐
'''
def format_clustal(clustal_output_file, formatted_output_file):
    msa_info = []
    with open(clustal_output_file, 'r') as f:
        seq_name = ''
        seq = ''
        for line in f:
            if line.startswith('>'):
                if seq_name:
                    msa_info.append(seq_name)
                    msa_info.append(seq)
                seq_name = line.strip()
                seq = ''
            else:
                seq += line.strip()
        msa_info.append(seq_name)
        msa_info.append(seq.replace('U', '-'))
    # Make a temporary MSA file where the sequences are not split across multiple lines.
    # Generate the formatted CLUSTAL output.

    # Read clustal MSA
    outtxt = ''
    gaps = []
    # Iterate over each line
    for idx, line in enumerate(msa_info):
        #             line = line.strip()
        # Add Header lines as they are
        if idx % 2 == 0:
            outtxt += line
            outtxt += '\n'
        # Special case for the first entry in the alignment
        # Find all of the gaps in the alignment since we only care about using the MSA with regard to the current UniProt
        # query. We don't care about any of the positions where the query has a gap
        elif idx == 1:  # Query
            for i in range(len(line)):  # Find all the Gaps
                gaps.append(line[i] == '-')
        # For all matches
        if idx % 2 == 1:
            # Update the sequence by removing all of the positions that were a gap in the current UniProt alignment
            newseq = ''
            for i in range(len(gaps)):
                if not gaps[i]:
                    if i < len(line):
                        newseq += line[i]
                    else:
                        newseq += '-'
            # Write the formatted alignment sequence
            outtxt += newseq
            outtxt += '\n'
    # Write all of the formatted alignment lines to the final alignment output
    with open(formatted_output_file, 'w') as f:
        f.write(outtxt)






def gen_msa(prot_id, prot_seq, rawmsa_file, seq_dict, output_dir, clustalo_path):
    formatted_fasta_file = os.path.join(output_dir, prot_id + '_rawmsa.fasta')
    clustal_input_file = os.path.join(output_dir, prot_id + '.clustal_input')
    clustal_output_file = os.path.join(output_dir, prot_id + '.clustal')
    formatted_clustal_file = os.path.join(output_dir, prot_id + '.msa')

    format_rawmsa(prot_id, rawmsa_file, seq_dict, formatted_fasta_file)

    if not os.path.exists(clustal_input_file):
        with open(formatted_fasta_file, 'r') as infile:
            lines = infile.readlines()

        with open(clustal_input_file, 'w') as outfile:
            outfile.write('>' + prot_id + '\n' + prot_seq + '\n')
            for line in lines:
                outfile.write(line)

        run_clustal(clustal_input_file, clustal_output_file, clustalo_path)

    if not os.path.exists(formatted_clustal_file):
        format_clustal(clustal_output_file, formatted_clustal_file)

    return formatted_clustal_file


def use_hhblits(seq_name, fasta_file, hhblits_path, uniRef30_path, hhm_dir):
    if not os.path.exists(hhm_dir + '/' + seq_name + '.hhm'):
        cmd = hhblits_path + ' -cpu {} -i {} -d {} -ohhm {}'. \
            format(16, fasta_file, uniRef30_path,
                   hhm_dir + '/' + seq_name + '.hhm')
        os.system(cmd)
    return hhm_dir + '/' + seq_name + '.hhm'



def get_pssm(pssm_path):
    pssm_dict, new_pssm_dict, res_dict = {}, {}, {}
    with open(pssm_path, 'r') as f_r:
        next(f_r)
        next(f_r)
        next(f_r)
        for line in f_r:
            line = line.split()
            if len(line) > 20:
                pos = line[0]
                aa = line[1]
                pssm = line[2:22]
                pssm_dict[pos] = [float(i) for i in pssm]
                res_dict[pos] = aa
        for key in pssm_dict.keys():
            pssm = np.array(pssm_dict[key])
            pssm = 1 / (np.exp(-pssm) + 1)
            new_pssm_dict[key] = list(pssm)
    return new_pssm_dict, res_dict



def process_hhm(path):
    with open(path,'r') as fin:
        fin_data = fin.readlines()
        hhm_begin_line = 0
        hhm_end_line = 0
        for i in range(len(fin_data)):
            if '#' in fin_data[i]:
                hhm_begin_line = i+5
            elif '//' in fin_data[i]:
                hhm_end_line = i
        feature = np.zeros([int((hhm_end_line-hhm_begin_line)/3),30])
        axis_x = 0
        for i in range(hhm_begin_line,hhm_end_line,3):
            line1 = fin_data[i].split()[2:-1]
            line2 = fin_data[i+1].split()
            axis_y = 0
            for j in line1:
                if j == '*':
                    feature[axis_x][axis_y]=9999/10000.0
                else:
                    feature[axis_x][axis_y]=float(j)/10000.0
                axis_y+=1
            for j in line2:
                if j == '*':
                    feature[axis_x][axis_y]=9999/10000.0
                else:
                    feature[axis_x][axis_y]=float(j)/10000.0
                axis_y+=1
            axis_x+=1
        feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

        return feature

def loadAASeq(infile):
    seqs = []
    for i in SeqIO.parse(infile, 'fasta'):
        seqs.append(i.seq)
    return seqs, len(seqs[0])


def calc_res_freq(infile):
    aa_name = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
    seqs, length = loadAASeq(infile)
    conservation_dict = {}
    for res_pos in range(1, length+1):
        conservation_dict[res_pos] = np.zeros((21))
    for seq in seqs:
        for res_pos in range(1, length+1):
            res = seq[int(res_pos)-1]
            # if res == 'X' or res == 'B':
            #     res = '-'
            try:
                index = aa_name.index(res)
            except ValueError:
                continue
            conservation_dict[res_pos][index] += 1
    for res_pos in range(1, length+1):
        conservation_dict[res_pos] = conservation_dict[res_pos] / len(seqs)
    return conservation_dict