import os
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa


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


def read_pdb(pdbfile, chain_id):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('PDB', pdbfile)
    model = struct[0]
    chain = model[chain_id]
    seq, position = '', []

    for res in chain:
        amino_acid = res.get_resname()
        res_id = res.get_id()
        pos = str(res_id[1]).strip() + str(res_id[2]).strip()
        if is_aa(amino_acid):
            try:
                aa = three_to_one(amino_acid)
            except KeyError:
                try:
                    aa = three_to_one(NON_STANDARD_SUBSTITUTIONS[amino_acid])
                except:
                    continue
            seq += aa
            position.append(pos)
    return seq, position



def gen_all_fasta(pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir, fasta_dir):
    pdbpos2uniprotpos_dict = {}

    mut_id = pdb_id + '_' + chain_id + '_' + wild_type + mut_pos + mutant
    wild_pdb = f'{cleaned_pdb_dir}/{pdb_id}_{chain_id}.pdb'
    mut_pdb = f'{cleaned_pdb_dir}/{mut_id}.pdb'
    wild_seq, pdb_positions = read_pdb(wild_pdb, chain_id)
    mut_seq, _ = read_pdb(mut_pdb, chain_id)

    wild_fasta = f'{fasta_dir}/{pdb_id}_{chain_id}.fasta'
    mut_fasta = f'{fasta_dir}/{mut_id}.fasta'

    if not os.path.exists(wild_fasta):
        with open(wild_fasta, 'w') as f_wild:
            f_wild.write(f'> {pdb_id}_{chain_id}\n{wild_seq}')

    if not os.path.exists(mut_fasta):
        with open(mut_fasta, 'w') as f_mut:
            f_mut.write(f'> {mut_id}\n{mut_seq}')

    for i, pos1 in enumerate(pdb_positions):
        pdbpos2uniprotpos_dict[pos1] = i+1

    return wild_fasta, mut_fasta, wild_seq, mut_seq, pdb_positions, pdbpos2uniprotpos_dict

