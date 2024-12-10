import os




def download_row_pdb(pdb_id, row_pdb_dir):
    pdb_file = os.path.join(row_pdb_dir, pdb_id + '.pdb')
    if not os.path.exists(pdb_file):
        os.chdir(row_pdb_dir)
        os.system(f'wget https://files.rcsb.org/view/{pdb_id}.pdb')

def cleaned_row_pdb(pdb_id, chain_id, row_pdb_dir, cleaned_pdb_dir):
    row_pdb = os.path.join(row_pdb_dir, pdb_id + '.pdb')
    cleaned_pdb = os.path.join(cleaned_pdb_dir, pdb_id+ '_'+chain_id+ '.pdb')
    with open(row_pdb, 'r') as f_r, open(cleaned_pdb, 'w') as f_w:
        for line in f_r:
            info = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27], line[30:38],
                    line[38:46], line[46:54]]
            info = [i.strip() for i in info]
            if info[5] == chain_id and (info[0] == 'ATOM' or info[0] == 'HETAT'):
                f_w.write(line)
            if 'ENDMDL' in line:
                break
    return cleaned_pdb


def gen_mut_pdb(pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir, foldx_path):
    mut_info = wild_type + chain_id + mut_pos + mutant
    os.chdir(foldx_path)
    os.system(f'cp {cleaned_pdb_dir}/{pdb_id}_{chain_id}.pdb ./')
    pdbfile = './' + pdb_id + '_' + chain_id + '.pdb'
    mut_id = pdb_id + '_' + chain_id + '_' + wild_type + mut_pos + mutant
    workdir = f'../source_pdb/' + mut_id
    if not os.path.exists(f'{cleaned_pdb_dir}/{mut_id}.pdb'):
        print(f'{cleaned_pdb_dir}/{mut_id}.pdb')
        if not os.path.exists(workdir):
            os.mkdir(workdir)
        with open(f'{foldx_path}/individual_list.txt', 'w') as f:
            f.write(mut_info + ';')

        cmd = './foldx --command=BuildModel --pdb={}  --mutant-file={}  --output-dir={} --pdb-dir={} >{}/foldx.log'.format( \
            pdbfile, 'individual_list.txt', workdir, './', workdir)

        os.system(cmd)
        os.system(f'cp {workdir}/{pdb_id}_{chain_id}_1.pdb  {cleaned_pdb_dir}/{mut_id}.pdb')
    return f'{cleaned_pdb_dir}/{mut_id}.pdb'




def gen_all_pdb(pdb_id, chain_id, mut_pos, wild_type, mutant, row_pdb_dir, cleaned_pdb_dir, foldx_path):
    download_row_pdb(pdb_id, row_pdb_dir)
    wild_pdb = cleaned_row_pdb(pdb_id, chain_id, row_pdb_dir, cleaned_pdb_dir)
    mut_pdb = gen_mut_pdb(pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir, foldx_path)
    return wild_pdb, mut_pdb
