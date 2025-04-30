import torch
from Bio import SeqIO
import os



def use_esm2(fasta_file, prot_id, saved_folder):
    os.environ['TORCH_HOME'] = '/storage2/student/djs0'
    pt_file = saved_folder + '/' + prot_id + '.pt'
    if os.path.exists(pt_file):
        return pt_file
    with torch.no_grad():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model, alphabet = torch.hub.load('facebookresearch/esm:main', 'esm2_t33_650M_UR50D')
        model = model.eval().to(device)

        batch_converter = alphabet.get_batch_converter()
        data = [('protein', str(list(SeqIO.parse(fasta_file, 'fasta'))[0].seq))]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        result = model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
        representations = result['representations'][33][0, 1:-1, :]
        torch.save(representations.detach().cpu().clone(), pt_file)
    return pt_file

