# PILOT
***
## PILOT: Deep Siamese network with hybrid attention improves prediction of mutation impact on protein stability
***

PILOT is a novel hybrid attention-based Siamese network that evaluates the impact of single mutations on protein stability by computing the change in the unfolded Gibbs free energy. 

## Step 1: Clone the GitHub repository

```commandline
git clone https://github.com/MLMIP/PILOT.git
cd PILOT
```

## Step 2: Build required dependencies
It is recommended to use [Anaconda](https://www.anaconda.com/download#downloads) to install PyTorch, PyTorch Geometrics 
and other required Python libraries. Executing the below command will automatically install the Anaconda virtual 
environment. Upon completion, a virtual environment named "pilot" will be created. You can obtain the model weights
from [Zenodo-PILOT](https://zenodo.org/records/15300032).
```commandline
source install.sh
```

## Step 3: Download required software
The download methods for various software packages are provided below. After downloading, you can install it directly 
according to the official tutorial. It should be noted that after installing the following software, the paths in 
gen_features.py need to be modified to the corresponding paths of the installed software.

1. **FoldX** \
Download from https://foldxsuite.crg.eu/
2. **DSSP**\
Install using the conda command:``conda install -c ostrokach dssp``
3. **Naccess**\
Download from http://www.bioinf.manchester.ac.uk/naccess/
4. **PSI-BLAST**\
*Software* can be downloaded from https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html  \
*Database(Uniref90)* can be downloaded from https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/
5. **HHblits**\
*Software* can be downloaded from https://github.com/soedinglab/hh-suite \
*Database(Uniref30)* can be downloaded from https://gwdu111.gwdg.de/~compbiol/uniclust/2023_02/
6. **CLUSTALO**\
Download from http://www.clustal.org/omega/


## Step 4: Running PILOT
Activate the installed pilot virtual environment and ensure that the current working directory is PILOT.
```commandline
conda activate pilot
```
Then, you can use the following command to batch predict the effect of mutations on protein stability.
```commandline
python predict.py -i /path/to/where/input/file -o /path/to/where/output/file -d /path/to/where/all/features/is/stored
```
For example:
```commandline
python predict.py -i ./mutation_list.txt -o ./output_file.txt -d /features
```
Where the input file is a given file, with each line representing a specific mutation in the format 
`pdb_id  chain_id    mut_pos amino_acid`, such as `1A23	A	32	H/S`.

