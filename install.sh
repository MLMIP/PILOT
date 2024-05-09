conda create -n pilot python==3.10 -y
source activate pilot
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install torch-geometric==2.3.1
pip install scikit-learn
pip install biopython==1.81
pip install networkx


