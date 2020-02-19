#conda create --name ntwks -y
conda activate ntwks

conda install numpy -y
conda install pandas -y
conda install scikit-learn -y
conda install -c conda-forge matplotlib -y
conda install -c anaconda scipy -y
conda install -c conda-forge jupyterlab -y
conda install -c anaconda seaborn -y
conda install -c conda-forge glob2 -y
conda install -c anaconda networkx -y
conda install -c plotly plotly -y

conda update --all
conda clean --all
