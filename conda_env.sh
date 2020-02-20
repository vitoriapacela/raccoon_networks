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

# the following is needed to display plotly plots in jupyter lab
# source: https://github.com/plotly/plotly.py#jupyterlab-support-python-35
conda install -c conda-forge nodejs
# Avoid "JavaScript heap out of memory" errors during extension installation
# (OS X/Linux)
export NODE_OPTIONS=--max-old-space-size=4096

# Jupyter widgets extension
jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.1 --no-build

# FigureWidget support
jupyter labextension install plotlywidget@1.5.0 --no-build

# and jupyterlab renderer support
jupyter labextension install jupyterlab-plotly@1.5.0 --no-build

# Build extensions (must be done to activate extensions since --no-build is used above)
jupyter lab build

# Unset NODE_OPTIONS environment variable
# (OS X/Linux)
unset NODE_OPTIONS

conda update --all
conda clean --all
