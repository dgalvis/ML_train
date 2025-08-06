First off, I would recommend using a Conda virual environment:

```bash
conda create --name ML_train
conda install -c conda-forge python ipython jupyterlab 
conda install -c conda-forge scipy pandas numpy networkx scikit-learn
conda install -c conda-forge nbconvert matplotlib seaborn plotly
conda install -c conda-forge streamlit scikit-plot sqlite grip
pip3 install torch torchvision torchaudio
pip3 install mkdocs mkdocs-material
pip3 install mkdocs-pymdownx-material-extras
conda install -c conda-forge r-base r-essentials r-pacman r-psychtools
```

We probably don't need all of these packages, but this is the environment I'm working with. You can also try:

```bash
    conda env create -f requirements_mac.yml
```

using the requirements file in the main directory. I have typically found that this only works on the same OS though.

### Some notes
- `scikit-learn` and `torch` are for machine learning.
- `sqlite` is used to manipulate SQL databases using SQL in Python. (Very useful data structures for industry).
- `mkdocs` is a great package for making code documentation websites (Like this one!).
- `nbconvert` converts Jupyter notebooks to .py files. 
- `matplotlib`, `seaborn`, `plotly` is for plotting and data visualisation, `streamlit` is for dashboards.
- `r-base`, `r-essentials`, `r-pacman` will let us use `R`. `r-pacman` is a package manager that will automatically download required packages from CRAN. For deployment, it might be better to download packages using `conda`. This will work fine for us though.
- `r-psychtools` has the package `psych` which contains useful functions like `describe()`. On my Mac, I needed to have XCode installed first for this to work.
