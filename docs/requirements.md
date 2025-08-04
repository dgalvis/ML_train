First off, I would recommend using a Conda virual environment:

```bash
conda create --name ML_train
conda install -c conda-forge python ipython jupyterlab 
conda install -c conda-forge scipy pandas numpy networkx scikit-learn
conda install -c nbconvert matplotlib seaborn plotly, streamlit scikit-plot sqlite grip
pip3 install torch torchvision torchaudio
pip3 install mkdocs mkdocs-material
pip3 install mkdocs-pymdownx-material-extras
```

We probably don't need all of these packages, but this is the environment I'm working with. 


### Some notes
- `scikit-learn` and `torch` are for machine learning

- `sqlite` is used to manipulate SQL databases using SQL in Python. (Very useful data structures for industry)
- `mkdocs` is a great package for making code documentation websites (Like this one!)
- `nbconvert` converts Jupyter notebooks to .py files. 
- `matplotlib`, `seaborn`, `plotly` is for plotting and data visualisation, `streamlit` is for dashboards
