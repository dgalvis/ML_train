First off, I would recommend using a Conda virual environment:

```bash
conda create --name ML_train
conda install -c conda-forge python ipython jupyterlab 
conda install -c conda-forge scipy pandas numpy networkx scikit-learn
conda install -c nbconvert matplotlib seaborn plotly scikit-plot sqlite grip
pip3 install torch torchvision torchaudio
pip3 instal mkdocs mkdocs-material
```

We probably don't need all of these packages, but this is the environment I'm working with. 
