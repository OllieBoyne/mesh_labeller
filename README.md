Tool for labelling 3D meshes.

Features:

- Part labelling

INSTALLATION

- Pyembree may need specific installation

On Apple Silicon:
- Pyembree not supported for Apple silicon, so need to use [the following workaround](https://towardsdatascience.com/how-to-manage-conda-environments-on-an-apple-silicon-m1-mac-1e29cb3bad12):
```
CONDA_SUBDIR=osx-64 conda create -n myenv_x86 python=3.9
conda activate myenv_x86
conda config --env --set subdir osx-64
conda install -c conda-forge pyembree
```

- Then, may say 'libembree.2.dylib' is missing. If so:
    * Download [.dylibs](https://github.com/embree/embree/releases/download/v2.7.0/embree-2.7.0.x86_64.macosx.tar.gz)
    * Copy the .dylib files to /venv/lib/...
    * Try opening them. If needed, go to Settings > Privacy and Security > Click 'allow' for each one