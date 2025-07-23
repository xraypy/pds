# PDS

surface scattering integration


## Prerequisites

**Ensure anaconda/miniconda is installed.**

## Configuring the python environment

```bash
conda create -n pdsENV python=3.13 -y && conda config --add channels conda-forge -y && conda install -c conda-forge h5py matplotlib numpy pillow pytest scipy tables wxpython -y && pip install pyshortcuts
```

## How to run

**Activate the environment**
```bash
conda activate pdsENV
```

**Start the integrator**
```
python pds.py -i
```

**Run tests**
```
python pds.py -t
```
