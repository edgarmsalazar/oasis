<!-- ![](res/oasis_logo_1.png "title") -->
# Orbiting Mass Assignment Scheme

When using this code please cite Salazar et. al. (2025) ([arXiv:XXXX.XXXX]()).

## Requirements
Create a conda environment. An `environment.yml` file is supplied as an example but feel free to use your own. It must include `mpi4py` and `python>=3.10`.
```sh
$ conda env create -f environment.yml
```
Alternatively
```sh
$ conda create --name <env name> python=3.10 mpi4py
```

## Installation
Build from source
```sh
$ git clone https://github.com/edgarmsalazar/oasis.git
$ cd oasis
$ python -m pip install .
```
