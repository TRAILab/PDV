# Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 18.04)
* Python 3.8
* PyTorch 1.7
* CUDA 10.2
* [`spconv v1.2 (commit abf0acf)`](https://github.com/traveller59/spconv/tree/abf0acf30f5526ea93e687e3f424f62d9cd8313a)

### Install `pcdet v0.1`
NOTE: Please re-install `pcdet v0.1` by running `python setup.py develop` even if you have already installed previous version.

a. Clone this repository.
```shell
git clone https://github.com/TRAILab/PDV.git
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries:
```
pip install -r requirements.txt
```

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv).
    * If you use PyTorch 1.1, then make sure you install the `spconv v1.0` with ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.
    * If you use PyTorch 1.3+, then you need to install the `spconv v1.2`. As mentioned by the author of [`spconv`](https://github.com/traveller59/spconv), you need to use their docker if you use PyTorch 1.4+.

c. Install this `pcdet` library by running the following command:
```shell
python setup.py develop
```
