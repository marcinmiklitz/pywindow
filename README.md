![alt tag](docs/pyWINDOW_logo.png)
### Python package for the analysis of structural properties of molecular pores (*porous organic cages*, but also *MOFs* and *metalorganic cages* - see examples directory).

### Documentation

https://marcinmiklitz.github.io/pywindow/

### How to install `pywindow`

Git clone this repository or download a zipped version.

cd pywindow/

and run

python setup.py install

### Overview

Structural parameters associated with porous organic molecules that are available
to calculate using `pywindow` software.

* COM: centre of mass of a molecule.
* d<sub>max</sub>: the maximum diameter of a molecule.
* d<sub>avg</sub>: the average diameter of a molecule.
* d<sub>void</sub>: the intrinsic void diameter of a molecule.
* V<sub>void</sub>: the intrinsic void volume of a molecule.
* d<sub>void_opt</sub>: the optimised intrinsic void diameter of a molecule.
* V<sub>void_opt</sub>: the optimised intrinsic void volume of a molecule.
* d<sub>window</sub>: the circular diameter of an xth window of a molecule.

Instructions and examples how to calculate these structural parameters are in form of Jupyter notebooks in the examples directory.

---------------------------------------------------------------
MIT License | Copyright (c) 2017 Marcin Miklitz, Jelfs Materials Group
