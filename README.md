# adomo-mocis-wape-daml-mooc

#### Marc Bocquet¹ [marc.bocquet@enpc.fr](mailto:marc.bocquet@enpc.fr) and Alban Farchi¹ [alban.farchi@enpc.fr](mailto:alban.farchi@enpc.fr)
##### (1) CEREA, École des Ponts and EdF R&D, IPSL, Île-de-France, France

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10491402.svg)](https://doi.org/10.5281/zenodo.10491402)

* This jupyter notebook has been created by Marc Bocquet and Alban Farchi for the ENPC/ADOMO, MOCIS/NUM2.2 and WAPE/NUM2.2 course.

* This corresponds to one lecture block (3.5 hours).

* This is an introduction to the use of deep learning techniques into geophysical data assimilation.

## Installation

Install conda, for example through [miniconda](https://docs.conda.io/en/latest/miniconda.html) or through [mamba](https://mamba.readthedocs.io/en/latest/installation.html).

Clone the repertory:

    $ git clone git@github.com:cerea-daml/adomo-mocis-wape-daml-mooc.git

Go to the repertory. Once there, create a dedicated anaconda environment for the sessions:

    $ conda env create -f environment.yaml

Activate the newly created environment:

    $ conda activate mooc

Open the notebooks (e.g. with Jupyter) and run the cells:

    $ jupyter-notebook daml-part1.ipynb
    $ jupyter-notebook daml-part2.ipynb
