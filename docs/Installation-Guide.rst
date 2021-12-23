Installation Guide
==================

This is a guide to install Graphormer. Currently Graphormer supports intallation on Linux only.

Linux
~~~~~

On Linux, Graphormer can be easily installed with the install.sh script with prepared python environments.

1. Please use Python3.9 for Graphormer. It is recommended to create a virtual environment with `conda <https://docs.conda.io/en/latest/>`__ or `virtualenv <https://virtualenv.pypa.io/en/latest/>`__.
For example, to create and activate a conda environment with Python3.9

.. code::

    conda create -n graphormer python=3.9
    conda activate graphormer

2. Run the following commands:

.. code::

    git clone --recursive https://github.com/microsoft/Graphormer.git
    cd Graphormer
    bash install.sh
