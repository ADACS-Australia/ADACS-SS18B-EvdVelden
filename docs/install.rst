.. _Installation:

Installation
============

This section describes how to download and install this project.

Obtaining the code
------------------

This project's code can be obtained from its `Github repository <https://github.com/gbpoole/prism_adacs>`_.  To download it, first select a destination directory for the repository (for the example code below, we assume `bash` or `zsh` syntax is valid for your shell):

.. code-block:: console

    export INSTALL_DIR=/set/your/install/directory/here

and then run the following:

.. code-block:: console

    cd ${INSTALL_DIR}
    git clone https://github.com/gbpoole/prism_adacs

Installing the code
-------------------

.. note:: A makefile is provided in the project directory to ease the use of this software project.  Type `make help` for a list of options.

Once the code has been obtained, management of the project can be achieved most simply using the provided Make file.  To install all dependencies and build/install all request codes, run the following:

.. code-block:: console

    cd ${INSTALL_DIR}/prism_adacs
    make init
    make install

.. warning:: Make sure that the `make init` line is run first-thing before installing.  It will ensure that all needed dependencies are present in your current Python environment.
    Make sure to re-run this line before re-installing, if you change Python environments.

Building documentation
----------------------

To build the project's documentation, run the following:

.. code-block:: console

    cd ${INSTALL_DIR}/prism_adacs
    make docs

The resulting documentation will be placed in the directory `${INSTALL_DIR}/prism_adacs/docs/_build/`.  The following will open the documentation in the browser:

.. code-block:: console

    cd ${INSTALL_DIR}/prism_adacs
    make docs-open
