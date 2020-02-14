.. _develop:

Developer's Guide
=================

.. warning:: This section is still very incomplete!

This section describes things that may be of interest to developers of this codebase: its
structure, development standards, etc.

Project vs. Package
-------------------

Throughout this codebase, the term `project` will be used to refer to the totality of everything hosed within the
codebase's repository.  The term `package` will be used to refer to a specific Python package (found within the
``python`` directory).  Things are structured such that multiple packages can be housed within a single project,
although that is not a particularly common situation.

Testing
-------

To run the test suite for this project, execute the following from the root directory:

.. code-block:: console

    make tests

Alternatively, if the following is executed:

.. code-block:: console

    make tests-py-tox

``tox`` will be used to run ``pytest`` under all supported versions of python.  A few quick points
about ``tox`` however.  First: it does not play well with Anaconda.  While Anaconda can be
carefully configured to work with ``tox``, it can be tricky to get this right.  So, it is
generally recommended not to have any Ananconda environments activated when running the tests. A
variety of other environment managers are available, but the simplest and most
straight-forward environment manager to use is probably ``pyenv``.  If installed on your system,
make sure that all the needed versions of python have been installed within it (using ``pyenv
install X.Y.Z``, where ``X.Y.Z`` is a desired python version number) and create a development
environment with access to all the needed versions of python as follows:

.. code-block:: console

    pyenv virtualenv <environment_name>
    pyenv local <environment_name> X1.Y1.Z1 X2.Y2.Z2 X3.Y3.Z3 ... XN.YN.ZN

.. note:: Sometimes the ``tox`` caches can fail to update.  If in doubt, run ``make clean`` and try again.

.. warning:: Running validation tests under ``tox`` can be problematic, since the validation files are generated only using one version of Python while version changes can result in subtle pickling differences.

Possible problems and solutions
-------------------------------

Common Errors
^^^^^^^^^^^^^

``Command "python setup.py egg_info" failed with error code 1``
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Update setuptools as follows: ``pip install --upgrade setuptools``

