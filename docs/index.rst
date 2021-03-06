ADACS Development of PRISM
==========================

This repository documents the GPU optimisation investigation conducted by `ADACS <https://adacs.org.au/>`_ on the model emulation code `PRISM <https://github.com/1313e/PRISM>`_ as part of `Ellert van der Velden's <https://github.com/1313e>`_ 2019A ADACS Software Support Project.

This work was carried-out by ADACS developer `Gregory B. Poole <https://github.com/gbpoole>`_.

Implementation Description
--------------------------

The approach taken to move the `mlxtend <https://rasbt.github.io/mlxtend/>`_ least-squares computation invoked by PRISM to the GPU has involved the following steps:

    1. Some increased coverage of the :code:`scikit-cuda` library's coverage of the Magma API.  This library's maintainer has kindly implemented these changes at the request of ADACS.

    2. Slight changes to the `mlxtend` code base.  This has involved:

      1. Addition of a :code:`permit_estimator_override(est,func)` function which calls function :code:`func` unless an estimator method of the same name exists, in which case it calls that method:

        .. code-block:: python
        
           def permit_estimator_override(est, func):
               est_atr = getattr(est, func.__name__, None)
               if callable(est_atr):
                   return est_atr
               else:
                   return func

      2. Wrapping of all function calls to :code:`cross_val_score` and :code:`Parrallel` in an estimator (for this work, that in :code:`sequential_feature_selector.py`) with this function.  For example, this:

        .. code-block:: python
        
           parallel = Parallel(n_jobs=n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)
        
        becomes this
        
        .. code-block:: python
        
           parallel = permit_estimator_override(self, Parallel)(n_jobs=n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)

      .. note::
          These changes have not been pushed into the official `mlxtend` repository.  A version can be found `here <https://github.com/ADACS-Australia/ADACS-SS18B-EvdVelden-mlxtend>`_, which can be installed as follows:

          .. code-block:: bash

              pip install git+https://github.com/ADACS-Australia/ADACS-SS18B-EvdVelden-mlxtend@adacs

    3. Implementation of a metaclass (provided by this repository; see :py:mod:`prism_adacs.mlxtend_cuda.estimator <prism_adacs.mlxtend_cuda.estimator>` for code and an example of it's use) which provides all the needed support for CUDA-enabled GPU execution of `mlxtend` estimators.  The user need-only:

        #. construct their estimators as they usual do, but with :code:`metaclass=CudaEstimator` instead of :code:`metaclass=type` (Python's default).

        #. add a :code:`fit` method to their estimator, which gets called in place of the estimator's normal fitting routine.

Test Problem
------------

For benchmarking, a 4-dimenstional "xsinx" function of the form :math:`F(\vec{x})=\prod_{i=1}^{4}(a_ix_i)sin(b_ix_i); \vec{a}=\vec{b}=[2,2]` was used.  A set of random samples :math:`\vec{s}` were selected randomly in the range :math:`[0,10]` and a polynomial fit performed to the values :math:`S(\vec{s})=F(\vec{s})+G(\sigma=0.5)`, where :math:`G(\sigma)` is a Gaussian-random value (centroid at the origin; variance of :math:`\sigma`).  Figure 1 illustrates a 2-dimensional example.

.. figure:: static/fit_example.png

   Figure 1: An example fit to :math:`S(\vec{x})` for the case of 250 randomly selected points and a fit polynomial order of 7.

Results
-------

Due to incomplete coverage of the :code:`*_gpu` functions of the `MAGMA <https://developer.nvidia.com/magma>`_ library's API, the current implementation presented here is far from optimal due to unnecessary host->device communications.  

We have investigated benchmarks for two cases:

    #. The "current" implementation presented in this repository

    #. An optimistic "best case" scenario where SVD calculations are assumed to be of no expense and device communication are minimised.

Cases with fit polynomial orders of 5, 7 and 10 have been considered with a number of samples ranging from 100 to 100000.  Larger problems were not run because they exceeded the maximum wallclock on `OzSTAR <https://supercomputing.swin.edu.au/ozstar/>`_, where all testing was conducted.

The runtimes of the current implementation are always poorer when utilizing the GPU, reflecting the cost of suboptimal host-device communication.  Only for very large problems are accellerations realised for the best case scenario and even then, only by a maximum factor of approximately 2.  

These results are presented in the figure below.

.. figure:: static/timing_grid.png

   Figure 2: GPU speed-ups for the current implementation and for a best-case scenario with minimised host-device communication and computing costs dictated only by least-squares fitting.  Values less than 1 represent slower run-times; values greater than 1 represent accellerated run-times.

Table of Contents
-----------------

.. prism_adacs documentation master file

.. toctree::
   :maxdepth: 3

   install.rst
   src/Python_execs.rst
   src/Python_API.rst
   development.rst

* :ref:`genindex`
