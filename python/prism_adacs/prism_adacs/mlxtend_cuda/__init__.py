"""This package provides a Cuda-enabled estimator that can be used with a
suitably modified version of Mlxtend to perform polynomial fits of the sort
employed by the model emulator PRISM.  Three submodules are involved:

#) ``prism_adacs.mlxtend.estimator``:  this is effectively the code that
would have to be added to PRISM for the Cuda-enabled estimator to work
#) ``prism_adacs.mlxtend.fit``:        this constructs and runs a
Scikit-learn pipeline of the sort used by PRISM, to facilitate testing
of the Cuda-enabled estimator #) ``prism_adacs.mlxtend.fit_inputs``:
this provides model data and fitting function inputs needed by the
``fit`` submodule, to facilitate testing of the Cuda-enabled estimator
"""
