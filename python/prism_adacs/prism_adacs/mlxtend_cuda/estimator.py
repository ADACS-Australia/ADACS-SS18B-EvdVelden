"""The code in this submodule is effectively the code that would have to be
added to PRISM for the Cuda-enabled estimator to work."""

from __future__ import division, print_function, absolute_import

import os
import sys
from distutils.util import strtobool
from abc import ABCMeta
from functools import wraps
import time
import numbers
import importlib
import warnings

from collections import deque
from itertools import count

from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork
from scipy.linalg.misc import LinAlgError, _datacopied, LinAlgWarning
from scipy.linalg.decomp import _asarray_validated

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable
from sklearn.metrics.scorer import check_scoring, _check_multimetric_scoring
from sklearn.model_selection._split import check_cv
from sklearn.externals.joblib import delayed

from sklearn.utils.metaestimators import _safe_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.externals.six.moves import zip

import scipy.sparse as sp

from sklearn.utils import check_X_y

import numpy as np

# Infer the name of this package from the path of __file__
package_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
package_name = os.path.basename(package_root_dir)

# Make sure that what's in this path takes precedence
# over an installed version of the project
sys.path.insert(0, package_parent_dir)

# Import internal modules
pkg = importlib.import_module(package_name)
prj = importlib.import_module(package_name + '._internal.project')

# Import package submodules
pkg.import_submodules()


default_lapack_driver = 'gelsd'

# Import Cuda packages and establish a context with the device
_cuda_enabled = strtobool(os.environ.get("CUDA_ENABLED", "1"))
if _cuda_enabled:
    try:
        import pycuda.autoinit
        import pycuda.driver as drv
        import pycuda.gpuarray as gpuarray
        import skcuda.magma as magma
        magma.magma_init()
        context = True
    except ImportError:
        context = False
else:
    context = False

class _timer(object):
    """This class will keep track of a bunch of timing information for testing purposes."""
    def __init__(self):
        self._vals = {}

    def _create(self,name):
        if not name in self._vals:
            self._vals[name] = [None,0.]

    def start(self,name):
        if name in self._vals:
            self._vals[name][0]=time.time()
        else:
            self._create(name)
            self.start(name)

    def stop(self,name):
        _time = time.time()
        if name in self._vals:
            if not self._vals[name][0] is None:
                self._vals[name][1] += _time-self._vals[name][0]
            else:
                pkg.log.error(Exception("timer stop method called without having been started for item named {%s}."%(name)))
        else:
            pkg.log.error(Exception("timer stop method called for non-existant item {%s}."%(name)))

    def print(self):
        flag_init = False
        for key in self._vals.keys():
            if not flag_init:
                pkg.log.open("Timing information:")
                flag_init = True
            pkg.log.comment("%s = %.2f seconds"%(key,self._vals[key][1]))
        if flag_init:
            pkg.log.close()

timer = _timer()

class CudaEstimator(ABCMeta):
    """This metaclass supplies all the needed support for the Cuda-enabled
    estimator presented in this submodule."""

    def __prepare__(name, *args, **kwargs):
        """Initialise the dictionary that gets passed to __new___.

        This is the only method that gets sourced before the class code
        is executed, so it needs to be done here, not in __new___.
        """
        result = dict()

        # List of method overrides that the generated subclass must provide
        result['_required_method_overrides'] = list()
        if context:
            result['_required_method_overrides'].append('fit')

        return result

    def __new__(mcs, name, bases, dct):

        # Perform super-metaclass construction
        new_class = super(CudaEstimator, mcs).__new__(mcs, name, bases, dct)

        # This will override mlxtend calls to cross_val_score()
        def cross_val_score(cls, estimator, X, y=None, groups=None, scoring=None, cv='warn',
                            n_jobs=None, verbose=0, fit_params=None,
                            pre_dispatch='2*n_jobs', error_score='raise-deprecating'):

            # To ensure multimetric format is not supported
            scorer = check_scoring(estimator, scoring=scoring)

            def cross_validate(estimator, X, y=None, groups=None, scoring=None, cv='warn',
                               n_jobs=None, verbose=0, fit_params=None,
                               pre_dispatch='2*n_jobs', return_train_score=False,
                               return_estimator=False, error_score='raise-deprecating'):

                X, y, groups = indexable(X, y, groups)

                cv = check_cv(cv, y, classifier=is_classifier(estimator))
                scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

                def _score(estimator, X_test, y_test, scorer, is_multimetric=False):

                    if is_multimetric:
                        return _multimetric_score(estimator, X_test, y_test, scorer)
                    else:
                        if y_test is None:
                            score = scorer(estimator, X_test)
                        else:
                            score = scorer(estimator, X_test, y_test)

                        if hasattr(score, 'item'):
                            try:
                                # e.g. unwrap memmapped scalars
                                score = score.item()
                            except ValueError:
                                # non-scalar?
                                pass

                        if not isinstance(score, numbers.Number):
                            raise ValueError("scoring must return a number, got %s (%s) "
                                             "instead. (scorer=%r)"
                                             % (str(score), type(score), scorer))

                    return score

                def _multimetric_score(estimator, X_test, y_test, scorers):
                    """Return a dict of score for multimetric scoring."""
                    scores = {}

                    for name, scorer in scorers.items():
                        if y_test is None:
                            score = scorer(estimator, X_test)
                        else:
                            score = scorer(estimator, X_test, y_test)

                        if hasattr(score, 'item'):
                            try:
                                # e.g. unwrap memmapped scalars
                                score = score.item()
                            except ValueError:
                                # non-scalar?
                                pass
                        scores[name] = score

                        if not isinstance(score, numbers.Number):
                            raise ValueError("scoring must return a number, got %s (%s) "
                                             "instead. (scorer=%s)"
                                             % (str(score), type(score), name))
                    return scores

                def _aggregate_score_dicts(scores):

                    out = {}
                    for key in scores[0]:
                        out[key] = np.asarray([score[key] for score in scores])
                    return out

                def _fit_and_score(estimator, X, y, scorer, train, test, verbose,
                                   parameters, fit_params, return_train_score=False,
                                   return_parameters=False, return_n_test_samples=False,
                                   return_times=False, return_estimator=False,
                                   error_score='raise-deprecating'):

                    start_time = time.time()

                    if verbose > 1:
                        if parameters is None:
                            msg = ''
                        else:
                            msg = '%s' % (', '.join('%s=%s' % (k, v)
                                                    for k, v in parameters.items()))
                        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

                    # Adjust length of sample weights
                    fit_params = fit_params if fit_params is not None else {}
                    fit_params = dict([(k, _index_param_value(X, v, train))
                                       for k, v in fit_params.items()])

                    train_scores = {}
                    if parameters is not None:
                        estimator.set_params(**parameters)

                    X_train, y_train = _safe_split(estimator, X, y, train)
                    X_test, y_test = _safe_split(estimator, X, y, test, train)

                    is_multimetric = not callable(scorer)
                    n_scorers = len(scorer.keys()) if is_multimetric else 1

                    try:
                        #########################################
                        ############ FIT CALLED HERE ############
                        #########################################
                        if y_train is None:
                            estimator.fit(X_train, **fit_params)
                        else:
                            estimator.fit(X_train, y_train, **fit_params)
                        #########################################
                    except Exception as e:
                        # Note fit time as time until error
                        fit_time = time.time() - start_time
                        score_time = 0.0
                        if error_score == 'raise':
                            raise
                        elif error_score == 'raise-deprecating':
                            warnings.warn("From version 0.22, errors during fit will result "
                                          "in a cross validation score of NaN by default. Use "
                                          "error_score='raise' if you want an exception "
                                          "raised or error_score=np.nan to adopt the "
                                          "behavior from version 0.22.",
                                          FutureWarning)
                            raise
                        elif isinstance(error_score, numbers.Number):
                            if is_multimetric:
                                test_scores = dict(zip(scorer.keys(),
                                                       [error_score, ] * n_scorers))
                                if return_train_score:
                                    train_scores = dict(zip(scorer.keys(),
                                                            [error_score, ] * n_scorers))
                            else:
                                test_scores = error_score
                                if return_train_score:
                                    train_scores = error_score
                            warnings.warn("Estimator fit failed. The score on this train-test"
                                          " partition for these parameters will be set to %f. "
                                          "Details: \n%s" %
                                          (error_score, format_exception_only(type(e), e)[0]),
                                          FitFailedWarning)
                        else:
                            raise ValueError("error_score must be the string 'raise' or a"
                                             " numeric value. (Hint: if using 'raise', please"
                                             " make sure that it has been spelled correctly.)")

                    else:
                        fit_time = time.time() - start_time
                        # _score will return dict if is_multimetric is True
                        test_scores = _score(estimator, X_test, y_test, scorer, is_multimetric)
                        score_time = time.time() - start_time - fit_time
                        if return_train_score:
                            train_scores = _score(estimator, X_train, y_train, scorer,
                                                  is_multimetric)

                    if verbose > 2:
                        if is_multimetric:
                            for scorer_name, score in test_scores.items():
                                msg += ", %s=%s" % (scorer_name, score)
                        else:
                            msg += ", score=%s" % test_scores
                    if verbose > 1:
                        total_time = score_time + fit_time
                        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
                        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

                    ret = [train_scores, test_scores] if return_train_score else [test_scores]

                    if return_n_test_samples:
                        ret.append(_num_samples(X_test))
                    if return_times:
                        ret.extend([fit_time, score_time])
                    if return_parameters:
                        ret.append(parameters)
                    if return_estimator:
                        ret.append(estimator)

                    return ret

                if not context:
                    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                                        pre_dispatch=pre_dispatch)
                else:
                    parallel = cls.Parallel()

                # We clone the estimator to make sure that all the folds are
                # independent, and that it is pickle-able.
                scores = parallel(
                    delayed(_fit_and_score)(
                        clone(estimator), X, y, scorers, train, test, verbose, None,
                        fit_params, return_train_score=return_train_score,
                        return_times=True, return_estimator=return_estimator,
                        error_score=error_score)
                    for train, test in cv.split(X, y, groups))

                zipped_scores = list(zip(*scores))
                if return_train_score:
                    train_scores = zipped_scores.pop(0)
                    train_scores = _aggregate_score_dicts(train_scores)
                if return_estimator:
                    fitted_estimators = zipped_scores.pop()
                test_scores, fit_times, score_times = zipped_scores
                test_scores = _aggregate_score_dicts(test_scores)

                ret = {}
                ret['fit_time'] = np.array(fit_times)
                ret['score_time'] = np.array(score_times)

                if return_estimator:
                    ret['estimator'] = fitted_estimators

                for name in scorers:
                    ret['test_%s' % name] = np.array(test_scores[name])
                    if return_train_score:
                        key = 'train_%s' % name
                        ret[key] = np.array(train_scores[name])

                return ret

            cv_results = cross_validate(estimator=estimator, X=X, y=y, groups=groups,
                                        scoring={'score': scorer}, cv=cv,
                                        n_jobs=n_jobs, verbose=verbose,
                                        fit_params=fit_params,
                                        pre_dispatch=pre_dispatch,
                                        error_score=error_score)
            return cv_results['test_score']

        # This will override the mlxtend calls to Parallel()
        def Parallel():

            # Grabbed from:
            # https://stackoverflow.com/questions/5384570/whats-the-shortest-way-to-count-the-number-of-items-in-a-generator-iterator
            def ilen(it):
                # Make a stateful counting iterator
                cnt = count()
                # zip it with the input iterator, then drain until input exhausted at C level
                deque(zip(it, cnt), 0)  # cnt must be second zip arg to avoid advancing too far
                # Since count 0 based, the next value is the count
                return next(cnt)

            def parallel(iterable):
                work = []
                # TODO: Parallelize this
                for i_iterator, iterator in enumerate(iter(iterable)):
                    work.append(iterator[0](*iterator[1], **iterator[2]))
                return work
            return parallel

        # Rewrite __init__ to validate presence of required overrides and initialize magma
        old_init = new_class.__init__

        @wraps(old_init)
        def new_init(self, *args, **kwargs):

            # Call its constructor
            Cls = self.__class__
            super(Cls, self).__init__(*args, **kwargs)

            # This property creates the necessary information about the Cuda context
            self.context = context

        global context
        if context:
            setattr(new_class, cross_val_score.__name__, classmethod(cross_val_score))
            setattr(new_class, Parallel.__name__, staticmethod(Parallel))
            setattr(new_class, '__init__', new_init)

        # Class that the decorated class is inheriting from
        parent = new_class.__bases__[-1]

        # Check that required method overrides have been defined
        for required in new_class._required_method_overrides:
            self_method = getattr(new_class, required, None)
            if not self_method:
                raise AttributeError("Required override method {%s} not provided." % (required))

            parent_method = getattr(parent, required, None)
            if not parent_method:
                raise AttributeError(
                    "Required override {%s} not provided by parent class." % (required))

            if self_method.__code__ is parent_method.__code__:
                raise AttributeError(
                    "Required subclass override {%s} does not differ from parent's method." % (required))

        return new_class


class cuLR(LR, metaclass=CudaEstimator):
    """This class provides a Cuda-enabled linear regression estimator, derived
    from the scikit-learn linear regressor.

    In the case where a Cuda context has not been initiated (either
    because Cuda is not available or has been turned-off), it defaults
    simply to the scokit-learn form
    """

    def fit(self, X, y, sample_weight=None):
        """Fit linear model.

        Derived-from - and meant to override - the fit method of the base class.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data

        y : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
            Individual weights for each sample

            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.

        Returns
        -------
        self : returns an instance of self.
        """

        def lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False,
                  check_finite=True, lapack_driver=None):
            """
            Compute least-squares solution to equation Ax = b.
            Compute a vector x such that the 2-norm ``|b - A x|`` is minimized.

            This code was adapted from the Scipy distribution: https://github.com/scipy/scipy/blob/v1.2.1/scipy/linalg/basic.py#L1047-L1264

            Parameters
            ----------
            a : (M, N) array_like
                Left hand side matrix (2-D array).
            b : (M,) or (M, K) array_like
                Right hand side matrix or vector (1-D or 2-D array).
            cond : float, optional
                Cutoff for 'small' singular values; used to determine effective
                rank of a. Singular values smaller than
                ``rcond * largest_singular_value`` are considered zero.
            overwrite_a : bool, optional
                Discard data in `a` (may enhance performance). Default is False.
            overwrite_b : bool, optional
                Discard data in `b` (may enhance performance). Default is False.
            check_finite : bool, optional
                Whether to check that the input matrices contain only finite numbers.
                Disabling may give a performance gain, but may result in problems
                (crashes, non-termination) if the inputs do contain infinities or NaNs.
            lapack_driver : str, optional
                Which LAPACK driver is used to solve the least-squares problem.
                Options are ``'gelsd'``, ``'gelsy'``, ``'gelss'``. Default
                (``'gelsd'``) is a good choice.  However, ``'gelsy'`` can be slightly
                faster on many problems.  ``'gelss'`` was used historically.  It is
                generally slow but uses less memory.
                .. versionadded:: 0.17.0
            Returns
            -------
            x : (N,) or (N, K) ndarray
                Least-squares solution.  Return shape matches shape of `b`.
            residues : (0,) or () or (K,) ndarray
                Sums of residues, squared 2-norm for each column in ``b - a x``.
                If rank of matrix a is ``< N`` or ``N > M``, or ``'gelsy'`` is used,
                this is a length zero array. If b was 1-D, this is a () shape array
                (numpy scalar), otherwise the shape is (K,).
            rank : int
                Effective rank of matrix `a`.
            s : (min(M,N),) ndarray or None
                Singular values of `a`. The condition number of a is
                ``abs(s[0] / s[-1])``. None is returned when ``'gelsy'`` is used.
            Raises
            ------
            LinAlgError
                If computation does not converge.
            ValueError
                When parameters are wrong.
            See Also
            --------
            optimize.nnls : linear least squares with non-negativity constraint
            Examples
            --------
            >>> from scipy.linalg import lstsq
            >>> import matplotlib.pyplot as plt
            Suppose we have the following data:
            >>> x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
            >>> y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])
            We want to fit a quadratic polynomial of the form ``y = a + b*x**2``
            to this data.  We first form the "design matrix" M, with a constant
            column of 1s and a column containing ``x**2``:
            >>> M = x[:, np.newaxis]**[0, 2]
            >>> M
            array([[  1.  ,   1.  ],
                   [  1.  ,   6.25],
                   [  1.  ,  12.25],
                   [  1.  ,  16.  ],
                   [  1.  ,  25.  ],
                   [  1.  ,  49.  ],
                   [  1.  ,  72.25]])
            We want to find the least-squares solution to ``M.dot(p) = y``,
            where ``p`` is a vector with length 2 that holds the parameters
            ``a`` and ``b``.
            >>> p, res, rnk, s = lstsq(M, y)
            >>> p
            array([ 0.20925829,  0.12013861])
            Plot the data and the fitted curve.
            >>> plt.plot(x, y, 'o', label='data')
            >>> xx = np.linspace(0, 9, 101)
            >>> yy = p[0] + p[1]*xx**2
            >>> plt.plot(xx, yy, label='least squares fit, $y = a + bx^2$')
            >>> plt.xlabel('x')
            >>> plt.ylabel('y')
            >>> plt.legend(framealpha=1, shadow=True)
            >>> plt.grid(alpha=0.25)
            >>> plt.show()
            """

            a1 = _asarray_validated(a, check_finite=check_finite)
            b1 = _asarray_validated(b, check_finite=check_finite)
            if len(a1.shape) != 2:
                raise ValueError('expected matrix')
            m, n = a1.shape

            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1
            if m != b1.shape[0]:
                raise ValueError('incompatible dimensions')
            if m == 0 or n == 0:  # Zero-sized problem, confuses LAPACK
                x = np.zeros((n,) + b1.shape[1:], dtype=np.common_type(a1, b1))
                if n == 0:
                    residues = np.linalg.norm(b1, axis=0) ** 2
                else:
                    residues = np.empty((0,))
                return x, residues, 0, np.empty((0,))

            driver = lapack_driver
            if driver is None:
                global default_lapack_driver
                driver = default_lapack_driver
            if driver not in ('gelsd', 'gelsy', 'gelss'):
                raise ValueError('LAPACK driver "%s" is not found' % driver)

            lapack_func, lapack_lwork = get_lapack_funcs((driver,
                                                          '%s_lwork' % driver),
                                                         (a1, b1))
            real_data = True if (lapack_func.dtype.kind == 'f') else False

            if m < n:
                # need to extend b matrix as it will be filled with
                # a larger solution matrix
                if len(b1.shape) == 2:
                    b2 = np.zeros((n, nrhs), dtype=lapack_func.dtype)
                    b2[:m, :] = b1
                else:
                    b2 = np.zeros(n, dtype=lapack_func.dtype)
                    b2[:m] = b1
                b1 = b2

            overwrite_a = overwrite_a or _datacopied(a1, a)
            overwrite_b = overwrite_b or _datacopied(b1, b)

            if cond is None:
                cond = np.finfo(lapack_func.dtype).eps

            a1_wrk = np.copy(a1)
            b1_wrk = np.copy(b1)
            lwork, iwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
            x_check, s_check, rank_check, info = lapack_func(a1_wrk, b1_wrk, lwork, iwork, cond, False, False)

            driver = 'gelss'
            if driver in ('gelss', 'gelsd'):
                if driver == 'gelss':
                    if not context:
                        a1_wrk = np.copy(a1)
                        b1_wrk = np.copy(b1)
                        lwork, iwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
                        x, s, rank, info = lapack_func(a1_wrk, b1_wrk, lwork, iwork, cond, False, False)
                    else:
                        try:
                            # Check that we aren't dealing with an underconstrained problem ...
                            if m < n:
                                pkg.log.error(Exception("Underconstrained problems not yet supported by Magma."))

                            # Initialize
                            a1_trans = np.copy(a1, order='F')
                            a1_gpu = gpuarray.to_gpu(a1_trans)

                            # Note that the result for 'x' gets written to the vector inputted for b
                            x_trans = np.copy(b1, order='F')
                            x_gpu = gpuarray.to_gpu(x_trans)

                            # Init singular-value decomposition (SVD) output & buffer arrays
                            s = np.zeros(min(m, n), np.float32)
                            u = np.zeros((m, m), np.float32)
                            vh = np.zeros((n, n), np.float32)

                            # Query and allocate optimal workspace
                            # n.b.: - the result for 'x' gets written to the input vector for b, so we just label b->x
                            #       - assume magma variables lda=ldb=m throughout here
                            lwork_SVD = magma.magma_sgesvd_buffersize(
                                'A', 'A', m, n, a1_trans.ctypes.data, m, s.ctypes.data, u.ctypes.data, m, vh.ctypes.data, n)

                            # For some reason, magma_sgels_buffersize() does not return the right value for large problems, so
                            # we compute the values used for the validation check (see Magma SGELS documentation) directly and use that
                            #lwork_LS = magma.magma_sgels_buffersize('n', m, n, nrhs, a1_trans.ctypes.data, m, x_trans.ctypes.data, m)
                            nb = magma.magma_get_sgeqrf_nb(m, n)
                            check = (m - n + nb) * (nrhs + nb) + nrhs * nb
                            lwork_LS = check

                            # Allocate workspaces
                            hwork_SVD = np.zeros(lwork_SVD, np.float32, order='F')
                            hwork_LS = np.zeros(lwork_LS, np.float32)

                            # Compute SVD
                            timer.start("SVD")
                            magma.magma_sgesvd('A', 'A', m, n, a1_trans.ctypes.data, m, s.ctypes.data,
                                               u.ctypes.data, m, vh.ctypes.data, n,
                                               hwork_SVD.ctypes.data, lwork_SVD)
                            timer.stop("SVD")

                            # Note, the use of s_i>rcond here; this is meant to select
                            # values that are effectively non-zero.  Results will depend
                            # somewhat on the choice for this value.  This criterion was
                            # adopted from that utilized by scipy.linalg.basic.lstsq()
                            rcond = np.finfo(lapack_func.dtype).eps * s[0]
                            rank = sum(1 for s_i in s if s_i > rcond)

                            # Run LS solver
                            timer.start("LS")
                            magma.magma_sgels_gpu('n', m, n, nrhs, a1_gpu.gpudata, m, x_gpu.gpudata, m,
                                                  hwork_LS.ctypes.data, lwork_LS)
                            timer.stop("LS")

                            # Unload result from GPU
                            x = x_gpu.get()

                        except magma.MagmaError as e:
                            info = e._status
                        else:
                            info = 0

                elif driver == 'gelsd':
                    if real_data:
                        if not context:
                            raise Exception(
                                "For some reason, the CUDA implementation of fit() is being called when context is False.")
                        else:
                            raise Exception("gelsd not supported using Cuda yet")
                    else:  # complex data
                        raise LinAlgError("driver=%s not yet supported for complex data" % (driver))
                if info > 0:
                    raise LinAlgError("SVD did not converge in Linear Least Squares")
                if info < 0:
                    raise ValueError('illegal value in %d-th argument of internal %s'
                                     % (-info, lapack_driver))
                resids = np.asarray([], dtype=x.dtype)
                if m > n:
                    x1 = x[:n]
                    if rank == n:
                        resids = np.sum(np.abs(x[n:]) ** 2, axis=0)
                    x = x1

            elif driver == 'gelsy':
                raise LinAlgError("driver=%s not yet supported" % (driver))

            #pkg.log.close("Done", time_elapsed=True)
            return x, resids, rank, s

        n_jobs_ = self.n_jobs
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)

        if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        if sp.issparse(X):
            raise Exception("Sparse matrices not supported yet for Cuda implementation.")
        else:
            ###############################
            self.coef_, self._residues, self.rank_, self.singular_ = lstsq(X, y)
            ###############################
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self
