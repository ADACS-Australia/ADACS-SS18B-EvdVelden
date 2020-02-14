"""This provides model data and fitting function inputs needed by the ``fit``
submodule, to facilitate testing of the Cuda-enabled estimator."""

import sys
import os
import importlib
import numpy as np
import random

# Infer the name of this package from the path of __file__
package_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
package_name = os.path.basename(package_root_dir)

# Make sure that what's in this path takes precedence
# over an installed version of the project
sys.path.insert(0, package_parent_dir)

# Import needed internal modules
pkg = importlib.import_module(package_name)
prj = importlib.import_module(package_name + '._internal.project')

# Import package submodules
pkg.import_submodules()

#: Faster than random.shuffle, and will accept any sequence, not just indexables (like lists).
#: Obtained from: https://pythonexample.com/code/python-shuffle-not-in-place/
shuffled = lambda seq, rnd=random.random: sorted(seq, key=lambda _: rnd())

# Available models


def gauss(X, amplitude=1., coeffs=[[0., 1.]]):
    """Gaussian function, to be used with the model class."""
    assert(np.shape(X)[0] == np.shape(coeffs)[0])
    return amplitude * np.exp(np.sum([-(0.5 * (x_i - cen_i) / sigma_i)**2 for x_i, [cen_i, sigma_i] in zip(X, coeffs)]))


def xsinx(X, amplitude=1., coeffs=[[1, 1]]):
    """`xsinx` function, to be used with the model class."""
    assert(np.shape(X)[0] == np.shape(coeffs)[0])
    d = np.sqrt(np.sum([((x_i - cen_i)**2) / sigma_i for x_i, [cen_i, sigma_i] in zip(X, coeffs)]))
    return amplitude * d * np.sin(d)

# Model base class


class model(metaclass=pkg.validation.metaclass):
    """This class exposes models which can be used to generate model data and
    then subsequently perform fits to it."""

    # xsinx models
    validation_grid.add(name='xsinx', coeffs=[[1, 1]])          # 1D, offcen
    validation_grid.add(name='xsinx', coeffs=[[1, 1], [1, 1]])  # 2D, offcen
    validation_grid.add(name='xsinx', coeffs=[[0, 1]])          # 1D, cen
    validation_grid.add(name='xsinx', coeffs=[[0, 1], [0, 1]])  # 2D, cen

    # Gaussian models
    validation_grid.add(name='gauss', coeffs=[[1, 1]])          # 1D, offcen
    validation_grid.add(name='gauss', coeffs=[[1, 1], [1, 1]])  # 2D, offcen
    validation_grid.add(name='gauss', coeffs=[[0, 1]])          # 1D, cen
    validation_grid.add(name='gauss', coeffs=[[0, 1], [0, 1]])  # 2D, cen

    # High-dimension models
    validation_grid.add(name='gauss', coeffs=[[1, 1], [1, 1], [1, 1]])          # 3D, offcen
    validation_grid.add(name='gauss', coeffs=[[1, 1], [1, 1], [1, 1], [1, 1]])  # 4D, offcen

    def __init__(self, name='xsinx', coeffs=[[1, 1]]):

        # Save inputs
        self.name = str(name)
        self.coeffs = np.array(coeffs)

        # Validate coefficients
        assert(len(np.shape(self.coeffs)) == 2)

        # Derive model dimension from coeffs
        self.n_dim = np.shape(self.coeffs)[0]

        if self.name == 'xsinx':
            self._func = xsinx
        elif self.name == 'gauss':
            self._func = gauss
        else:
            pkg.log.error(Exception("Unsupported model: %s" % (self.name)))

    def __call__(self, X):
        return self._func(X, coeffs=self.coeffs)

    def __str__(self):
        """Return a string representation of the parameter set."""
        return "%s (n_dim=%d, coeffs=%s)" % (self.name, self.n_dim, str(self.coeffs).replace('\n', ''))


class dataset(metaclass=pkg.validation.metaclass):
    """This class provides the inputs to the polynomial fitting pipeline
    presented in the ``prism_adacs.mlxtend.fit`` submodule."""

    # For each model to be validated, test a set of datasets ...
    for model in model.validation_grid:
        # Add noise to high-D model and use lots of data points.
        if np.shape(model.kwargs_in['coeffs'])[0] > 2:
            validation_grid.add(f_model=model, sigma=0.5, n_fit=500, n_plot=100, x_range=[0, 10])
        # ... for all other models ...
        else:
            # No noise
            validation_grid.add(f_model=model, sigma=0., n_fit=20, n_plot=100, x_range=[0, 10])  # sparse
            validation_grid.add(f_model=model, sigma=0., n_fit=50, n_plot=100, x_range=[0, 10])  # dense

            # Add noise
            validation_grid.add(f_model=model, sigma=0.5, n_fit=20, n_plot=100, x_range=[0, 10])  # sparse
            validation_grid.add(f_model=model, sigma=0.5, n_fit=50, n_plot=100, x_range=[0, 10])  # dense

    # Validate the dataset inputs to make sure that tests are being performed on the exact-same inputs
    validation_members.add('X', atol=0)
    validation_members.add('Y', atol=0)

    def __init__(self, f_model=model(), sigma=0.5, n_fit=20, n_plot=100, x_range=[0, 10]):
        """
        :param f_model: a model instance
        :param sigma: variance to add to the instantiated data
        :param n_fit: number of points to be generated
        :param n_plot: number of points to be used in each dimension for plotting
        :param x_range: an ``f_model.n_dim``-dimensional array specifying the ordinate ranges of the points
        """

        pkg.log.open("Generating dataset for %dD %s model..." % (f_model.n_dim, f_model.name))
        self.f_model = f_model
        self.sigma = sigma
        self.n_fit = n_fit
        self.n_plot = n_plot
        self.x_range = np.array(x_range)

        # Validate inputs
        assert (self.n_plot ** self.f_model.n_dim >= self.n_fit)

        # Initialize random seeds (keep fixed for reproducability)
        random.seed(10247843)
        np.random.seed(3729365)

        if f_model.n_dim < 3:
            # Generate 2 sets of samples; 1 set to fit to & 1 set for plotting the fit
            self.x_model = (1. / self.n_plot) * (self.x_range[1] - self.x_range[0]) * (np.indices(
                self.f_model.n_dim * (self.n_plot,), dtype=float).T.reshape(-1, self.f_model.n_dim) + 1 - self.x_range[0])

            # Sub-select and randomize (needed for effective cross-validation)
            self.X = np.asarray(shuffled(np.copy(self.x_model))[:n_fit], dtype=np.float32)
        else:
            self.x_model = None
            self.X = (self.x_range[1] - self.x_range[0]) * \
                np.random.randn(n_fit, f_model.n_dim).astype('f') + self.x_range[0]

        # Generate training points
        self.Y = np.asarray([self.f_model(X_i) for X_i in self.X[:, ...]], dtype=np.float32)

        # Add noise to model (optionally)
        if (self.sigma > 0.):
            self.Y += self.sigma * np.random.randn(n_fit)

        pkg.log.close("%d points generated." % (n_fit))
