"""This submodule constructs and runs a Scikit-learn pipeline of the sort used
by PRISM, to facilitate testing of the Cuda-enabled estimator."""

import sys
import os
import importlib

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import Pipeline as Pipeline_sk
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

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

# This is to fix the following problem on OSX: https://github.com/scipy/scipy/issues/5998
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


class polynomial(metaclass=pkg.validation.metaclass):
    """This class provides the main testing capability of this module.

    It constructs a scikit-learn fitting pipeline of the sort used by
    PRISM.
    """

    # For each model & dataset, add a couple of validations for different fit polynomial orders
    for _dataset in pkg.mlxtend_cuda.fit_inputs.dataset.validation_grid:
        _n_fit = _dataset.kwargs_in['n_fit']
        _n_dim = np.shape(_dataset.kwargs_in['f_model'].kwargs_in['coeffs'])[0]
        for _poly_order in [5, 10]:
            if _n_dim**_poly_order < _n_fit:
                validation_grid.add(_dataset, poly_order=_poly_order)

    # Add validation members
    validation_members.add('poly_idx', rtol=1e-4)
    validation_members.add('X_poly', rtol=1e-4)
    validation_members.add('rsdl_var', rtol=1e-4)
    validation_members.add('regr_score', rtol=1e-4)
    validation_members.add('poly_powers', rtol=1e-4)
    validation_members.add('poly_coef', rtol=1e-4)
    validation_members.add('include_bias', atol=0)
    validation_members.add('intercept', rtol=1e-4)

    def __init__(self, dataset, poly_order=5, features='parsimonious'):
        """
        :param dataset: An instance of ``prism_adacs.mlxtend_cuda.dataset``
        :param poly_order: The maximum polynomial order to attempt a fit to
        :param features: The features option to pass to the Mlxtend k_features option.  Usually "best" or "parsimonious".
        """

        pkg.log.open("Generate order-%d polynomial fit to %d %d-D data points..." %
                     (poly_order, dataset.n_fit, dataset.f_model.n_dim))

        # Parse inputs
        self.X = np.asarray(dataset.X, dtype=np.float32)
        self.Y = np.asarray(dataset.Y)
        poly_order = np.int32(poly_order)

        pkg.log.open("Creating pipeline...")

        # Create SequentialFeatureSelector object
        estimator = pkg.mlxtend_cuda.estimator.cuLR()
        if (features == 'parsimonious'):
            sfs_obj = SFS(estimator,
                          k_features=features,
                          scoring='r2',
                          forward=False,
                          floating=False,
                          cv=min(poly_order, len(self.X)))
            self.include_bias = False
        elif (features == 'best'):
            sfs_obj = SFS(estimator,
                          k_features=features,
                          scoring='neg_mean_squared_error',
                          forward=True,
                          floating=False,
                          cv=min(poly_order, len(self.X)))
            self.include_bias = False

        else:
            sfs_obj = None
            pkg.log.error(Exception("Unsupported feature selection: %s" % (features)))

        # Create Scikit-learn Pipeline object
        self.pipe = Pipeline_sk([('poly', PF(poly_order, include_bias=self.include_bias)),
                                 ('SFS', sfs_obj),
                                 ('linear', LR())])
        pkg.log.close("Done.")

        pkg.log.open("Performing polynomial fit...")

        # Perform regression
        pkg.log.open("Perform regression...")
        self.pipe.fit(self.X, self.Y)
        pkg.log.close("Done.", time_elapsed=True)

        # Extract the selected features
        self.poly_idx = np.array(self.pipe.named_steps['SFS'].k_feature_idx_)

        # Create transformed space
        self.X_poly = self.pipe.named_steps['poly'].transform(self.X)[:, self.poly_idx]

        # Extract the rest of the fit results
        self.rsdl_var = mse(self.Y, self.pipe.named_steps['linear'].predict(self.X_poly))
        self.regr_score = self.pipe.named_steps['linear'].score(self.X_poly, self.Y)
        self.poly_powers = self.pipe.named_steps['poly'].powers_[self.poly_idx]
        self.poly_coef = self.pipe.named_steps['linear'].coef_
        self.feature_names = [self.pipe.named_steps['poly'].get_feature_names()[i] for i in self.poly_idx]

        if(self.include_bias):
            self.intercept = 0.
        else:
            self.intercept = self.pipe.named_steps['linear'].intercept_

        # Report fit results
        pkg.log.comment(self)

        pkg.log.close("Done.", time_elapsed=True)

        pkg.log.close("Done.", time_elapsed=True)

    def predict(self, x_plot):
        """Generate a prediction from the linear regression fit for a given
        input vector.

        :param x_plot: Input vector
        :return:  Output vector
        """

        # Map the given samples to the polynomial features
        poly_terms = self.pipe.named_steps['poly'].transform(x_plot)

        # Generate prediction
        y_plot = np.zeros(len(x_plot))
        for i_plot, (x_plot_i, poly_terms_i) in enumerate(zip(x_plot, poly_terms)):
            y_plot[i_plot] = self.intercept + np.sum(np.multiply(self.poly_coef, poly_terms_i[self.poly_idx]))

        return y_plot

    def plot(self, data, filename_plot):
        """Plot a dataset along-with the derived fit.

        :param data: dataset to plot
        :param filename_plot: output filename
        :return: None
        """

        if data.x_model is None:
            pkg.log.comment("No model ordinate system given.  Plot not generated.")
        else:

            def to_raster(X, y):
                """
                :param X: 2D image coordinates for values y
                :param y: vector of scalar or vector values
                :return: A, extent
                """

                def deduce_raster_params():
                    """Computes raster dimensions based on min/max coordinates
                    in X.

                    sample step computed from 2nd - smallest coordinate values
                    """
                    temp = [np.unique(v) for v in X.T]
                    unique_sorted = np.vstack(temp).T
                    d_min = unique_sorted[0]  # x min, y min
                    d_max = unique_sorted[-1]  # x max, y max
                    d_step = unique_sorted[1] - unique_sorted[0]  # x, y step
                    nsamples = (np.round((d_max - d_min) / d_step) + 1).astype(int)
                    return d_min, d_max, d_step, nsamples

                d_min, d_max, d_step, nsamples = deduce_raster_params()

                # Allocate matrix / tensor for raster. Allow y to be vector (e.g. RGB triplets)
                nsamples = np.append(nsamples, 1 if y.ndim == 1 else y.shape[-1])
                A = np.full(nsamples, np.NaN)
                A = np.squeeze(A)

                # Compute index for each point in X
                ind = np.round((X - d_min) / d_step).T.astype(int)

                # Prepare extent in imshow format
                extent = np.vstack((d_min, d_max)).T.ravel()

                # Scalar/vector values assigned over outer dimension
                for j, i in enumerate(np.transpose(ind)):
                    A[tuple(i)] = y[j]

                return A, extent

            pkg.log.open("Generate plot...")

            # We only support 1d and 2d plots for testing.  To much of a RAM overhead for anything more
            if(data.f_model.n_dim < 1 or data.f_model.n_dim > 2):
                pkg.log.close("plots can only be generated for n_dim=1 or 2.  Skipping.")
            elif(data.f_model.n_dim == 1):
                # Generate polynomial model so we can plot it
                pkg.log.open("Generate fit...")
                y_model = self.predict(data.x_model)
                pkg.log.close("Done.")

                colors = ['teal', 'yellowgreen', 'gold']
                lw = 2
                plt.plot(data.x_model,
                         np.asarray([data.f_model(x_i) for x_i in data.x_model]),
                         color='cornflowerblue',
                         linewidth=lw,
                         label="Ground truth")
                plt.scatter(self.X, self.Y, color='navy', s=30, marker='o', label="Training points")
                plt.plot(data.x_model, y_model, color=colors[0], linewidth=lw, label="Fit")
                plt.legend(shadow=True)
                plt.savefig(filename_plot)
                pkg.log.comment("%s written." % (filename_plot))
            elif(data.f_model.n_dim == 2):
                # Generate polynomial model so we can plot it
                pkg.log.open("Generate fit...")
                y_model = self.predict(data.x_model)
                pkg.log.close("Done.")

                # Plot ground truth
                y_truth = np.asarray([data.f_model(X_i) for X_i in data.x_model])
                A_truth, extent_truth = to_raster(data.x_model, y_truth)
                A_model, extent_model = to_raster(data.x_model, y_model)
                vmin = np.min(A_truth)
                vmax = np.max(A_truth)
                assert(np.array_equal(extent_model, extent_truth))
                extent = extent_truth

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                im_truth = axes[0].imshow(A_truth, extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
                axes[0].scatter(self.X[:, 0], self.X[:, 1], color='black', s=3, marker='o', label="Training points")
                axes[0].set_title("Ground truth")
                axes[0].set_xlim((extent[0], extent[1]))
                axes[0].set_ylim((extent[2], extent[3]))

                im_residual = axes[1].imshow(
                    A_truth - A_model,
                    extent=extent,
                    cmap='viridis',
                    vmin=vmin,
                    vmax=vmax,
                    origin='lower')
                axes[1].set_title("Residual")
                axes[1].set_xlim((extent[0], extent[1]))
                axes[1].set_ylim((extent[2], extent[3]))

                ax_cb = fig.add_axes([0.92, 0.14, 0.04, 0.72])
                plt.colorbar(im_residual, cax=ax_cb)

                # Generate output
                plt.savefig(filename_plot)
                pkg.log.comment("%s written." % (filename_plot))

            pkg.log.close("Done.")

    def __str__(self):
        """Return a string representation of the parameter set."""

        result = \
            "Fit results:\n" + \
            "    Residual variance: %s\n" % (str(self.rsdl_var)) + \
            "    Regression score:  %s\n" % (str(self.regr_score)) + \
            "    Terms in fit:      %s\n" % (str(self.feature_names).replace('\n', '')) + \
            "    Fit coefficients:  %s\n" % (str(self.poly_coef).replace('\n', ''))
        if(not self.include_bias):
            result += "    Intercept:         %s\n" % (str(self.intercept))
        return result
