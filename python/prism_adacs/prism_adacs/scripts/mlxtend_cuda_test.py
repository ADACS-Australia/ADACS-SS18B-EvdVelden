from __future__ import print_function
import sys
import os
import importlib
import click
import numpy as np
import timeit
import ast
import sklearn.metrics.regression as reg
import sklearn.model_selection._validation as skv
import mlxtend.feature_selection.sequential_feature_selector as SFS


# Infer the name of this package from the path of __file__
package_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
package_name = os.path.basename(package_root_dir)

# Make sure that what's in this path takes precedence
# over an installed version of the project
sys.path.insert(0, package_parent_dir)

# Import needed internal modules
pkg = importlib.import_module(package_name)
prj = importlib.import_module(package_name + '._internal.project')

# Import package submodules
pkg.import_submodules()

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def callback_literal_eval(ctx, param, value):
    return np.asarray(ast.literal_eval(value))


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--func_name', default='xsinx', show_default=True, help='Model ("xsinx" or "gauss")')
@click.option('--coeffs', is_flag=False, default="((1,1),(1,1))",
              show_default=True, metavar='<array>',
              required=True, type=click.STRING, help='Set model parameters',
              callback=callback_literal_eval)
@click.option('--x_range', type=(float, float), default=(0, 10), help='Range to be covered for all axes.')
@click.option('--n_plot', type=int, default=100, show_default=True, help='Model parameter: alpha0')
@click.option('--n_fit', type=int, default=20, show_default=True, help='Model parameter: alpha0')
@click.option('--sigma', type=float, default=0.5, show_default=True, help='Dispersion to add to model for data')
@click.option('--poly_order', type=int, default=10, show_default=True, help='Model parameter: alpha0')
@click.option('--write2stdout/--no-write2stdout', default=True, show_default=True, help='Write to standard out?')
@click.option('--write2bin/--no-write2bin', default=False, show_default=True, help='Write to binary files?')
@click.option('--check/--no-check', default=False, show_default=True,
              help='Check result against stored results for standard parameter sets (if a match is detected).')
@click.option('--filename_input', default=None, show_default=True, help='Filename to read inputs from.')
@click.option('--filename_plot', default=None, show_default=True, help='Filename to write plot to.')
@click.option(
    '--timing',
    type=(
        float,
        float,
        int,
        int),
    default=(
        None,
        None,
        None,
        None),
    help='Run in timing mode (all previous parameters are ignored).  Specify run as POLY_ORDER_LO POLY_ORDER_HI POLY_ORDER_STEP N_AVG.')
def mlxtend_cuda_test(
        func_name,
        coeffs,
        x_range,
        n_plot,
        n_fit,
        sigma,
        poly_order,
        write2stdout,
        write2bin,
        check,
        filename_input,
        filename_plot,
        timing):
    """This executable runs test fits of the polynomial fitting pipeline used
    by PRISM.

    Gaussian and xsinx models can be used to generate and then
    subsequently fit a model, for various numbers of data points, etc.
    Cuda support can be switched-off by setting the environment variable
    'ENABLE_CUDA' to something false-like (0, 'OFF', etc).
    """

    pkg.log.open("Running mlxtend_cuda...")

    # Write the CLI arguments to the log stream
    pkg.log.open("CLI arguments:")
    ctx = click.get_current_context()
    params = sorted(ctx.params)
    max_param_len = max([len(param) for param in ctx.params])
    for param in params:
        val = str(locals()[param]).replace('\n', '')
        pkg.log.comment(("%s: " + (max_param_len - len(param)) * ' ' + val) % (param))
    pkg.log.close()

    # Parse the 'timing' option.  If it is given,
    # then assume that it specifies a range of frequencies
    # to test, the number of frequencies to test, and the
    # number of calls to average results over
    if(timing[0] is not None):
        flag_timing = True
        poly_order_lo, poly_order_hi, poly_order_step, n_avg = timing
    # ... if it isn't given, just perform one run
    else:
        flag_timing = False
        poly_order_lo = 2
        poly_order_hi = 10
        poly_order_step = 1
        n_avg = 0

    # Generate timing tests
    if(flag_timing):

        # Generate the list of n_freq's that we are going to time
        poly_order_list = np.arange(poly_order_lo, poly_order_hi, poly_order_step)

        # Generate timing results for each n_freq
        n_burn = 1
        pkg.log.set_verbosity(False)
        for i_poly_order, poly_order_i in enumerate(poly_order_list):

            # Initialize the model call (apply some unit conversions here)
            f_model = pkg.mlxtend_cuda.fit_inputs.model(name=func_name, coeffs=coeffs)
            inputs = pkg.mlxtend_cuda.fit_inputs.dataset(
                f_model=f_model, x_range=x_range, n_plot=n_plot, n_fit=n_fit, sigma=sigma)

            # Create a timing callable
            t = timeit.Timer(lambda: pkg.mlxtend_cuda.fit.polynomial(inputs=inputs, poly_order=poly_order_i))

            # Burn a number of calls (to avoid contamination from Cuda context initialization if buf=None, for example)
            if(n_burn > 0):
                if(n_burn == 1):
                    pkg.log.comment("Burning a call: %f seconds." % (t.timeit(number=n_burn)))
                else:
                    pkg.log.comment("Burning %d calls: %f seconds." % (n_burn, t.timeit(number=n_burn)))
                n_burn = 0

            # Call the model n_avg times to generate the timing result
            wallclock_i = t.timeit(number=n_avg)

            # Print timing result
            if(len(poly_order_list) == 1):
                pkg.log.comment("Average timing of %d calls: %.5f seconds." % (n_avg, wallclock_i / float(n_avg)))
            else:
                if(i_poly_order == 0):
                    print("# model: %s" % (inputs.f_model))
                    print("# Column 01: Iteration")
                    print("#        02: No. polynomial terms")
                    print("#        03: Total time for %d calls [s]" % (n_avg))
                    print("#        04: Avg. time per call [s]")
                print("%3d %8d %10.3le %10.3le" % (i_poly_order, poly_order_i, wallclock_i, wallclock_i / float(n_avg)))
        pkg.log.unset_verbosity()

    # ... if n_avg<=1, then just run the model and exit.
    else:

        # Initialize model call
        if filename_input:
            inputs = pkg.mlxtend_cuda.fit_inputs.dataset.load(filename_input)
        else:
            f_model = pkg.mlxtend_cuda.fit_inputs.model(name=func_name, coeffs=coeffs)
            inputs = pkg.mlxtend_cuda.fit_inputs.dataset(
                f_model=f_model, x_range=x_range, n_plot=n_plot, n_fit=n_fit, sigma=sigma)

        # Perform call
        output = pkg.mlxtend_cuda.fit.polynomial(inputs, poly_order=poly_order)

        # Generate plot
        if filename_plot:
            output.plot(inputs, filename_plot)

        # Write results to stdout &/or binary files
        if(write2stdout):
            pkg.log.comment(output)
        if(write2bin):
            inputs.save()
            output.save()

        # Check results against standards (if parameters match)
        if(check):
            pkg.test.calc_difference_from_reference(inputs, results)

    pkg.mlxtend_cuda.estimator.timer.print()
    pkg.log.close("Done.")
    pkg.log._unhang()


# Permit script execution
if __name__ == '__main__':
    status = mlxtend_cuda_test()
    sys.exit(status)
