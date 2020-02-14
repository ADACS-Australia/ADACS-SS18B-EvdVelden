import os
import sys
import importlib
import click
import glob

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
docs = importlib.import_module(package_name + '._internal.docs')

# Import package submodules
pkg.import_submodules()

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def prism_adacs_helper(ctx):
    """This executable provides miscellaneous support for this project.

    It can be used to report project parameters (versions, etc.),
    updated API documentation, generate validation testing files,
    perform validation timing tests, etc.
    """
    pass


@prism_adacs_helper.command(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def info(ctx):
    """Print project and package parameters."""

    # Set/fetch all the project details we need
    project = prj.project(__file__, verbosity=False)

    # Print project information
    print(project)

    # Print package information
    for package in project.packages:
        print(package)


@prism_adacs_helper.command(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def update_docs(ctx):
    """Auto-generate the .rst files which describe all the project APIs (both
    C/C++ and Python, if present).

    This function is meant to be called automatically by the "docs-
    update" target of the project makefile.
    """
    # Set/fetch all the project details we need
    project = prj.project(__file__)

    # Generate the main project .rst index file
    # and any needed API files as well
    docs.generate_project_rsts(project)


@prism_adacs_helper.group(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def validate(ctx):
    """Perform validation operations."""
    pass


@validate.command(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def init(ctx):
    """Generate validation files."""

    pkg.log.open("Building validation files...")
    for class_i in pkg.validation.metaclass.list:
        class_name = class_i.__name__
        pkg.log.open("Processing validated class: " + class_name + '...')

        # First, clear old files
        filename_list = glob.glob(pkg.full_path_datafile("validation_tests/" + class_name + ".dat*"))
        if filename_list:
            for filename in filename_list:
                os.remove(filename)
            pkg.log.comment(str(len(filename_list)) + " old files removed...")

        # Build instances and write them
        class_i._build_validation_files()
        pkg.log.close("Done.")

    pkg.log.close("Done.")


@validate.command(context_settings=CONTEXT_SETTINGS)
@click.option('--n_avg', type=int, default=1, show_default=True, help='Number of timing runs for averaging purposes')
@click.option('--n_burn', type=int, default=0, show_default=True, help='Number of calls to discard before averaging')
@click.pass_context
def timing(ctx, n_avg, n_burn):
    """Perform validation timing test."""

    # Generate timings
    timing = pkg.validation.timing(n_burn=n_burn, n_avg=n_avg)

    # Print results
    timing.write()


# Permit script execution
if __name__ == '__main__':
    status = prism_adacs_helper()
    sys.exit(status)
