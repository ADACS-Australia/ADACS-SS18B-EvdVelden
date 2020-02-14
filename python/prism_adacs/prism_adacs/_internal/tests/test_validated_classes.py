import sys
import os
import glob
import importlib
import pytest

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

# # Import package submodules
pkg.import_submodules()


@pkg.log.test()
def build_validation_class_test_grid():
    """Build the grid of validations to run.

    :return: list
    """
    pkg.log.open("Building validation class test grid...")
    metaclass_validation_grid = []
    for class_i in pkg.validation.metaclass.list:
        pkg.log.open("Processing validated class: " + class_i.__name__ + '...')
        if class_i.validation_members:
            subclass_name = class_i.__name__
            filename_list = sorted(glob.glob(pkg.full_path_datafile("validation_tests/" + subclass_name + ".P*")))
            for filename_base in filename_list:
                metaclass_validation_grid.append([class_i, filename_base])
            pkg.log.close(str(len(filename_list)) + " found...Done.")
        else:
            pkg.log.close("skipped. (no validation members)")

    pkg.log.close("Done.")
    return metaclass_validation_grid

# *** Initialise individual test ***


@pytest.fixture(params=build_validation_class_test_grid())
@pkg.log.test()
def init_validation_test(request):
    """Load the validation file to act as reference for a specific validation.

    :param request: element from the list returned from build_validation_class_test_grid()
    :return: class instance to act as reference for the test
    """

    # Read reference inputs & results
    validation_subclass = request.param[0]
    filename_base = request.param[1]

    pkg.log.open("Load files for validation fixture {class=%s}..." % (validation_subclass.__name__))
    reference = pkg.validation.file.load(filename_base=filename_base)
    pkg.log.close("Done.")

    return [reference]

# *** Perform individual test ***


@pkg.log.test()
def test_validation(init_validation_test):
    """Construct a class instance and perform comparison to its validation
    reference.

    :param init_validation_test: reference returned by init_validation_test()
    :return: None
    """

    # Parse fixture
    reference = init_validation_test[0]
    validated_class = type(reference)

    # Report the fixture
    pkg.log.open("Testing fixture {class=%s}..." % (validated_class.__name__))
    pkg.log.comment(reference)

    # Build test instance from reference inputs
    test = validated_class(*reference._validation_args, **reference._validation_kwargs)

    # Calculate difference between the validation members of the test and reference instances
    difference = test._validation_difference(reference)

    # assert no difference, else print message
    assert difference.none, "validation failure for class=%s:\n%s" % (validated_class.__name__, difference)

    pkg.log.close("Done.")
