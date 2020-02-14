import os
import sys
import importlib
import pkgutil

if sys.version_info >= (3, 3):
    from unittest.mock import MagicMock
else:
    from mock import MagicMock

# Make sure that what's in this path takes precedence
# over an installed version of the project
sys.path.insert(0, os.path.join(os.path.abspath(__file__), '..'))

# Infer the name of this package from the path of __file__
package_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
package_name = os.path.basename(package_root_dir)

# Read the package docstring.  We do things this way to separate package-specific
# content from implementation, to make it easier to update this file, for example
path_package_docstring = os.path.join(package_root_dir, "%s.docstring" % (package_name))
with open(path_package_docstring, "r") as fp_docstring:
    __doc__ = ''.join(fp_docstring.readlines())


# Make sure that what's in this path takes precedence
# over an installed version of the project
sys.path.insert(0, package_parent_dir)

# Import needed internal modules
validation = importlib.import_module(package_name + '._internal.validation')
_log = importlib.import_module(package_name + '._internal.log')

#: The library log stream (see the :py:mod:`._internal.log` module for more details)
log = _log.log_stream()

#: The absolute path to the module root path
_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# List the string-like classes
PY2 = sys.version_info[0] == 2
if not PY2:
    string_types = (str,)
else:
    string_types = (str, unicode)


def import_submodules(package=package_name, recursive=True):
    """Import all submodules of this module, recursively, including
    subpackages.

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """

    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if is_pkg:
            if recursive:
                results.update(import_submodules(full_name))
    return results


class _mock_module(MagicMock):
    """This class is used to generate mock modules in cases where we don't have
    access to the module-proper.

    This is particularly useful for RTD builds.
    """
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


def import_mock_RTD(package_name):
    """Import a package unless a Readthedocs environment is detected.  In that
    case, create a mock of the package. Useful for cases where a package is not
    available during a RTD build, but we want the build to proceed without
    error.

    :param package_name: The name of the package to import or mock.
    :return: The imported package or mock-package object.
    """
    if(not os.environ.get('READTHEDOCS') == 'True'):
        return importlib.import_module(package_name)
    else:
        log.comment("Using a mock for package {%s}." % (package_name))
        return _mock_module()


def full_path_datafile(path):
    """Return the full *INSTALLED* path to a file in the package's data
    directory.

    :param path: A path relative to the package's `/data` directory
    :return: The installed path
    """
    return os.path.join(_PACKAGE_ROOT, 'data', path)


def find_in_parent_path(path_start, filename_search, check=True, failure=None):
    """Find the path to a given filename, scanning up the directory tree from
    the given path_start.  Optionally throw an error (if check=True) if not
    found.

    :param path_start: The path from which to start the search.
    :param filename: The filename to search for.
    :param failure: Value to return on failure.
    :return: Path to the file if found, None (default) or `failure` if not found.
    """
    path_result = None
    if(os.path.isdir(path_start)):
        cur_dir = path_start
    else:
        cur_dir = os.path.dirname(path_start)

    # Scan upwards until we find the file or run out of path
    while(True):
        filename_test = os.path.join(cur_dir, filename_search)
        if(os.path.isfile(filename_test) or os.path.isdir(filename_test)):
            path_result = cur_dir
            break
        elif (cur_dir == os.sep):
            break
        else:
            cur_dir = os.path.dirname(cur_dir)

    # Check if the file has been found
    if(check and path_result is None):
        log.error(Exception("Could not find {%s} in parent directories of path {%s}." % (filename_search, path_start)))

    # On failure, set to failure default
    if(path_result is None):
        path_result = failure

    return path_result
