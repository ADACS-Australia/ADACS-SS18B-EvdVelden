"""This module provides a `package` class for polling the meta data describing
a Python package."""
import os
import sys
import importlib
import json

# Infer the name of this package from the path of __file__
package_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
package_name = os.path.basename(package_root_dir)

# Make sure that what's in this path takes precedence
# over an installed version of the project
sys.path.insert(0, package_parent_dir)

# Import needed internal modules
_internal = importlib.import_module(package_name + '._internal')
pkg = importlib.import_module(package_name)


class package:
    """This class yields an object which exposes the parameters describing a
    Python package which is part of this project."""

    def __init__(self, path_call, verbosity=True):
        """
        :param path_call: this needs to be the FULL (i.e. absolute) path to a file or directory living somewhere in the package
        :param verbosity: Optionally, set the log stream verbosity for this function (defaults to True)
        """

        # Set verbosity of log for this function call
        pkg.log.set_verbosity(verbosity=verbosity)

        # Scan upwards from the given path until 'setup.py' is found.  That will be the package parent directory.
        self.path_package_parent = pkg.find_in_parent_path(path_call, ".package.json")

        # Assume that the tail of the root path is the package name
        self.package_name = os.path.basename(self.path_package_parent)

        # Set the path where all the package modules start
        self.path_package_root = os.path.join(self.path_package_parent, self.package_name)

        # Read the package file
        with open_package_file(self.path_package_parent) as file_in:
            self.params = file_in.load()

        # Assemble a list of data files to bundle with the package
        self.package_files = self.collect_package_files()

        # Assemble a list of package scripts
        self.scripts = self.collect_package_scripts()

        # Return the stream verbosity to its previous state
        pkg.log.unset_verbosity()

    def collect_package_files(self):
        """Generate a list of non-code files to be included in the package.

        By default, all files in the 'data' directory in the package root will be added.

        :return: a list of absolute paths.
        """
        paths = []
        # Add the .project.json and .package.json files.  There are instances where these
        # don't get installed by default (tox virtual envs, for example) and we need
        # to make sure they are present
        paths.append(os.path.abspath(os.path.join(self.path_package_parent, ".project.json")))
        paths.append(os.path.abspath(os.path.join(self.path_package_parent, ".project_aux.json")))
        paths.append(os.path.abspath(os.path.join(self.path_package_parent, ".package.json")))

        # Add the data directory
        for (path, directories, filenames) in os.walk(os.path.join(self.path_package_parent, "data"), followlinks=True):
            if(path != "__pycache__"):
                for filename in filenames:
                    paths.append(os.path.join('..', path, filename))

        # Add any .docstring files
        for (path, directories, filenames) in os.walk(self.path_package_root, followlinks=True):
            for filename in filenames:
                if(filename.endswith('.docstring')):
                    paths.append(os.path.join('..', path, filename))

        # setup() struggles when these filenames have unicode under python 2.7, so strip that here
        return [str(path) for path in paths]

    def collect_package_scripts(self):
        """Generate a list of script files associated with this package.

        By default, all files in the 'scripts' directory in the package root will be added.

        :return: a list of absolute paths.
        """
        paths = []

        # Add the scripts directory
        path_start = os.path.join(self.path_package_root, "scripts")
        for (path, directories, filenames) in os.walk(path_start, followlinks=True):
            for filename in filenames:
                filename_base = os.path.basename(filename)
                script_name, filename_extension = os.path.splitext(filename_base)
                if(filename_extension == '.py'):
                    if(script_name != '__init__'):
                        path_relative = os.path.relpath(path, path_start)
                        script_pkg_path = path_relative.replace("/", ".")
                        if(script_pkg_path == '.'):
                            script_pkg_path = ''
                        else:
                            script_pkg_path += '.'
                        script_pkg_path += script_name
                        paths.append([script_name, script_pkg_path])
        return paths

    def __str__(self):
        """Convert dictionary of package parameters to a string.

        :return: string
        """
        result = "Package information:\n"
        result += "--------------------\n"
        for k, v in sorted(self.params.items()):
            result += '   ' + k + " = " + str(v) + '\n'

        return result


class package_file():
    """Class for reading and writing package .json files.

    Intended to be used with the `open_package_file` context manager.
    """

    def __init__(self, path_package_parent):
        """
        :param path_package_parent: The path to the directory hosting the package's `setup.py` file.
        """
        # File pointer
        self.fp = None

        # Assume this filename for the package file
        self.filename_package_filename = '.package.json'

        # Set the filename of the package copy of the package file
        self.filename_package_file = os.path.join(path_package_parent, self.filename_package_filename)

    def open(self):
        """Open the package .json file.  Intended to be accessed through the
        `open_package_file` class using a `with` block.

        :return: None
        """
        try:
            self.fp = open(self.filename_package_file)
        except BaseException:
            pkg.log.error(Exception("Could not open package file {%s}." % (self.filename)))
            raise

    def close(self):
        """Close the package .json file.

        :return: None
        """
        try:
            self.fp.close()
        except BaseException:
            pkg.log.error(Exception("Could not close package file {%s}." % (self.filename)))
            raise

    def load(self):
        """Load an opened project .json file.

        :return: None
        """
        try:
            params_list = json.load(self.fp, object_hook=_internal.ascii_encode_dict)
        except BaseException:
            pkg.log.error(Exception("Could not load package file {%s}." % (self.filename)))
            raise
        finally:
            return {k: v for d in params_list for k, v in d.items()}


class open_package_file:
    """Context manager for reading a package .json files.

    Intended for use with a `with` block.
    """

    def __init__(self, path_call):
        """
        :param path_call: Context expression
        """
        self.path_call = path_call

    def __enter__(self):
        """Open the project .json file when entering the context.

        :return: file pointer
        """
        # Open the package's copy of the file
        pkg.log.open("Opening package...")
        try:
            self.file_in = package_file(self.path_call)
            self.file_in.open()
        except BaseException:
            pkg.log.error(Exception("Could not open package file."))
            raise
        finally:
            pkg.log.close("Done.")
            return self.file_in

    def __exit__(self, *exc):
        """Close the package .json file when exiting the context.

        :param exc: Context expression arguments.
        :return: False
        """
        self.file_in.close()
        return False
