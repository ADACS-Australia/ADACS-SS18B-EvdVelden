"""This submodule provides a set of classes to enable quick-and-easy set-up of
class regression tests.  The following steps need to be followed:

1. When defining a class, use the metaclass provided by this submodule
2. Optionally call the ``validation_grid.add()`` method one-or-more times to add a *set of inputs* to validate against
3. Optionally call the ``validation_member.add()`` method one-or-more times to add a *class member* to validate against

You should end-up with a class definition that looks something like this::

   class MyValidatedClass(metaclass=pkg.validation.metaclass):

      # Run a test against each of these sets of inputs
      for X_o,Y_o in [[0,0],[0,1],[1,0],[1,1]]:
         validation_grid.add(X_o=X_o,Y_o=Y_o)

      # Check the results of these members, to the tolerances given
      validation_members.add('X',atol=0.1,rtol=1e-4)
      validation_members.add('Y',atol=0.1,rtol=1e-4)

      def __init__(self):
         ...

To then run the tests, validation files need to be constructed and then the tests need to be run.  To generate the
validation files, use the helper executable provided to run the following::

   prism_adacs_helper validate init

and then run the tests as follows::

   make tests
"""
import sys
import os
import importlib
import pickle
import math
import numpy as np
import difflib
import timeit

# Infer the name of this package from the path of __file__
package_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
package_name = os.path.basename(package_root_dir)

# Make sure that what's in this path takes precedence
# over an installed version of the project
sys.path.insert(0, package_parent_dir)

# Import internal modules
pkg = importlib.import_module(package_name)


class timing(object):
    """This class provides support for timing of validation tests."""

    def __init__(self, n_burn=0, n_avg=1):
        """
        :param n_burn: Number of calls to burn before starting averaging runs
        :param n_avg: Number of averaging runs
        """
        self.n_burn = n_burn
        self.n_avg = n_avg
        self.results = []

        pkg.log.open("Generating validation timings...",)
        for class_i in metaclass.list:
            for i_grid, grid_i in enumerate(class_i.validation_grid):

                # Create a timing callable
                t = timeit.Timer(lambda: grid_i())

                # Burn a number of calls (to avoid contamination from Cuda context initialization, for example)
                if(n_burn > 0):
                    t_burn = t.timeit(number=n_burn)
                else:
                    t_burn = 0.

                # Call the model n_avg times to generate the timing result
                t_avg = t.timeit(number=n_avg)

                # Compure averages
                if n_avg > 0:
                    dt_avg = t_avg / n_avg
                else:
                    dt_avg = 0
                if n_burn > 0:
                    dt_burn = t_burn / n_burn
                else:
                    dt_burn = 0

                self.results.append({'t_burn': t_burn, 't_avg': t_avg, 'n_burn': n_burn,
                                     'n_avg': n_avg, 'dt_burn': dt_burn, 'dt_avg': dt_avg})
        pkg.log.unset_verbosity()
        pkg.log.close("Done.")

    def write(self, fp=sys.stdout):
        """Write timing results to a file.

        :param fp: Optional file pointer (defaults to sys.stdout)
        :return: None
        :raises: IOError
        """
        try:
            fp.write("# Column(1): t_avg\n")
            fp.write("# Column(2): n_avg\n")
            fp.write("# Column(3): dt_avg\n")
            fp.write("# Column(4): t_burn\n")
            fp.write("# Column(5): n_burn\n")
            fp.write("# Column(6): dt_burn\n")
            for result in self.results:
                fp.write(
                    "%10.3f %3d %10.3f %10.3f %3d %10.3f\n" %
                    (result['t_avg'],
                     result['n_avg'],
                        result['dt_avg'],
                        result['t_burn'],
                        result['n_burn'],
                        result['dt_burn']))
        except BaseException:
            raise IOError("Could not write timing results.")


class file(object):
    """Class for reading and writing validation files."""

    # A colloquial filename type
    file_type_name = 'validation'

    def __init__(self, path, mode=None):
        """
        :param path_package_parent: The path to the directory hosting the package's `setup.py` file.
        """

        # File pointer
        self.fp = None

        # Save path & mode
        self.path = path
        self.mode = mode

    def open(self, mode):
        """Open a validation file.

        :param mode: read/write mode for the opened file (i.e. 'r' or 'w')
        :return: None
        """
        try:
            if self.fp:
                self.fp.close()
            self.mode = mode
            self.fp = open(self.path, self.mode)
        except Exception as error:
            pkg.log.error(error)

    def close(self):
        """Close a validation file.

        :return: None
        """
        try:
            if self.fp:
                self.fp.close()
            self.fp = None
        except BaseException:
            pkg.log.error(Exception("Could not close %s file {%s}." % (self.file_type_name, self.path)))
            raise

    @classmethod
    def filename(Cls, filename_base=None, filename_label=None):
        """Create a filename for a validation file.

        :param filename_base: full path to file, modulo optional label (see below).  Defaults to class name if None.
        :param filename_label: an optional label to add as an extension at the end of the filename.
        :return: string
        """

        if not filename_base:
            filename_base = Cls.__name__
        if (filename_label):
            return filename_base + '.' + filename_label
        else:
            return filename_base

    @classmethod
    def save(Cls, instance, filename_base=None, filename_label=None):
        """Save an opened file.

        :param instance: Object to write to file
        :param filename_base: full path to file, modulo optional label (see below).  Defaults to class name if None.
        :param filename_label: an optional label to add as an extension at the end of the filename.
        :return:
        """

        # Compose the filename
        if filename_base:
            filename = Cls.filename(filename_base=filename_base, filename_label=filename_label)
        else:
            filename = Cls.filename(filename_base=pkg.full_path_datafile(
                "validation_tests/" + type(instance).__name__ + ".P"), filename_label=filename_label)

        pkg.log.open("Saving validation file {%s}..." % (filename))
        try:
            with Cls(filename, "wb") as file_out:
                pickle.dump(instance, file_out.fp)
        except BaseException:
            raise
        pkg.log.close("Done.")

    @classmethod
    def load(Cls, filename_base=None, filename_label=None):
        """Load a validation file and return the instance it has stored.

        :param filename_base: full path to file, modulo optional label (see below).  Defaults to class name if None.
        :param filename_label: an optional label to add as an extension at the end of the filename.
        :return:
        """

        # Compose the filename
        if filename_base:
            filename = Cls.filename(filename_base=filename_base, filename_label=filename_label)
        else:
            filename = Cls.filename(filename_base=pkg.full_path_datafile(
                "validation_tests/" + type(instance).__name__ + ".P"), filename_label=filename_label)

        pkg.log.open("Instantiating from validation file '%s'..." % (filename))
        try:
            with Cls(filename, "rb") as file_in:
                instance = pickle.load(file_in.fp)
        except BaseException:
            raise
        pkg.log.close("Done.")

        return instance

    def __enter__(self):
        """Open the file when entering a context.

        :return: file pointer
        """
        pkg.log.open("Opening %s file..." % (self.file_type_name))
        try:
            self.open(self.mode)
        except BaseException:
            pkg.log.error(Exception("Could not open %s file." % (self.file_type_name)))
            raise
        finally:
            pkg.log.close("Done.")
            return self

    def __exit__(self, *exc):
        """Close the file when exiting a context.

        :param exc: Context expression arguments.
        :return: False
        """
        self.close()
        return False


class difference(object):
    """This class provides the ability to compute and evaluate differences in
    the validation members of two validation class instances."""

    def __init__(self, instance, reference):
        """
        :param instance: An instance of a validation class
        :param reference: An associated reference instance to compare to
        :return: None
        """

        self.none = False
        self.differences = dict()
        self.validated = False
        self.instance = instance
        self.reference = reference

        def calc_diff(x, y):
            """Calculate the absolute and fractional difference between two
            numbers.

            If the reference value is 0 and the value to check is not, returns np.inf for fractional difference.

            :param x: Value to check
            :param y: Reference value
            :return: Value
            """
            if (y != 0):
                adiff = math.fabs(x - y)
                return adiff, (adiff / y)
            elif (x != 0.):
                return math.fabs(x), np.inf
            else:
                return 0., 0.

        # TODO: Ensure that the constructor inputs are the same here
        pass

        # Compute statistics of difference from test reference
        for member in instance.validation_members:
            name = member['name']
            atol = member['atol']
            rtol = member['rtol']
            try:
                obj_instance = instance.__getattribute__(name)
                obj_reference = reference.__getattribute__(name)
            except BaseException:
                raise
            else:

                # Make sure the types are the same
                type_instance = type(obj_instance)
                type_reference = type(obj_reference)
                if type_instance != type_reference:
                    raise TypeError(
                        "Member '%s' differs in type (%s != %s)" % (name, str(type_instance), str(type_reference)))

                # Make sure the constructor parameters are the same
                if reference._validation_args != instance._validation_args:
                    print("reference args:", reference._validation_args)
                    print("instance args:", reference._validation_args)
                    pkg.log.error(Exception("reference and instance args don't match in validation differencing."))

                if reference._validation_kwargs != instance._validation_kwargs:
                    print("reference kwargs:", reference._validation_kwargs)
                    print("instance kwargs:", reference._validation_kwargs)
                    pkg.log.error(Exception("reference and instance kwargs don't match in validation differencing."))

                # Compute the appropriate difference, depending on the type
                if isinstance(obj_instance, pkg._internal.string_types):
                    if not (obj_instance == obj_reference):
                        self.differences[name] = str(difflib.ndiff(obj_instance, obj_reference).join())
                elif isinstance(obj_instance, (np.ndarray, list, tuple)):
                    obj_instance = np.asarray(obj_instance)
                    obj_instance = np.asarray(obj_instance)
                    if np.shape(obj_instance) != np.shape(obj_reference):
                        self.differences[name] = "shapes differ (%s != %s)" % (
                            str(np.shape(obj_instance)), str(np.shape(obj_reference)))
                    else:
                        if not np.allclose(obj_instance, obj_reference, rtol=rtol, atol=atol, equal_nan=False):
                            n_diff = 0
                            rdiff_avg = 0.
                            rdiff_max = 0.
                            for output_i, ref_i in zip(obj_instance.flatten(), obj_reference.flatten()):
                                adiff_i, rdiff_i = calc_diff(output_i, ref_i)
                                rdiff_avg += rdiff_i
                                rdiff_max = max([rdiff_max, rdiff_i])
                                if(adiff_i >= atol or rdiff_i > rtol):
                                    n_diff += 1
                            rdiff_avg /= float(len(obj_instance))
                            self.differences[name] = "avg. rel. diff, max. rel. diff, n_diff = %le, %le, %d" % (
                                rdiff_avg, rdiff_max, n_diff)
                else:
                    try:
                        check = (obj_instance == obj_reference)
                        if not check:
                            adiff, rdiff = calc_diff(obj_instance, obj_reference)
                            self.differences[name] = 'abs. diff, rel. diff = %le, %le' % (adiff, rdiff)
                    except BaseException:
                        raise TypeError("Could not calculate a difference for member '%s'" % (name))

        self.none = (len(self.differences) == 0)
        self.validated = True

    def validate(self):
        """Return the true/false state of the difference check (true if
        instances are the same; false otherwise)

        :return: Boolean or RuntimeError exception
        """
        if self.validated:
            return self.none
        else:
            raise RuntimeError("Invalid validation difference.")

    def __repr__(self):
        return "%s(%s,%s)" % (type(self).__name__, type(self.instance).__name__, type(self.reference).__name__)

    def __str__(self):
        if self.differences:
            max_name_length = len(max(self.differences.keys(), key=len))
            result = ''
            for name in self.differences.keys():
                lines = self.differences[name].split('\n')
                for line in lines:
                    result += "%s: %s\n" % (name.ljust(max_name_length), line)
        else:
            result = 'None'
        return result


class grid_element(object):
    """This class manages sets of inputs to be used for a specific validation
    class test."""

    def __init__(self, element_type_in, *args_in, **kwargs_in):
        """
        :param element_type_in: instance type or string-name of element
        :param args_in: constructor args of for element
        :param kwargs_in: constructor kwargs of for element
        """

        self.element_type = element_type_in
        self.args_in = args_in
        self.kwargs_in = kwargs_in

    def render_args(self):
        """Instantiate any grid elements that have been passed as args to a
        grid element instance.

        :return: tuple
        """

        args = list(self.args_in)
        for i_arg, arg in enumerate(args):
            if isinstance(arg, grid_element):
                args[i_arg] = arg()
        return tuple(args)

    def render_kwargs(self):
        """Instantiate any grid elements that have been passed as kwargs to a
        grid element instance.

        :return: dict
        """

        kwargs = self.kwargs_in.copy()
        for key in kwargs.keys():
            if isinstance(kwargs[key], grid_element):
                kwargs[key] = kwargs[key]()
        return kwargs

    def __call__(self):
        """Instantiate the grid element.

        :return: 'element_type' instance
        """
        if isinstance(self.element_type, (str)):
            try:
                self.element_type = globals()[self.element_type]
            except BaseException:
                pkg.log.error(Exception("Could not interpret the type of grid element: %s" % (str(self))))
        args = self.render_args()
        kwargs = self.render_kwargs()
        return self.element_type(*args, **kwargs)

    def __repr__(self):
        return("<Grid element>")

    def __str__(self):
        return("<Grid element: %s %s>" % (self.args_in, self.kwargs_in))


class grid(object):
    """This class manages lists of grid elements."""

    def __init__(self, fixed_type=None):
        """
        :param fixed_type: an instance type to be used for all elements in the grid
        """
        self._fixed_type = fixed_type
        self.list = []

    def __iter__(self):
        """Expose a grid as an iterable.

        :return: grid element
        """
        return iter(self.list)

    def add(self, *args, **kwargs):
        """Add a new instance to the grid.

        :param args: instance constructor args
        :param kwargs: instance constructor kwargs
        :return: None
        """

        # Fix args w/ fixed type
        if self._fixed_type:
            self.list.append(grid_element(self._fixed_type, *args, **kwargs))
        else:
            # The first element of args is the element type in this case
            args_list = list(args)
            element_type = args_list.pop(0)
            args = tuple(args_list)
            self.list.append(grid_element(element_type, *args, **kwargs))


class member_list(object):
    """Manage a list of validation members."""

    def __init__(self):
        self.list = []

    def __iter__(self):
        """Expose a member_list as an iterable.

        :return: member
        """
        return iter(self.list)

    def add(self, name, rtol=0., atol=0.):
        """Add a validation member.

        :param name: name of validation member
        :param rtol: relative tolerance for validation
        :param atol: absolute tolerance for validation
        :return: None
        """
        self.list.append({'name': name, 'rtol': rtol, 'atol': atol})


class metaclass(type):
    """This metaclass drives all the functionality provided in this subpackage.

    Any class that is to be validated needs to use this metaclass for
    its construction.
    """

    #: A list of all classes defined using this metaclass
    list = []

    def __prepare__(name, *args, **kwargs):
        """Initialise the dictionary that gets passed to __new___.

        This is needed here because we don't want the user to have to
        initialise the members that we are adding.  This is the only
        method that gets sourced before the class code is executed, so
        it needs to be done here, not in __new___.
        """
        result = dict()

        # Initialise the following members
        result['validation_grid'] = grid(fixed_type=name)
        result['validation_members'] = member_list()

        return result

    def __new__(mcs, name, bases, dct):

        def _validation_difference(self, other):
            """Compute the difference between two instances."""
            return difference(self, other)

        def _build_validation_files(cls):
            """Build all validation files (for the grid specified in class
            member 'validation_grid') for this class."""
            for i_grid, grid_i in enumerate(cls.validation_grid):
                pkg.validation.file.save(grid_i(), filename_label=str(i_grid))
            pkg.log.append(str(len(cls.validation_grid.list)) + " files saved...")

        # Perform super-metaclass construction
        new_class = super(metaclass, mcs).__new__(mcs, name, bases, dct)

        # Add the following methods
        setattr(new_class, _validation_difference.__name__, _validation_difference)
        setattr(new_class, _build_validation_files.__name__, classmethod(_build_validation_files))

        # Perform some clean-up and validation of the validation grid
        new_class.validation_grid.fixed_type = new_class
        for element_i in new_class.validation_grid:
            if isinstance(element_i.element_type, (str)):
                if not element_i.element_type == name:
                    pkg.log.error(
                        Exception(
                            "Type names don't match somehow in validation metaclass (%s!=%s)." %
                            (element_i.element_type, name)))
                element_i.element_type = new_class
            if not element_i.element_type == new_class:
                pkg.log.error(Exception("Types don't match somehow in validation metaclass (%s)." % (name)))

        # Rewrite __init__ so that *args and **kwargs are saved
        old_init = new_class.__init__

        def __init__(self, *args, **kwargs):
            old_init(self, *args, **kwargs)
            self._validation_args = args
            self._validation_kwargs = kwargs
        setattr(new_class, '__init__', __init__)

        # Add new class instance to the metaclass instance list
        mcs.list.append(new_class)

        return new_class

    def __str__(self):
        result = str()
        # for item in self.validation_members:
        #     result += "validation member: %s rtol=%f atol=%f\n" % (
        #         item['name'], item['rtol'], item['atol'])
        result += "<Validation class {%s}>" % (self.__name__)
        return result

    def __repr__(self):
        result = str()
        # for item in self.validation_members:
        #     result += "validation member: %s rtol=%f atol=%f\n" % (
        #         item['name'], item['rtol'], item['atol'])
        result += "<Validation class {%s}>" % (self.__name__)
        return result
