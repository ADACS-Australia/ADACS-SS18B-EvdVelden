"""This module is generally only of use to developers of this codebase.  It
provides several submodules for:

1. ``_internal.docs``:       generating API documentation files
2. ``_internal.log``:        generating course logging information for the user
3. ``_internal.package``:    examining meta data relating to this python package
4. ``_internal.project``:    examining meta data relating to the project this package belongs to
5. ``_internal.validation``: easy configuration of tests (primarily, regression tests) on package classes
"""

import sys

PY2 = sys.version_info[0] == 2

if PY2:
    raise Exception("Package not compatible with Python 2.")

if not PY2:
    string_types = (str,)
else:
    string_types = (str, unicode)


def is_nonstring_iterable(object_in):
    """Determine if an object is a non-string iterable.

    :param object: An object of any type.
    :return: Boolean.  True if object is a non-string iterable.
    """
    return hasattr(object_in, '__iter__') and not isinstance(object_in, string_types)


def ascii_encode_value(value):
    """Encode an object as ascii, if possible.

    :param object: An object of any type.
    :return: An ascii encoding of the object.  The object if that encoding is not defined.
    """

    if(is_nonstring_iterable(value)):
        result = [ascii_encode_value(value_i) for value_i in value]
    elif(isinstance(value, string_types)):
        result = str(value)
    else:
        result = value
    return result


def ascii_encode_dict(data):
    """Encode the keys and values of a dictionary as ascii.

    :param object: A dictionary.
    :return: An ascii encoding of the dictionary.
    """

    # Split the dictionary into keys and values
    keys = data.keys()
    values = data.values()
    # Convert keys
    #keys_ascii = [key.encode('ascii') for key in keys]
    keys_ascii = [str(key) for key in keys]
    # Convert values
    values_ascii = [ascii_encode_value(value) for value in values]
    # Return dictionary as a result
    return dict(zip(keys_ascii, values_ascii))
