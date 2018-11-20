def find_sum(a, b):
    """Given integers a and b where a < b, return the sum a + (a + 1) + (a + 2) ... + b"""
    sum = 0
    while a <= b:
        sum += a
        a += 1
    return sum

def get_info(name, age, ssn):
    """
    Given a name, age, and ssn, this returns a string of the form, complete with newlines and tabs.

    Example: get_info('Scott', 100, 1234567890) returns:
    Name: Scott
        Age: 100
        SSN: 1234567890
    get_info('Scott', 'older', '123-45-7890') returns:
    Name: Scott
        Age: older
        SSN: 123-45-7890
    """
    return "Name: {}\n\tAge: {}\n\tSSN: {}".format(name, age, ssn)

def get_method_and_var_names(obj):
    """
    Given an object, this method returns a list of the names of the methods which do not contain "__".
    Example:
    get_method_and_var_names("a string") returns the list:
    ['capitalize', 'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'format_map', 'index', 'isalnum', 'isalpha', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']

    import numpy as np; get_method_and_var_names(np) returns the list:
    ['ALLOW_THREADS', 'AxisError', 'BUFSIZE', 'CLIP', 'ComplexWarning', ... 'where', 'who', 'zeros', 'zeros_like']
    (The full list is considerably longer than that)
    """
    list = dir(obj)
    retList = []
    for str in list:
        if(str.find('__') != 0):
            retList.append(str)
    return retList

def evaluate(f, x):
    """
    Given a function f, and a value x, this method returns the result of calling f on x.

    Example:
    evaluate(len, [1, 2, 3])
    returns 3

    def foo(var):
        return var + 5
    evaluate(foo, 1)
    returns 6.
    """
    return f(x)

def threshold_factory(thresh):
    """
    This method returns a function.
    The function it returns is a threshold function which takes an argument x and returns:
    True if x < thresh
    False if x >= thresh

    Example:
    t5 = threshold_factory(5)
    t5(1) # returns True
    t5(6) # returns False
    t_other = threshold_factory(1.234)
    t_other(1.23) # returns False
    """
    def threshold(x):
        return x < thresh
    return threshold
