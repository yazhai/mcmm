from pyibex import Interval, Function, CtcFwdBwd


def root_box(variables, expression, initial_box, value=0.0):
    """
    Find the box that may contain the root of a function

    params:
    variables: list of variables, list of str
    expression: expression of the function, expression str
    initial_box: initial box to search for the root, IntervalVector

    return:
    root box
    """
    f = Function(*variables, expression)
    X_in = initial_box.copy()

    ctc = CtcFwdBwd(f, Interval(value, value))  # root is when f = 0
    ctc.contract(X_in)

    # sep = SepFwdBwd(f, Interval(value, value))  # root is when f = 0
    # sep.separate(X_in)
    return X_in
