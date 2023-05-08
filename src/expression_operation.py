import numpy as np

# from pyibex import Interval, IntervalVector, Function, CtcFwdBwd, SepFwdBwd


# def root_box(variables, expression, initial_box, value=0.0):
#     """
#     Find the box that may contain the root of a function

#     params:
#     variables: list of variables, list of str
#     expression: expression of the function, expression str
#     initial_box: initial box to search for the root, IntervalVector

#     return:
#     root box
#     """
#     f = Function(*variables, expression)
#     X_in = initial_box.copy()

#     ctc = CtcFwdBwd(f, Interval(value, value))  # root is when f = 0
#     ctc.contract(X_in)

#     # sep = SepFwdBwd(f, Interval(value, value))  # root is when f = 0
#     # sep.separate(X_in)
#     return X_in


def expression_minus(exp1: str, exp2: str) -> str:
    """
    Return the expression of the difference of two expressions

    """
    return f"({exp1}) - ({exp2})"


def expression_quadratic(
    x0: list, k: list, c: float = 0.0, k_thres: float = 0.0
) -> str:
    """
    Return the quadratic function expression in string format
    The function is in the form of \sum (k * (x - x0)^2) + c

    params:
    x0: center of the quadratic function, float or numpy array
    k: coefficient of the quadratic function, float or numpy array
    c: constant of the quadratic function, float
    k_thres: smallest coefficient that will be kept non-zero, float

    return:
    variables in the quadratic function, list of str
    expression of the quadratic function, str
    """

    x0_list = x0
    if isinstance(x0, np.ndarray):
        assert x0.ndim == 1
        x0_list = x0.tolist()
    if not isinstance(x0_list, list):
        x0_list = [x0_list]

    k_list = k
    if isinstance(k, np.ndarray):
        assert k.ndim == 1
        k_list = k.tolist()

    assert len(x0_list) == len(k_list)

    variables = [f"x[{len(x0_list)}]"]
    expression = ""
    for i in range(len(x0_list)):
        if k_list[i] > k_thres:
            expression += f" + ({k_list[i]} * (x[{i}] - ({x0_list[i]}))^2) "

    expression += f" + ({c}) "

    return variables, expression
