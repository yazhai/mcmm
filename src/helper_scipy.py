import numpy as np
from scipy.optimize import root as scipy_root


def root_find(fn: callable, x0, gradient: callable = None):
    """
    Find a root of a function (f(x) == 0), using scipy.optimize.root

    Input:
        fn: a function that takes a vector x and returns a vector y
        x0: initial guess
        gradient: gradient of fn, callable (default = None)

    Output:
        x: the best x found. If solution is found, it is the root of fn; otherwise, it is the best x found
        y: the best y found. If solution is found, it is 0; otherwise, it is the best y found

    """
    dims = x0.shape[-1]

    # scipy.optimize.root requies the objective function as n-dim vector
    # so we wrap the function to return a vector
    def fn_wrapper(x):
        result = np.zeros(dims)
        result[0] = fn(x)
        return result

    # scipy.optimize.root requies the jacobian as a matrix
    # so we wrap the gradient to return a matrix
    def jacob_wrapper(x):
        if gradient is None:
            return None
        jacob = np.zeros((dims, dims))
        jacob[0, :] = gradient(x)
        return jacob

    jacob = False
    if gradient is not None:
        jacob = jacob_wrapper

    # call scipy.optimize.root
    # if fails, return the best x/y found
    solution = scipy_root(
        fn_wrapper,
        x0=x0,
        jac=jacob,
        method="hybr",
    )

    x = solution.x
    y = solution.fun[0]

    return x, y
