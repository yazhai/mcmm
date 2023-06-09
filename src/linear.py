import numpy as np

from scipy.optimize import minimize as scipy_minimize


def objective(x):
    # May be a convex function
    # E.g. a quadratic function (note: this is incorrect. Max the sum of x**2 may lead to a large value in one dimension but all others to 0)
    # return -np.sum(x**2)
    #
    # We want to maximize the min of all elements
    return -np.min(x)


def constraint(a, *args):
    # define the linear constraint x.dot(a) < y
    x, y = args
    return y - x.dot(a)


def max_linear_weight(x, y):
    """
    Maximize the weights A with objective function sum(A**2),
    by the constraint x.dot(A) <= y and A >= 0

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)

    Returns
    -------
    A : np.ndarray, shape (n_features,)
    """

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert x.ndim == 2

    dims = x.shape[1]
    a0 = np.ones(dims)  # initial guess
    bounds = [(0, None) for _ in range(dims)]

    res = scipy_minimize(
        objective,  # objective function
        a0,  # initial guess
        constraints={"type": "ineq", "fun": constraint, "args": (x, y)},
        bounds=bounds,  # lb/ub box
        method="SLSQP",
        options={"disp": False},
    )

    return res.x
