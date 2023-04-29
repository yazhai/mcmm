import numpy as np

from scipy.optimize import minimize as scipy_minimize


def objective(x):
    # May be a convex function
    # Such as a quadratic function
    # or min of all elements
    return -np.sum(x**2)


def constraint(a, *args):
    # define the linear constraint x.dot(a) < y
    x, y = args
    return y - x.dot(a)


def quadra_lb(x, y):
    # find A, such that x.dot(A).dot(x) < y
    # A is a diagonal matrix
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
        options={"disp": True},
    )

    return res.x
