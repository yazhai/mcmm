import numpy as np
import jax.numpy as jnp


class Levy:
    def __init__(self, dims: int = 2) -> None:
        self.dims = dims
        return

    def __call__(self, x: np.ndarray) -> float:
        
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        
        w = 1 + (x - 1) / 4
        
        # sin(pi*w0)**2
        term1 = (jnp.sin(jnp.pi * w[0])) ** 2
        
        # (w_d -1)**2 * (1+sin(2pi w_d)**2)
        term3 = (w[-1] - 1) ** 2 * (
            1 + (jnp.sin(2 * jnp.pi * w[-1])) ** 2
        )
        
        # (w_i -1)**2 * (1+10sin(pi w_i +1)**2)  , i = first_dim ~ last_but_two_dim
        term2 = (w[:-1]-1)**2 * (
            1 + 10 * (jnp.sin(jnp.pi * w[:-1] + 1)) ** 2
        )
        term2 = jnp.sum(term2) 
        
        result = term1 + term2 + term3
        
        return result


class Ackley:
    def __init__(self, dims=10):
        self.dims = dims

    def __call__(self, x):
        if isinstance(x, numbers.Number):
            x = np.array([x])
        if isinstance(x, list):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            x = x.X

        assert len(x) == self.dims
        assert x.ndim == 1
        x = np.clip(x, self.lb, self.ub)
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        self.counter += 1

        result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) -
                  np.exp(np.cos(2*np.pi*x).sum() / x.size) + 20 + np.e)
        self.history.append(result)
        return result


class Dropwave:
    def __init__(self, ):
        self.dims = 2

    def __call__(self, x):
        if isinstance(x, numbers.Number):
            x = np.array([x])
        if isinstance(x, list):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            x = x.X

        assert len(x) == self.dims
        assert x.ndim == 1
        x = np.clip(x, self.lb, self.ub)
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        self.counter += 1

        l2_sum = np.sum(x**2)
        result = - (1 + np.cos(12*np.sqrt(l2_sum))) / (0.5*l2_sum+2)
        return result


class GramacyLee:
    def __init__(self, ):
        self.dims = 1

    def __call__(self, x):
        if isinstance(x, numbers.Number):
            x = np.array([x])
        if isinstance(x, list):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            x = x.X

        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        self.counter += 1

        result = np.sin(10*np.pi*x) / (2*x) + (x-1)**4

        if isinstance(result, np.ndarray):
            result = result[0]

        return result


class SumSquare:
    def __init__(self, dims=2):
        self.dims = dims

    def __call__(self, x):
        if isinstance(x, numbers.Number):
            x = np.array([x])
        if isinstance(x, list):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            x = x.X

        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        self.counter += 1

        coef = np.arange(self.dims) + 1

        result = np.sum(coef * (x**2))

        return result


class CrossInTray:
    def __init__(self, ):
        self.dims = 2

    def __call__(self, x):
        if isinstance(x, numbers.Number):
            x = np.array([x])
        if isinstance(x, list):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            x = x.X

        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        self.counter += 1

        expterm = np.exp(np.abs(100-np.sqrt(np.sum(x**2)) / np.pi))
        twosin = np.prod(np.sin(x))

        result = -0.0001 * (np.abs(twosin * expterm)+1)**0.1

        return result


class Easom:
    def __init__(self, ):
        self.dims = 2

    def __call__(self, x):
        if isinstance(x, numbers.Number):
            x = np.array([x])
        if isinstance(x, list):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            x = x.X

        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        self.counter += 1

        result = - np.prod(np.cos(x)) * np.exp(- np.sum((x-np.pi)**2))
        return result


class Michalewicz:
    def __init__(self, dims=2, m=10):
        self.dims = dims

        self.m = 10

    def __call__(self, x):
        if isinstance(x, numbers.Number):
            x = np.array([x])
        if isinstance(x, list):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            x = x.X

        assert len(x) == self.dims
        assert x.ndim == 1
        x = np.clip(x, self.lb, self.ub)
        try:
            assert np.all(x <= self.ub) and np.all(x >= self.lb)
        except:
            x = np.clip(x, self.lb, self.ub)
        self.counter += 1

        a_sin = np.sin(x)
        b = x**2 * (np.arange(x.shape[0]) + 1) / np.pi
        b_sin = np.sin(b)**(2*self.m)

        result = -np.sum(a_sin*b_sin)

        self.history.append(result)
        return result
