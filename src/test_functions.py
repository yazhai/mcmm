import numpy as np
import jax.numpy as jnp


class ToyObjective:
    def __init__(self, dims: int = 2) -> None:
        self.dims = dims  #
        return

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1

        term1 = 1000 * jnp.sum(jnp.square((x + 1))) - 3
        term2 = jnp.sum(jnp.square(x - 1)) - 1

        result = jnp.minimum(term1, term2)

        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[-3, 3]] * self.dims)


class Levy:
    def __init__(self, dims: int = 2) -> None:
        self.dims = dims  #
        return

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1

        w = 1 + (x - 1) / 4

        # sin(pi*w0)**2
        term1 = (jnp.sin(jnp.pi * w[0])) ** 2

        # (w_d -1)**2 * (1+sin(2pi w_d)**2)
        term3 = (w[-1] - 1) ** 2 * (1 + (jnp.sin(2 * jnp.pi * w[-1])) ** 2)

        # (w_i -1)**2 * (1+10sin(pi w_i +1)**2)  , i = first_dim ~ last_but_two_dim
        term2 = (w[:-1] - 1) ** 2 * (1 + 10 * (jnp.sin(jnp.pi * w[:-1] + 1)) ** 2)
        term2 = jnp.sum(term2)

        result = term1 + term2 + term3

        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[-10, 10]] * self.dims)


class Ackley:
    def __init__(self, dims: int = 2) -> None:
        self.dims = dims
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1

        term1 = -self.a * jnp.exp(-self.b * jnp.sqrt(jnp.mean(x**2)))

        term2 = -jnp.exp(jnp.mean(jnp.cos(self.c * x)))

        result = term1 + term2 + self.a + jnp.exp(1)

        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[-32.768, 32.768]] * self.dims)


class Dropwave:
    def __init__(self, dims: int = 2) -> None:
        self.dims = 2  # only 2D

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        assert len(x) == self.dims

        l2_sum = jnp.sum(x**2)
        term1 = 1 + jnp.cos(12 * jnp.sqrt(l2_sum))
        term2 = 0.5 * l2_sum + 2

        result = -term1 / term2

        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[-5.12, 5.12]] * self.dims)


class SumSquare:
    def __init__(self, dims: int = 2) -> None:
        self.dims = dims

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        assert len(x) == self.dims

        coef = jnp.arange(self.dims) + 1

        result = jnp.sum(coef * (x**2))

        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[-10, 10]] * self.dims)


class Easom:
    def __init__(self, dims: int = 2) -> None:
        self.dims = 2

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        assert len(x) == self.dims

        result = -jnp.prod(jnp.cos(x)) * jnp.exp(-jnp.sum((x - np.pi) ** 2))
        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[-100, 100]] * self.dims)


class Michalewicz:
    def __init__(self, dims: int = 2, m: float = 10.0) -> None:
        self.dims = dims
        self.m = m

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1

        sin_term = jnp.sin(x)

        index_term = np.arange(len(x)) + 1.0
        m_term = jnp.sin((x**2) * index_term / np.pi) ** (2 * self.m)

        result = -jnp.sum(sin_term * m_term)
        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[0, np.pi]] * self.dims)
