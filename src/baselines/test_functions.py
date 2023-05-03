import numpy as np
import jax.numpy as jnp
import gurobipy as gp
from gurobipy import GRB
import gurobipy as gp


from typing import List


def add_cont_var_unbounded(m: gp.Model, name: str) -> gp.Var:
    return m.addVar(lb=float("-inf"), ub=float("inf"), vtype=GRB.CONTINUOUS, name=name)
def add_cont_var_positive(m: gp.Model, name: str) -> gp.Var:
    return m.addVar(lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name=name)
def add_cont_var_negative(m: gp.Model, name: str) -> gp.Var:
    return m.addVar(lb=float("-inf"), ub=0, vtype=GRB.CONTINUOUS, name=name)


class TestFunction:
    def __init__(self, dims: int = 2) -> None:
        self.dims = dims  #

    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def get_default_domain(self) -> np.ndarray:
        raise NotImplementedError

    def encode_to_gurobi(self, m: gp.Model = None, x: List[gp.Var] = None):
        raise NotImplementedError


class Levy(TestFunction):
    def __init__(self, dims: int = 2) -> None:
        super().__init__(dims)

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

    def encode_to_gurobi(self, m: gp.Model = None, x: List[gp.Var] = None):
        if m is None and x is not None:
            assert False, "m is None but x is not None"

        if m is None:
            m = gp.Model("Levy")
        if x is None:
            x = [add_cont_var_unbounded(m, "x" + str(i)) for i in range(self.dims)]

        assert len(x) == self.dims, "len(x) != self.dims"

        w = []
        for i in range(self.dims):
            w.append(add_cont_var_unbounded(m, "w" + str(i)))
            m.addConstr(w[i] == 1 + (x[i] - 1) / 4, "w" + str(i))

        # first term
        sin_1_in = add_cont_var_unbounded(m, "sin_1_in")
        sin_1_out = add_cont_var_unbounded(m, "sin_1_out")
        m.addConstr(sin_1_in == np.pi * w[0], "sin_1_in")
        m.addGenConstrSin(sin_1_in, sin_1_out, "sin_1_out")
        term1 = sin_1_out**2

        # last term
        sin_last_in = add_cont_var_unbounded(m, "sin_last_in")
        sin_last_out = add_cont_var_unbounded(m, "sin_last_out")
        m.addConstr(sin_last_in == 2 * np.pi * w[self.dims - 1], "sin_last_in")
        m.addGenConstrSin(sin_last_in, sin_last_out, "sin_last_out")

        term_last_1 = add_cont_var_unbounded(m, "term_last_1")
        term_last_2 = add_cont_var_unbounded(m, "term_last_2")
        m.addConstr(term_last_1 == (w[self.dims - 1] - 1) ** 2, "term_last_1")
        m.addConstr(term_last_2 == 1 + sin_last_out**2, "term_last_2")

        term_last = term_last_1 * term_last_2

        # middle terms
        # summation = 0
        summation_items = []

        for i in range(self.dims - 1):
            sin_inter_in = add_cont_var_unbounded(m, "sin_inter_in" + str(i))
            sin_inter_out = add_cont_var_unbounded(m, "sin_inter_out" + str(i))
            m.addConstr(sin_inter_in == np.pi * w[i] + 1, "const_sin_inter_in" + str(i))
            m.addGenConstrSin(
                sin_inter_in, sin_inter_out, "const_sin_inter_out" + str(i)
            )

            term_inter_1 = add_cont_var_unbounded(m, "term_inter_1" + str(i))
            term_inter_2 = add_cont_var_unbounded(m, "term_inter_2" + str(i))

            m.addConstr(term_inter_1 == (w[i] - 1) ** 2, "const_term_inter_1" + str(i))
            m.addConstr(
                term_inter_2 == 1 + 10 * sin_inter_out**2,
                "const_term_inter_2" + str(i),
            )
            summation_item = add_cont_var_unbounded(m, "var_summation_item" + str(i))
            m.addConstr(
                summation_item == term_inter_1 * term_inter_2,
                "const_summation_item" + str(i),
            )

            summation_items.append(summation_item)
            # summation += summation_item

        # Set objective
        # m.setObjective(term1 + summation + term_last, GRB.MINIMIZE)
        m.setObjective(
            gp.quicksum([term1] + summation_items + [term_last]), GRB.MINIMIZE
        )

        m.params.NonConvex = 2

        return m


class Ackley(TestFunction):
    def __init__(self, dims: int = 2) -> None:
        super().__init__(dims)
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

    def encode_to_gurobi(self, m: gp.Model = None, x: List[gp.Var] = None) -> gp.Model:
        if m is None and x is not None:
            assert False, "m is None but x is not None"

        if m is None:
            m = gp.Model("ackley")
            print("Creating new model")
        if x is None:
            x = [add_cont_var_unbounded(m, "x" + str(i)) for i in range(self.dims)]
            print("Creating new variables")

        assert len(x) == self.dims, "len(x) != self.dims"

        # First term

        # Inner summation
        squared_summation = 0
        for i in range(self.dims):
            squared_summation += x[i] ** 2

        # Square root
        sqrt_in = add_cont_var_positive(m, "sqrt_in")
        sqrt_out = add_cont_var_positive(m, "sqrt_out")

        m.addConstr(sqrt_in == squared_summation / self.dims, "sqrt_in")
        m.addGenConstrPow(sqrt_in, sqrt_out, 0.5, "sqrt")

        exp_1_in = add_cont_var_negative(m, "exp_1_in")
        exp_1_out = add_cont_var_positive(m, "exp_1_out")

        m.addConstr(exp_1_in == -self.b * sqrt_out, "exp_1_in")

        m.addGenConstrExp(exp_1_in, exp_1_out, "exp_1")

        term1 = -self.a * exp_1_out

        # Second term
        cos_summation = 0
        for d in range(self.dims):
            cos_in = add_cont_var_unbounded(m, "cos_in" + str(d))
            cos_out = add_cont_var_unbounded(m, "cos_out" + str(d))
            m.addConstr(cos_in == self.c * x[d], "cos_in" + str(d))
            m.addGenConstrCos(cos_in, cos_out, "cos_out" + str(d))
            cos_summation += cos_out
        
        exp_2_in = m.addVar(lb=-1., ub=1., vtype=GRB.CONTINUOUS, name="exp_2_in")
        # exp_2_in = add_cont_var_positive(m, "exp_2_in")
        exp_2_out = add_cont_var_positive(m, "exp_2_out")

        m.addConstr(exp_2_in == (1 / self.dims) * cos_summation, "exp_2_in")
        m.addGenConstrExp(exp_2_in, exp_2_out, "exp_2")

        term2 = -exp_2_out

        m.setObjective(term1 + term2 + self.a + np.exp(1), GRB.MINIMIZE)

        m.params.NonConvex = 2
        m.params.Presolve = 2

        return m


class Dropwave(TestFunction):
    def __init__(self, dims: int = 2) -> None:
        assert dims == 2, "Dropwave function is only defined for 2D"
        super().__init__(dims=2)

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


class SumSquare(TestFunction):
    def __init__(self, dims: int = 2) -> None:
        super().__init__(dims)

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        assert len(x) == self.dims

        coef = jnp.arange(self.dims) + 1

        result = jnp.sum(coef * (x**2))

        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[-10, 10]] * self.dims)

    def encode_to_gurobi(self, m: gp.Model = None, x: List[gp.Var] = None):
        if m is None and x is not None:
            assert False, "m is None but x is not None"

        if m is None:
            m = gp.Model("SumSquare")
        if x is None:
            x = [add_cont_var_unbounded(m, "x" + str(i)) for i in range(self.dims)]

        assert len(x) == self.dims, "len(x) != self.dims"

        summation = 0
        # Loop thorugh to get summation of all terms
        for i in range(self.dims):
            summation += (i + 1) * x[i] ** 2

        # Set objective
        m.setObjective(summation, GRB.MINIMIZE)

        m.params.NonConvex = 1

        return m


class Easom(TestFunction):
    def __init__(self, dims: int = 2) -> None:
        super().__init__(dims)

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        assert len(x) == self.dims

        result = -jnp.prod(jnp.cos(x)) * jnp.exp(-jnp.sum((x - np.pi) ** 2))
        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[-100, 100]] * self.dims)


class Michalewicz(TestFunction):
    def __init__(self, dims: int = 2, m: float = 10.0) -> None:
        super().__init__(dims)
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

    def encode_to_gurobi(self, m: gp.Model = None, x: List[gp.Var] = None):
        if m is None and x is not None:
            assert False, "m is None but x is not None"

        if m is None:
            m = gp.Model("Michalewicz")
        if x is None:
            x = [add_cont_var_unbounded(m, "x" + str(i)) for i in range(self.dims)]

        assert len(x) == self.dims, "len(x) != self.dims"

        summation = 0
        # Loop thorugh to get summation of all terms
        for i in range(self.dims):
            sin_1_in = add_cont_var_unbounded(m, "sin_1_in" + str(i))
            sin_1_in_out = add_cont_var_unbounded(m, "sin_1_in_out" + str(i))

            m.addConstr(sin_1_in == x[i], "sin_1_in" + str(i))
            m.addGenConstrSin(sin_1_in, sin_1_in_out, "sin_1_in_out" + str(i))

            sin_2_in = add_cont_var_unbounded(m, "sin_2_in" + str(i))
            sin_2_in_out = add_cont_var_unbounded(m, "sin_2_in_out" + str(i))

            m.addConstr(sin_2_in == (i + 1) * x[i] ** 2 / np.pi, "sin_2_in" + str(i))
            m.addGenConstrSin(sin_2_in, sin_2_in_out, "sin_2_in_out" + str(i))

            sin_2_out_squraed = add_cont_var_positive(m, "sin_2_out_squraed" + str(i))
            m.addConstr(sin_2_out_squraed == sin_2_in_out ** 2, "sin_2_out_squraed" + str(i))

            pow_in = add_cont_var_positive(m, "pow_in" + str(i))
            pow_out = add_cont_var_positive(m, "pow_out" + str(i))

            m.addConstr(pow_in == sin_2_out_squraed, "pow_in" + str(i))
            m.addGenConstrPow(pow_in, pow_out, self.m, "pow_out" + str(i))

            summation += sin_1_in_out * pow_out

        # Set objective
        m.setObjective(-summation, GRB.MINIMIZE)

        m.params.NonConvex = 2

        return m
