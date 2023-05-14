import numpy as np
import jax.numpy as jnp

import torch
import torch.nn as nn
import sympy

from .nlp import *


class TestFunction:
    def __init__(self, dims: int = 2) -> None:
        self.dims = dims  #

    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def get_default_domain(self) -> np.ndarray:
        raise NotImplementedError

    def expression(self):
        raise NotImplementedError

    def get_default_bounds(self) -> np.ndarray:
        try:
            box = self.get_default_domain()

            lb = []
            ub = []
            for b in box:
                lb.append(b[0])
                ub.append(b[1])

            return lb, ub
        except:
            raise NotImplementedError


class ToyObjective(TestFunction):
    def __init__(self, dims: int = 2) -> None:
        super().__init__(dims)

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1

        term1 = 1000 * jnp.sum(jnp.square((x + 1))) - 3
        term2 = jnp.sum(jnp.square(x - 1)) - 1

        result = jnp.minimum(term1, term2)

        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[-3, 3]] * self.dims)

    def expression(self) -> str:
        dims = self.dims
        variables = [f"x[{dims}]"]
        x_plus_1 = [f"((x[{i}] + 1)^2)" for i in range(dims)]
        x_minus_1 = [f"((x[{i}] - 1)^2)" for i in range(dims)]

        # write the expression for levy function
        expression_1 = "-3"
        expression_2 = "-1"

        for i in range(dims):
            expression_1 += f" + 1000 * {x_plus_1[i]}"
            expression_2 += f" + {x_minus_1[i]}"

        expression = f"min(({expression_1}), ({expression_2}))"

        return variables, expression

    def local_lb_expression(self) -> str:
        dims = self.dims

        variables = [f"x[{dims}]"]
        x_minus_1 = [f"( 0.5 * (x[{i}] - 1)^2)" for i in range(dims)]

        expression = "-1.5"
        for i in range(dims):
            expression += f" + {x_minus_1[i]}"

        return variables, expression

    def global_lb_expression(self) -> str:
        dims = self.dims

        variables = [f"x[{dims}]"]
        x_minus_1 = [f"( 0.1 * (x[{i}] + 1)^2)" for i in range(dims)]

        expression = "-3.1"
        for i in range(dims):
            expression += f" + {x_minus_1[i]}"

        return variables, expression


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

    def expression(self) -> str:
        """
        Return the levy function expression in string format

        return:
        variables in the levy function, list of str
        expression of the levy function, str
        """

        dims = self.dims
        variables = [f"x[{dims}]"]
        w = [f"(1 + (x[{i}] - 1) / 4)" for i in range(dims)]

        # write the expression for levy function
        expression = f"sin(pi * {w[0]})^2"

        for i in range(dims - 1):
            expression += f"+ ({w[i]}-1) ^2 * (1 + 10 * sin(pi * {w[i]} + 1)^2) "

        expression += f"+ ({w[-1]}-1)^2 * (1 + sin(2 * pi * {w[-1]})^2)"

        return variables, expression

    def local_lb_expression(self):
        dims = self.dims

        x = [f"x[{i}]" for i in range(dims)]
        expression = ""

        for i in range(dims):
            expression += f" + (0.05 * ({x[i]})^2)"

        variables = [f"x[{dims}]"]

        return variables, expression

    def global_lb_expression(self):
        dims = self.dims

        x = [f"x[{i}]" for i in range(dims)]
        expression = ""

        for i in range(dims):
            expression += f" + (5e-5 * ({x[i]} - 1.0)^2)"

        variables = [f"x[{dims}]"]

        return variables, expression


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

    def expression(self):
        dims = self.dims
        a = self.a
        b = self.b
        c = self.c

        # symbolic variables in use
        x = [f"x[{i}]" for i in range(dims)]

        # sum of x_i^2
        sum_square = ""
        for i in range(dims):
            sum_square += f" + ({x[i]}^2)"
        sum_square = f"sqrt(({sum_square}) / {dims})"
        term1 = f"-({a}) * exp(-({b})*{sum_square})"

        # sum of cos(c * x_i)
        sum_cos = ""
        for i in range(dims):
            sum_cos += f" + cos(({c})*{x[i]})"
        term2 = f" - exp(({sum_cos}) / {dims})"

        term3 = f" + ({a}) + exp(1.0) "

        expression = f" ({term1}) + ({term2}) + ({term3})"

        # make variable as a vector
        variables = [f"x[{dims}]"]

        return variables, expression

    def local_lb_expression(self) -> str:
        dims = self.dims

        x = [f"x[{i}]" for i in range(dims)]
        expression = " + 3.6"

        for i in range(dims):
            expression += f" + (0.5 * ({x[i]} - 1.0)^2)"

        variables = [f"x[{dims}]"]
        return variables, expression

    def global_lb_expression(self) -> str:
        dims = self.dims

        x = [f"x[{i}]" for i in range(dims)]
        expression = ""

        for i in range(dims):
            expression += f" + (5e-5 * ({x[i]})^2)"

        variables = [f"x[{dims}]"]
        return variables, expression


class Dropwave(TestFunction):
    def __init__(self, dims: int = 2) -> None:
        super().__init__(2)  # only 2D

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

    def expression(self):
        dims = self.dims
        x = [f"x[{i}]" for i in range(dims)]

        l2_sum = ""
        for i in range(dims):
            l2_sum += f" + (({x[i]})^2)"
        l2_sum_sqrt = f"sqrt({l2_sum})"
        term1 = f"1.0 + cos(12.0 * ({l2_sum_sqrt}))"
        term2 = f"0.5 * ({l2_sum}) + 2.0"

        expression = f"-({term1}) / ({term2})"
        variables = [f"x[{dims}]"]
        return variables, expression


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

    def expression(self):
        dims = self.dims

        x = [f"x[{i}]" for i in range(dims)]

        variables = [f"x[{dims}]"]
        expression = ""
        for i in range(dims):
            expression += f" + (({i+1}) * ({x[i]})^2)"

        return variables, expression


class Easom(TestFunction):
    def __init__(self, dims: int = 2) -> None:
        super().__init__(2)  # only 2D

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        assert len(x) == self.dims

        result = -jnp.prod(jnp.cos(x)) * jnp.exp(-jnp.sum((x - np.pi) ** 2))
        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[-100, 100]] * self.dims)

    def expression(self):
        dims = self.dims
        x = [f"x[{i}]" for i in range(dims)]

        sum_term = ""
        for i in range(dims):
            sum_term += f" + (({x[i]}) - pi)^2"
        exp_term = f"exp(-({sum_term}))"

        expression = f"-({exp_term})"
        for i in range(dims):
            expression += f" * (cos(({x[i]})))"

        variables = [f"x[{dims}]"]
        return variables, expression


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

    def expression(self):
        dims = self.dims
        x = [f"x[{i}]" for i in range(dims)]

        sin_term = ""
        for i in range(dims):
            sin_term += (
                f" + sin(({x[i]})) * (sin(({x[i]})^2 * ({i+1}) / pi))^({int(2*self.m)})"
            )

        expression = f"-({sin_term})"
        variables = [f"x[{dims}]"]
        return variables, expression


class NeuralNetworkOneLayer(TestFunction):
    def __init__(
        self, dims: int = 2, domain=None, hidden_dims=16, state_dict=None, device="cpu"
    ) -> None:
        super().__init__(dims)
        self.model = nn.Sequential(
            nn.Linear(dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1),
        )

        self.model.to(device)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.hidden_dims = hidden_dims
        self.device = device

        if domain is None:
            domain = self.get_default_domain()

        self.domain = domain

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1

        # nn_in = torch.FloatTensor(x).to(self.device)
        # with torch.no_grad():
        #     nn_out = self.model(nn_in)
        # result = nn_out.item()

        # return result

        x = jnp.expand_dims(x, axis=1)
        w0, b0 = [
            mat.detach().cpu().numpy()
            for mat in [self.model[0].weight, self.model[0].bias]
        ]
        w0, b0 = jnp.array(w0), jnp.array(b0)
        x = jnp.dot(w0, x) + jnp.expand_dims(b0, axis=1)
        x = jnp.maximum(0, x)
        w1, b1 = [
            mat.detach().cpu().numpy()
            for mat in [self.model[2].weight, self.model[2].bias]
        ]
        w1, b1 = jnp.array(w1), jnp.array(b1)
        x = jnp.dot(w1, x) + jnp.expand_dims(b1, axis=1)

        result = x[0][0]

        return result

    def get_default_domain(self) -> np.ndarray:
        return np.array([[-10, 10]] * self.dims)

    def expression(self):
        layer_idx = 2
        layer = self.model[layer_idx]

        weight = layer.weight
        bias = layer.bias
        width = weight.shape[1]

        inter_vars = sympy.symbols(
            ", ".join(["x[{}][{}]".format(layer_idx, i) for i in range(width)])
        )

        X = np.expand_dims(np.array(inter_vars), axis=1)
        W = weight.detach().cpu().numpy()
        b = bias.detach().cpu().numpy()
        expression = (W @ X).squeeze(axis=1) + b
        expression = expression.item()

        layer_idx = 1
        layer = self.model[layer_idx]
        layer

        prev_inter_vars = inter_vars
        width = len(inter_vars)
        inter_vars = sympy.symbols(
            ", ".join(["x[{}][{}]".format(layer_idx, i) for i in range(width)])
        )

        replace_dict = {}
        for curr_var, prev_var in zip(inter_vars, prev_inter_vars):
            replace_dict[prev_var] = sympy.Max(curr_var, 0)

        prev_expression = expression
        for key in replace_dict:
            expression = expression.subs(key, replace_dict[key])

        layer_idx = 0
        layer = self.model[layer_idx]

        weight = layer.weight
        bias = layer.bias
        width = weight.shape[1]

        prev_inter_vars = inter_vars
        inter_vars = sympy.symbols(", ".join(["x[{}]".format(i) for i in range(width)]))
        if isinstance(inter_vars, sympy.Symbol):
            inter_vars = [inter_vars]

        X = np.expand_dims(np.array(inter_vars), axis=1)
        W = weight.detach().cpu().numpy()
        b = bias.detach().cpu().numpy()

        mat_res = (W @ X).squeeze(axis=1) + b

        replace_dict = {}
        for curr_expr, prev_var in zip(mat_res, prev_inter_vars):
            replace_dict[prev_var] = curr_expr

        prev_expression = expression
        for key in replace_dict:
            expression = expression.subs(key, replace_dict[key])

        expr_nn = str(expression).replace("Max", "max")

        variables = [f"x[{self.dims}]"]

        return variables, expr_nn


class NeuralNetworkOneLayerTrained(NeuralNetworkOneLayer):
    def __init__(self, model_path, device="cpu"):
        import os

        assert os.path.exists(model_path), f"Model path {model_path} does not exist"

        model_info = torch.load(model_path, map_location=device)
        state_dict = model_info["state_dict"]
        input_dims = model_info["input_dims"]
        hidden_dims = model_info["hidden_dims"]
        bounds = model_info["bounds"]

        super().__init__(
            dims=input_dims,
            domain=bounds,
            hidden_dims=hidden_dims,
            state_dict=state_dict,
            device=device,
        )

    def get_default_domain(self) -> np.ndarray:
        return self.domain


class Biggsbi1(TestFunction):
    def __init__(self, dims: int = 1001) -> None:
        super().__init__(1001)

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        result = biggsbi1(x)
        return result

    def get_default_domain(self) -> np.ndarray:
        bounds = np.array([[0, 0.9]] * self.dims)
        bounds[0] = [0.0, 0.0]
        return bounds

    def expression(self):
        variables = [f"x[{self.dims}]"]
        expression = biggsbi1_exp()
        return variables, expression


class Eigenals(TestFunction):
    def __init__(self, dims: int = 111) -> None:
        super().__init__(111)

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        result = eigenals(x)
        return result

    def get_default_domain(self) -> np.ndarray:
        bounds = np.array([[-10.0, 10.0]] * self.dims)
        bounds[0] = [0.0, 0.0]
        return bounds

    def expression(self):
        variables = [f"x[{self.dims}]"]
        expression = eigenals_exp()
        return variables, expression


class Harkerp(TestFunction):
    def __init__(self, dims: int = 101) -> None:
        super().__init__(101)

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        result = harkper(x)
        return result

    def get_default_domain(self) -> np.ndarray:
        bounds = np.array([[0.0, 10.0]] * self.dims)
        bounds[0] = [0.0, 0.0]
        return bounds

    def expression(self):
        variables = [f"x[{self.dims}]"]
        expression = harkper_exp()
        return variables, expression


class Vardim(TestFunction):
    def __init__(self, dims: int = 101) -> None:
        super().__init__(101)

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        result = vardim(x)
        return result

    def get_default_domain(self) -> np.ndarray:
        bounds = np.array([[-10.0, 10.0]] * self.dims)
        bounds[0] = [0.0, 0.0]
        return bounds

    def expression(self):
        variables = [f"x[{self.dims}]"]
        expression = vardim_exp()
        return variables, expression


class Watson(TestFunction):
    def __init__(self, dims: int = 32) -> None:
        super().__init__(32)

    def __call__(self, x: np.ndarray) -> float:
        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)
        assert x.ndim == 1
        result = watson(x)
        return result

    def get_default_domain(self) -> np.ndarray:
        bounds = np.array([[-10.0, 10.0]] * self.dims)
        bounds[0] = [0.0, 0.0]
        return bounds

    def expression(self):
        variables = [f"x[{self.dims}]"]
        expression = watson_exp()
        return variables, expression
