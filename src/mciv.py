import copy
from typing import Any

import numpy as np
import nlopt

import jax.numpy as jnp

# from jax import device_put, grad, jit, random, vmap
from jax import grad as jax_grad
from jax import hessian as jax_hessian

from pyibex import Interval, IntervalVector

from .node import RealVectorTreeNode
from .helper_ibex import evaluate_function_interval
from .box_util import partition_space


class MCIV:
    def __init__(
        self,
        fn,
        lb,
        ub,
        root=None,
        log="./log.dat",
        verbose=0,  # Run-time information level, 0: no info, 1: summary, 2: run time info, 3: entire tree info
        batch_size=1,  # Number of function evaluations at the same time
        max_iterations=10,  # Number of total iterations
        n_eval_local=10,  # Number of allowed function calls in local opt in every iteration
        n_initial="auto",  # Number of initial points
        dist_max=0.75,  # Max relative distance when create new node
        dist_min=0.25,  # Min relative distance when create new node
        dist_decay=0.5,  # decay factor of distance in every level
        clip_periodic=True,  # use periodic boundary when create new node
        node_uct_improve=0.0,  # C_d  for node
        node_uct_n_improve=0,  # number of improves
        node_uct_explore=1.0,  # C_p  for node
        leaf_improve=100.0,  # C_d'' for leaf
        leaf_explore=10.0,  # C_p'' for leaf
        branch_explore=1.0,  # C_p' for branch
        function_variables=None,  # symbolic variables of the function
        function_expression=None,  # symbolic expression of the function
        suggest_by="root",  # "root" (the root of f==lb) or "box" (within box f==lb)
        **kwargs,
    ):
        self.fn = fn
        self.lb = lb
        self.ub = ub
        self.log = log
        self.dims = len(lb)
        self.verbose = verbose

        # root is initialized to the anchor
        # if anchor is provided
        self.root_anchor = root
        self.root = None
        if n_initial == "auto":
            self.n_initial = 2 * self.dims + 1
        else:
            assert n_initial > 0
            self.n_initial = n_initial

        # Caches for history and node best
        self._cache_expand_size = 1000

        # history saves the history of all evaluations
        self.if_save_history = True
        self._history = None
        self.history_current_idx = -1

        # node best saves the best value of each node
        self.if_save_node_best = True
        self._node_best = None
        self.node_best_current_idx = -1

        # iterations
        self.max_iterations = max_iterations

        # new node position
        self.dist_max = dist_max  # Max relative distance when create new node
        self.dist_min = dist_min  # Min relative distance when create new node
        self.dist_decay = dist_decay  # decay factor of distance in every level
        self.clip_periodic = clip_periodic  # use periodic boundary when create new node

        # exploration related
        self.node_uct_improve = node_uct_improve  # C_d  for node
        self.node_uct_n_improve = node_uct_n_improve  # number of improves
        self.node_uct_explore = node_uct_explore  # C_p  for node
        self.leaf_improve = leaf_improve  # C_d'' for leaf
        self.leaf_explore = leaf_explore  # C_p'' for leaf
        self.branch_explore = branch_explore  # C_d' for branch
        self.min_node_visit = 1  # minimum number of visits before expand

        # Symbolic expression of the function
        self.function_variables = function_variables
        self.function_expression = function_expression

        # Learning the lower bound
        self.suggest_failed = False
        self.lb_function = None
        self.lb_prime_function = None
        self.lb_expression = None
        self.lb_variables = None
        self.lb_ignore_min_dist = (
            1e-4  # ignore points with distance smaller than this value
        )
        self.lb_lower_by = 0.1  # lower the lower bound by this value
        self.lb_ignore_coeff = (
            1e-4  # ignore points with coefficient smaller than this value
        )
        self.suggest_by = (
            suggest_by  # "root" (the root of f==lb) or "box" (within box f==lb)
        )

        # dict to keep the lb/ub, gradient, and hessian for all node
        self.node_bounds = {}
        self._node_grad_expect = {}
        self._node_hessian_expect = {}

        # Update other parameters
        self._box_init = None
        self.set_parameters(**kwargs)

        return

    def set_parameters(self, **kwargs):
        for key, val in kwargs.items():
            if getattr(self, key, "Not Exist") != "Not Exist":
                setattr(self, key, val)
        return

    def optimize(self, restart=False, **kwargs):
        self.set_parameters(**kwargs)

        if not restart:
            self.root = None
            if self.if_save_history:
                self._history = self._cache_expand()
                self.history_current_idx = -1
            if self.if_save_node_best:
                self._node_best = self._cache_expand()
                self.node_best_current_idx = -1

        if self.root is None:
            if self.verbose >= 2:
                print("Creating root node ...")
            self.create_root(self.root_anchor)
            if self.verbose >= 1:
                print(f"  Best found value until iteration {0} is: {self.root.y} \n")

        for tt in range(1, self.max_iterations + 1):
            if self.verbose >= 1:
                print(f"Running iteration: {tt} ...\n")

            # selection and expansion
            node = self.select()

            # new samples, and new guess within the node
            y_best, x_best = self.node_exploitation(node)

            # backpropagation is done in above step
            # # back propagation
            # self.backprop(node, x_best, y_best)

            # Information
            if self.verbose >= 1:
                print(f"  Best found value until iteration {tt} is: {self.root.y} \n")

        return self.root.y

    def create_root(self, *args, **kwargs):
        # create root node by anchor
        self.root = self.create_empty_node()
        self.root.is_root = True
        self.root.set_level()
        self.root.identifier = "root"
        self.node_bounds[self.root.identifier] = [self.lb, self.ub]

        # create root node by anchor
        self.node_exploitation(self.root)
        return

    def select(self, node=None):
        if node is None:
            node = self.root

        while node.children:
            node = self.select_by_uct_interval(node)

        return node

    def select_by_uct_interval(self, node):
        self.split_box_for_children(node)
        ucts = []
        for child in node.children:
            fn_interv = self.node_function_interval(child)
            uct = -child.y - self.node_uct_explore * fn_interv[0]
            ucts.append(uct)

        best = np.argmax(ucts)
        return node.children[best]

    def split_box_for_children(self, node):
        lb, ub = self.node_bounds[node.identifier]

        # split the box to all children
        if node.children:
            # use index to make sure the order is correct
            xs = []
            for ii in range(len(node.children)):
                child = node.children[ii]
                xs.append(child.anchor)
            xs = np.array(xs)

            # split the box; randomly choose a dimension
            split_dim = np.random.randint(0, self.dims)
            boxes = partition_space(lb, ub, xs, split_dim)

            if self.verbose >= 3:
                print("splitted box:", boxes)

            # update the bounds for all children
            for ii in range(len(node.children)):
                child = node.children[ii]
                name = child.identifier
                _lb, _ub = boxes[ii]
                self.node_bounds[name] = [_lb, _ub]
        return

    def backprop(self, node, x_best=None, y_best=None, iterative=True):
        curr_node = node

        # Retrieve the best value and its corresponding x
        # if not provided
        # Skip the leaf node because it has been recorded
        if (x_best is None) or (y_best is None):
            x_best = node.X
            y_best = node.y
            curr_node = node.parent

        if self.if_save_node_best:
            self.add_node_best(x_best, y_best)

        if iterative:
            while curr_node:
                curr_node.update_stat(y_best, x_best)
                curr_node.visit_once()
                curr_node = curr_node.parent
        return

    def create_node(self, x_anchor=None, y_anchor=None):
        node = self.create_empty_node()

        if x_anchor is None:
            x_anchor = self.uniform_sample()
        else:
            x_anchor = copy.deepcopy(x_anchor)
        if y_anchor is None:
            y_anchor = self.get_groundtruth(x_anchor)

        assert x_anchor.ndim == 1
        y_X = np.array([y_anchor, *x_anchor])
        node.add_samples(y_X)
        return node

    def create_empty_node(self):
        node = RealVectorTreeNode(
            improve_factor=self.node_uct_improve,
            improve_count=self.node_uct_n_improve,
        )
        return node

    def uniform_sample(self, lb=None, ub=None):
        if lb is None:
            lb = self.lb
        if ub is None:
            ub = self.ub
        ratio = self._cache_rand()
        sample = lb + (ub - lb) * ratio
        return sample

    def get_groundtruth(self, input_value):
        x = input_value
        if isinstance(x, RealVectorTreeNode):
            x = x.X

        x = np.clip(x, self.lb, self.ub)
        y = self.fn(x)

        if isinstance(y, np.ndarray) or isinstance(y, jnp.ndarray):
            y = y.item()

        if self.if_save_history:
            self.add_history(x, y)
        return y

    def add_history(self, x, y):
        self._add_to_cache(x, y, cache="global")
        return

    def add_node_best(self, x, y):
        self._add_to_cache(x, y, cache="node")
        return

    def _add_to_cache(self, x, y, cache="global"):
        idx = -1
        if cache == "global":
            self.history_current_idx += 1
            if self.history_current_idx >= self._history.shape[0]:
                self._history = self._cache_expand(self._history)
            history = self._history
            idx = self.history_current_idx
        elif cache == "node":
            self.node_best_current_idx += 1
            if self.node_best_current_idx >= self._node_best.shape[0]:
                self._node_best = self._cache_expand(self._node_best)
            history = self._node_best
            idx = self.node_best_current_idx
        else:
            raise ValueError("Invalid cache type.")

        history[idx, 0] = y
        history[idx, 1:] = x
        return

    def _cache_rand(self, cache_size=1000):
        # Cache uniform sampling from numpy.random.rand
        # Store a pool of random samples as class variable for next call.
        # Input:
        #     cache_size
        # Output:
        #     A random sample
        try:
            # use cached random samples
            self._uniform_sample_pool_last_idx += 1
            return self._uniform_sample_pool[self._uniform_sample_pool_last_idx]
        except:
            # create new random samples
            self._uniform_sample_pool = np.random.rand(cache_size, self.dims)
            self._uniform_sample_pool_last_idx = 0
            return self._uniform_sample_pool[self._uniform_sample_pool_last_idx]

    def _cache_expand(self, cache_array=None):
        if cache_array is None:
            cache_array = np.zeros((self._cache_expand_size, 1 + self.dims))
        else:
            cache_array = np.concatenate(
                (cache_array, np.zeros((self._cache_expand_size, 1 + self.dims))),
                axis=0,
            )
        return cache_array

    def _box_to_list(self, box):
        lb = []
        ub = []
        for intv in box:
            lb.append(intv[0])
            ub.append(intv[1])
        return lb, ub

    def _list_to_box(self, lb, ub):
        box = []
        for ii in range(len(lb)):
            box.append([lb[ii], ub[ii]])
        return IntervalVector(box)

    @property
    def history(self):
        return self._history[: self.history_current_idx + 1, :]

    @property
    def node_best(self):
        return self._node_best[: self.node_best_current_idx + 1, :]

    def expect_grad_hessian(self, inputs):
        # Compute the expected gradient and hessian(diagonal) of the function
        # inputs: list of nodes or list of x points
        # return: expectation of fprime and fhess as np.array(dims)
        fprime = jnp.zeros(self.dims)
        fhess = jnp.zeros((self.dims, self.dims))
        for v in inputs:
            if isinstance(v, RealVectorTreeNode):
                x = v.X
            else:
                x = v
            fprime += self._grad(x)
            fhess += self._hessian(x)
        fprime /= len(inputs)
        fhess /= len(inputs)
        fhess_diag = jnp.diag(fhess)
        return fprime, fhess_diag

    def _grad(self, x):
        return jax_grad(self.fn)(x)

    def _hessian(self, x):
        return jax_hessian(self.fn)(x)

    def _newton_step(self, x, gradinet, hessian_diagonal):
        x_new = x - gradinet / hessian_diagonal
        return x_new

    def _gd_step(self, x, gradient, step_size=0.1):
        x_new = x - gradient * step_size
        return x_new

    def node_function_interval(self, node):
        # Evaluate the function value interval over a input box
        # lb/ub: list of lower/upper bounds
        # return: function value Interval
        lb, ub = self.node_bounds[node.identifier]
        box = self._list_to_box(lb, ub)

        # Compute the function value interval
        function_interval = evaluate_function_interval(
            self.function_variables, self.function_expression, box
        )
        return function_interval

    def _guess_good(self, x, grad, hessian):
        # Guess a good point to improve the current node
        # Convex: use newton's method
        # Concave: use gradient descent
        x_new_nt = self._newton_step(x, grad, hessian)
        x_new_gd = self._gd_step(x, grad)

        x_new = np.zeros(self.dims)

        mask_convex = hessian >= 0
        x_new[mask_convex] = x_new_nt[mask_convex]
        x_new[~mask_convex] = x_new_gd[~mask_convex]

        return x_new

    def guess_good(self, node):
        # Compute the expected gradient and hessian(diagonal) from children
        fprime_child, fhess_child = self.expect_grad_hessian(node.children)

        # Compute local gradient and hessian(diagonal) from sampling
        neighbor_samples = 20
        dx = 1.0
        x = copy.deepcopy(node.X)
        lb = x - dx
        ub = x + dx
        grads = []
        for i in range(neighbor_samples):
            xi = self.uniform_sample(lb, ub)
            grads.append(self._grad(xi))
        fprime_local = np.mean(grads, axis=0)

        # Guess a good point to improve the current node
        x_new = self._guess_good(node.X, fprime_local, fhess_child)

        return x_new

    def guess_good_local(self, node):
        # Use newton's method to find a good point to improve the current node
        x = copy.deepcopy(node.X)
        fprime, fhess = self.expect_grad_hessian([x])

        # Guess a good point to improve the current node
        x_new = self._guess_good(x, fprime, fhess)

        return x_new

    def node_exploitation(self, parent):
        # Create next level nodes by random sampling
        parent_name = parent.identifier
        for ii in range(self.n_initial):
            anchor = self.uniform_sample(*self.node_bounds[parent_name])
            # Put nodes into the child level
            child = self.create_node(anchor)
            child.set_parent(parent)
            child.identifier = parent_name + "_" + f"{len(parent.children)}"
            X_best = child.X
            y_best = child.y
            self.backprop(child, X_best, y_best)

        if self.verbose >= 2:
            print(
                f"After creating child nodes for {parent_name}, best point: ",
                parent.y,
                parent.X,
            )

        # Create a new sample by guessing a good point by neighbor fprime and global fhess
        # using newton's method for convex functions
        # using gradient descent for concave functions
        anchor = self.guess_good(parent)
        child = self.create_node(anchor)
        child.set_parent(parent)
        child.identifier = parent.identifier + "_" + f"{len(parent.children)}"
        X_best = child.X
        y_best = child.y
        self.backprop(child, X_best, y_best)
        if self.verbose >= 2:
            print(
                f"New node {child.identifier} by optimizing with global Hessian and neighbor gradient: ",
                child.y,
                child.X,
            )

        # Create another new sample by guessing a good point from local fprime and fhess
        # using newton's method for convex functions
        # using gradient descent for concave functions
        anchor = self.guess_good_local(parent)
        child = self.create_node(anchor)
        child.set_parent(parent)
        child.identifier = parent.identifier + "_" + f"{len(parent.children)}"
        X_best = child.X
        y_best = child.y
        self.backprop(child, X_best, y_best)

        if self.verbose >= 2:
            print(
                f"New node {child.identifier} by optimizing with local Hessian and local gradient: ",
                child.y,
                child.X,
            )

        return parent.y, parent.X


if __name__ == "__main__":
    # run a test case with Ackley function
    from .test_functions import Ackley

    dims = 2
    fn = Ackley(dims=dims)
    lb = -10 * np.ones(dims)
    ub = 10 * np.ones(dims)

    var_exp = None
    fn_exp = None
    try:
        var_exp, fn_exp = fn.expression()
    except:
        pass

    alg = MCIV(
        fn=fn,
        lb=lb,
        ub=ub,
        function_variables=var_exp,
        function_expression=fn_exp,
        n_eval_local=1000,
        lb_ignore_coeff=1e-3,
        lb_displacement=0.1,
    )

    result_list = []
    history_list = []
    for _ in range(5):
        res = alg.optimize(
            verbose=2,
            max_iterations=100,
            n_initial=5,
            node_uct_explore=1.0,
        )
        result_list.append(res)
        history_list.append(alg.history)

    print("Result list:", result_list)
