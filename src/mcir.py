import copy, time

from typing import Any

import numpy as np

from collections import defaultdict

import jax.numpy as jnp

from jax import grad as jax_grad
from jax import hessian as jax_hessian
from jax import jit as jax_jit

from scipy.stats import qmc
from scipy.optimize import minimize as scipy_minimize

from pyibex import Interval, IntervalVector

from .node import RealVectorTreeNode
from .helper_ibex import evaluate_function_interval
from .box_util import partition_space


class MCIR:
    def __init__(
        self,
        fn,
        lb,
        ub,
        root=None,
        log="./log.dat",
        verbose=0,  # Run-time information level, 0: no info, 1: summary, 2: run time info, 3: entire tree info
        max_iterations=10,  # Number of total iterations
        n_opt_local=0,  # Number of allowed function calls in local opt in every iteration
        num_node_expand="auto",  # Number of initial points
        clip_periodic=True,  # use periodic boundary when create new node
        node_uct_lb_coeff=1.0,  # coefficient for interval lb
        node_uct_box_coeff=1.0,  # coefficient for box size
        node_uct_explore=1.0,  # C_p  for node
        function_variables=None,  # symbolic variables of the function
        function_expression=None,  # symbolic expression of the function
        seed=None,  # random seed
        **kwargs,
    ):
        # function, input domain, and dimension
        if not isinstance(lb, np.ndarray):
            lb = np.array(lb)
        self.lb = lb
        if not isinstance(ub, np.ndarray):
            ub = np.array(ub)
        self.ub = ub
        self.dims = len(lb)

        self.fn = fn

        # Symbolic expression of the function
        self.function_variables = function_variables
        self.function_expression = function_expression
        if (self.function_variables is None) or (self.function_expression is None):
            try:
                self.function_variables, self.function_expression = self.fn.expression()
            except Exception as e:
                print("Failed to get symbolic expression of the function")
                print(str(e))

        # logging
        self.log = log
        self.verbose = verbose

        # Caches expansion for history and node best
        self._cache_expand_size = 1000

        # history saves the history of all evaluations
        self.if_save_history = True
        self._history = None
        self.history_current_idx = -1

        # node best saves the best value of each node
        self.if_save_node_best = True
        self._node_best = None
        self.node_best_current_idx = -1

        # Nomenclature:
        # `bounds` refers to the input domain with the form of [[lb1, lb2, ...], [ub1, ub2, ...] ]
        # `box` refers to the input domain with the form of [[lb1, ub1], [lb2, ub2], ..., [lbn, ubn]]
        # `interval` refers to the output range of the function with the form of [lb, ub] (since it is a scalar function)

        # dict to save the input domain bounds for each node; node_bounds[node.name] = [lb, ub]
        self.node_bounds = defaultdict(list)

        # dict to save the lower bound of the function value for each node; node_interval[node.name] = [lb(f(node)), ub(f(node))]
        self.node_interval = defaultdict(list)
        self.node_interval_lb_from_child = defaultdict(lambda: None)
        self._interval_global = (
            None  # global function interval on the whole input domain
        )
        self._interval_max_score = 100.0  # max score of the interval when f(x) = lb

        # dict to save the index of the children that covers the box of the parent
        self.node_cover_children = defaultdict(list)

        # dict to save the size of the box of each node
        self._global_volume = self._bound_volume(self.lb, self.ub)
        self.node_box_size = defaultdict(lambda: self._global_volume)

        # Advaced sampler for creating evenly distributed points
        self.advanced_sampler = qmc.LatinHypercube(self.dims)

        ##### Optimization related ##########
        # iterations
        self.max_iterations = max_iterations

        self.seed = seed

        # root is initialized to the anchor
        # if anchor is provided
        self.root_anchor = root
        self.root = None

        # expansion
        if num_node_expand == "auto":
            self.num_node_expand = 2 * self.dims + 1
        else:
            assert num_node_expand > 0
            self.num_node_expand = num_node_expand

        # new node position
        self.clip_periodic = clip_periodic  # use periodic boundary when create new node

        # exploration related
        self.node_uct_lb_coeff = node_uct_lb_coeff  # coefficient for interval lb
        self.node_uct_box_coeff = node_uct_box_coeff  # coefficient for box size
        self.node_uct_explore = node_uct_explore  # C_p  for node

        
        # local optimization
        self.local_optimizer = None
        self.n_opt_local = n_opt_local

        ########## Update other parameters ##########

        self.time_jit = 0.0
        try:
            start = time.time()

            if self.dims > 50:
                if self.verbose >= 2:
                    print("Start jitting function...")
                self.fn = jax_jit(fn)
                if self.verbose >= 2:
                    print("Finish jitting function...")

            # jit the function gradient and hessian
            if self.verbose >= 2:
                print("start jitting grad ...")
            self._grad_jit = jax_jit(jax_grad(self.fn))
            if self.verbose >= 2:
                print("finish jitting grad ...")
                print("start jitting hessian ...")
            self._hessian_jit = jax_jit(jax_hessian(self.fn))
            if self.verbose >= 2:
                print("finish jitting hessian ...")

            # test the time for jit as its first call
            if self.verbose >= 2:
                print("testing jitted calls ...")
            self.fn(self.lb)
            self._grad_jit(self.lb)
            self._hessian_jit(self.lb)
            if self.verbose >= 2:
                print("finish testing jitted calls ...")
            end = time.time()
            self.time_jit = end - start
        except Exception as e:
            print(f"Failed to jit the function as: {str(e)}")

        self.set_parameters(**kwargs)

        return

    def set_parameters(self, **kwargs):
        for key, val in kwargs.items():
            if getattr(self, key, "Not Exist") != "Not Exist":
                setattr(self, key, val)
        return

    def optimize(self, restart=False, **kwargs):
        self.set_parameters(**kwargs)

        if self.n_opt_local > 0:
            self._init_local_opt(self.n_opt_local)

        if not restart:
            if self.root is not None:
                np.random.seed(self.seed)

            self.root = None
            if self.if_save_history:
                self._history = self._cache_expand()
                self.history_current_idx = -1
            if self.if_save_node_best:
                self._node_best = self._cache_expand()
                self.node_best_current_idx = -1

            self.node_interval = defaultdict(list)
            self.node_interval_lb_from_child = defaultdict(lambda: None)
            self.node_bounds = defaultdict(list)
            self.node_cover_children = defaultdict(list)
            self.node_box_size = defaultdict(
                lambda: self._bound_volume(self.lb, self.ub)
            )

        if self.root is None:
            if self.verbose >= 1:
                print("Creating root node ...")
            self.create_root(self.root_anchor)
            if self.verbose >= 1:
                print(
                    f"  Best found value until iteration {0} is: {self.root.y:.4f} \n"
                )

        for tt in range(1, self.max_iterations + 1):
            if self.verbose >= 1:
                print(f"Running iteration: {tt} ...\n")

            # selection and expansion
            node = self.select()
            if self.verbose >= 2:
                print(f"  Selected node: {node.name}")

            # new samples, and new guess within the node
            y_best, x_best = self.node_exploitation(node)
            if self.verbose >= 2:
                print(f"  Best found point in node {node.name} is: {y_best:.4f}")
            if self.verbose >= 3:
                print(f"  Best found point in node {node.name} is: ", x_best)

            # backpropagation is done in above step
            # # back propagation
            # self.backprop(node, x_best, y_best)

            # Information
            if self.verbose >= 1:
                print(
                    f"  Best found value until iteration {tt} is: {self.root.y:.4f} \n"
                )

        return self.root.y

    def create_root(self, *args, **kwargs):
        # create root node by anchor
        self.root = self.create_empty_node()
        self.root.is_root = True
        self.root.set_level()
        self.root.name = "root"
        self.assign_box(self.root, self.lb, self.ub)

        # explore the root node
        self.node_exploitation(self.root)
        return

    def select(self, node=None):
        if node is None:
            node = self.root

        while node.children:
            # node = self.select_by_uct_interval(node)
            node = self.select_by_uct_lb_boxsize(node)

        return node

    def select_by_uct_lb_boxsize(self, node):
        ucts = []
        for child in node.children:
            fn_interv = self.node_interval[child.name]
            lb_box_from = self.node_interval_lb_from_child[child.name]
            lb_box_size = self.node_box_size[lb_box_from.name]

            # # Use original value
            # term1 = child.y
            # term2 = fn_interv[0]
            # term3 = lb_box_size

            # # Use normalized value
            sample_score = self._interval_max_score * (-self.normalize_value(child.y))
            interv_score = self._interval_max_score * (
                -self.normalize_value(fn_interv[0])
            )
            volume_score = self._interval_max_score * (
                1 - self.normalize_volume(lb_box_size)
            )
            exploration_score = np.sqrt(np.log(node.visit) / (1 + child.visit))

            uct = (
                sample_score
                + self.node_uct_lb_coeff * interv_score
                + self.node_uct_box_coeff * volume_score
                + self.node_uct_explore * exploration_score
            )

            if self.verbose >= 2:
                print(
                    f"  Node {child.name} has uct: {uct:.4f} =  ({sample_score:.4f}) + {self.node_uct_lb_coeff} * ({interv_score:.4f}) + {self.node_uct_box_coeff} * ({volume_score:.4f}) + {self.node_uct_explore} * ({exploration_score:.4f})"
                )
            ucts.append(uct)

        best = np.argmax(ucts)
        if self.verbose >= 2:
            print(f"  Node {node.children[best].name} is selected")
        return node.children[best]

    def split_box_for_children(self, node):
        lb, ub = self.node_bounds[node.name]
        if self.verbose >= 3:
            print(f"Box from node : {node.name} lb = {lb}, ub = {ub}")

        # split the box to children
        if node.children:
            # find x of valid children
            xs = []
            self.node_cover_children[node.name].clear()
            cover_set = self.node_cover_children[node.name]

            for ii in range(len(node.children)):
                child = node.children[ii]
                _x = child.anchor

                # # Option 2: pick only samples within the box
                if self.point_in_bound(_x, lb, ub):
                    xs.append(_x)
                    cover_set.append(child)

            xs = np.array(xs)

            # split the box by a dim
            # split_dim = np.random.randint(0, self.dims)
            split_dim = self.longest_dim_in_bound(lb, ub)
            if self.verbose >= 3:
                print("split_dim:", split_dim)
            boxes = partition_space(lb, ub, xs, split_dim)

            if self.verbose >= 3:
                print("splitted box:", boxes)

            # update the bounds for valid children
            for ii, child in enumerate(cover_set):
                _lb, _ub = boxes[ii]
                self.assign_box(child, _lb, _ub)
                self.evaluate_node_function_interval(child)
                if self.verbose >= 2:
                    print(
                        f"assign {child.name} fn_interv:",
                        self.node_interval[child.name],
                    )
                if self.verbose >= 3:
                    print(f"assign {child.name} lb|ub:", _lb, _ub)

            # update the function interval for valid children
            self.backprop_interval_from_leaf(cover_set[0])
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
        node = RealVectorTreeNode()
        return node

    def advance_sample(self, num_samples, lb=None, ub=None):
        # Get num_samples samples within the bounds
        # Use more evenly distributed sampler
        if lb is None:
            lb = self.lb
        if ub is None:
            ub = self.ub
        samples = self.advanced_sampler.random(n=num_samples)
        samples = lb + (ub - lb) * samples
        return samples

    def uniform_sample(self, lb=None, ub=None):
        # Uniformly sample a point within the bounds
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
        return box

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
        try:
            return np.asarray(self._grad_jit(x)).astype(x.dtype)
        except:
            print(" Failed to use jit grad, use grad instead")
            return np.asarray(jax_grad(self.fn)(x)).astype(x.dtype)

    def _hessian(self, x):
        try:
            return np.asarray(self._hessian_jit(x)).astype(x.dtype)
        except:
            print(" Failed to use jit hessian, use hessian instead")
            return np.asarray(jax_hessian(self.fn)(x)).astype(x.dtype)

    def _newton_step(self, x, gradinet, hessian_diagonal):
        x_new = x - gradinet / hessian_diagonal
        return x_new

    def _gd_step(self, x, gradient, step_size=0.1):
        x_new = x - gradient * step_size
        return x_new

    def _function_interval_on_box(self, box):
        # Compute the function value interval
        # box: IntervalVector or list of Interval pairs
        # return: function value Interval
        if not isinstance(box, IntervalVector):
            box = IntervalVector(box)
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

        mask_convex = hessian > 0
        x_new[mask_convex] = x_new_nt[mask_convex]
        x_new[~mask_convex] = x_new_gd[~mask_convex]

        # Check if the new point is valid
        mask_invalid = np.isfinite(x_new)
        x_new[mask_invalid] = x[mask_invalid]

        return x_new

    def guess_good(self, node):
        # Compute the expected gradient and hessian(diagonal) from children
        fprime_child, fhess_child = self.expect_grad_hessian(node.children)

        # Compute local gradient and hessian(diagonal) from sampling
        neighbor_samples = 20
        lb_parent, ub_parent = self.node_bounds[node.name]
        dx_ratio = 0.1
        x = copy.deepcopy(node.X)
        dx = (ub_parent - lb_parent) * dx_ratio
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

    def node_exploitation(self, node):
        # Exploit the current node for better function value:
        # 1. Create num_node_expand nodes at the next level by random sampling
        # 2. Guess a good point from global Hessian and neighbor gradient

        # Step 1: Create num_node_expand nodes at the next level by random sampling

        # # Randomly sample num_node_expand nodes
        # for ii in range(self.num_node_expand):
        #     anchor = self.uniform_sample(*self.node_bounds[node.name])
        #     # Put nodes into the child level
        #     child = self.create_node(anchor)
        #     child.set_parent(node)
        #     child.name = node_name + "_" + f"{len(node.children)}"
        #     X_best = child.X
        #     y_best = child.y
        #     self.backprop(child, X_best, y_best)

        # # Advanced sampling approach to sample num_node_expand nodes
        lb, ub = self.node_bounds[node.name]
        anchors_cover_set = self.advance_sample(self.num_node_expand, lb=lb, ub=ub)
        for anchor in anchors_cover_set:
            # Put nodes into the child level
            child = self.create_node(anchor)
            child.set_parent(node)
            child.name = node.name + "_" + f"{len(node.children)}"

            # Local optimization
            if self.local_optimizer is not None:
                self.node_local_optimize(child)

            self.backprop(child)
            if self.verbose >= 3:
                print(f"Advanced sampling for parent {node.name} at :", child.X)
        # Assign boxes; evaluate function intervals and find box size with lb
        self.split_box_for_children(node)

        if self.verbose >= 2:
            print(
                f"After creating child nodes for {node.name}, best point value: {node.y:.4f}  "
            )
        if self.verbose >= 3:
            print(f"Best point is at : ", node.X)

        # Step 2. Guess a good point from global Hessian and neighbor gradient

        # Create a new sample by guessing a good point by neighbor fprime and global fhess
        # Put it under the parent = root
        # using newton's method for convex functions
        # using gradient descent for concave functions
        anchor = self.guess_good(node)
        anchor = self._clip_point(anchor)

        parent = self.root
        child = self.create_node(anchor)
        child.set_parent(parent)
        child.name = parent.name + "_" + f"{len(parent.children)}"

        # Local optimization
        if self.local_optimizer is not None:
            self.node_local_optimize(child)
        self.backprop(child)

        # Assign box for the new node
        lb, ub = self.node_bounds[node.name]
        dist = 0.5 * (ub - lb)
        lb = anchor - dist
        ub = anchor + dist
        lb = np.clip(lb, self.lb, self.ub)
        ub = np.clip(ub, self.lb, self.ub)
        self.assign_box(child, lb, ub)
        # Evaluate function interval
        self.evaluate_node_function_interval(child)

        if self.verbose >= 2:
            print(
                f"New node {child.name} by optimizing with global Hessian and neighbor gradient: {child.y:.4f} "
            )
        if self.verbose >= 3:
            print(f"New node {child.name} is at : ", child.X)

        return node.y, node.X

    def _clip_point(self, x, lb=None, ub=None, clip_periodic=None):
        if lb is None:
            lb = self.lb
        if ub is None:
            ub = self.ub
        if clip_periodic is None:
            clip_periodic = self.clip_periodic
        if clip_periodic:
            box = ub - lb
            for ii in range(self.dims):
                while x[ii] > ub[ii]:
                    x[ii] -= box[ii]
                while x[ii] < lb[ii]:
                    x[ii] += box[ii]
        else:
            x = np.clip(x, lb, ub)
        return x

    def assign_box(self, node, lb, ub):
        if node.name not in self.node_bounds:
            self.node_bounds[node.name] = (lb, ub)
            self.node_box_size[node.name] = self._bound_volume(lb, ub)
        else:
            print(f"Warining: re-assigning box for node {node.name}; action not taken")

    def normalize_value(self, value):
        if self._interval_global is None:
            _box = self._list_to_box(self.lb, self.ub)
            self._interval_global = self._function_interval_on_box(_box)

        if self._interval_global[0] == self._interval_global[1]:
            return 0
        else:
            # default range is [0, 1]
            return (value - self._interval_global[0]) / (
                self._interval_global[1] - self._interval_global[0]
            )

    def _bounds_union(self, lbs, ubs):
        # Union of intervals
        # lbs: list of lower bounds
        # ubs: list of upper bounds
        # return: lower bound and upper bound of the union
        lb = np.min(lbs, axis=0)
        ub = np.max(ubs, axis=0)
        return lb, ub

    def _bound_volume(self, lb, ub):
        # return the volumn of the box
        dist = ub - lb
        dist[dist < 1e-6] = 1e-6  # avoid log(0)
        volume = np.sum(np.log(dist))
        return volume

    def normalize_volume(self, vol):
        if self._global_volume is None:
            self._global_volume = self._bound_volume(self.lb, self.ub)

        # default range is [0, 1.0]
        # if the volume ratio is (0.5)^dims, we set the value to 0.5
        # Note the volume is computed by log
        # so the ratio is computed by minus and exp
        ratio = (vol - self._global_volume) / self.dims
        value = np.exp(ratio)

        return value

    def _1dinterval_merge(self, intervals):
        # Union of intervals (or )
        # intervals: list or array of intervals [[lb, ub], ...]
        # return: lower bound and upper bound of the union, and where the lb and ub are in the original list

        # sort the intervals by bound
        lb_sorted = sorted(enumerate(intervals), key=lambda x: x[1][0])
        ub_sorted = sorted(enumerate(intervals), reverse=True, key=lambda x: x[1][1])

        idx_lb, intervals_lb = lb_sorted[0]
        idx_ub, intervals_ub = ub_sorted[0]
        lb = intervals_lb[0]
        ub = intervals_ub[1]
        return [lb, ub], [idx_lb, idx_ub]

    def point_in_bound(self, point, lb, ub):
        return np.all(point >= lb) and np.any(point <= ub)

    def longest_dim_in_bound(self, lb, ub):
        # return the longest dimension of the box
        return np.argmax(ub - lb)

    def evaluate_node_function_interval(self, node):
        # Update the function interval of a node
        # This function is called when the node is created or the node's box is updated
        lb, ub = self.node_bounds[node.name]
        box = self._list_to_box(lb, ub)
        function_interval = self._function_interval_on_box(box)
        self.node_interval[node.name] = function_interval

        # the lb is from itself; its parent may refer to this node as where the lb is from
        self.node_interval_lb_from_child[node.name] = node

        return function_interval

    def update_node_function_interval_from_cover_set(self, node):
        # Update the function interval of this node
        # from the function intervals in its covering set

        # no cover set found
        if not self.node_cover_children:
            return

        # no function intervals for all children in cover set
        for child in self.node_cover_children[node.name]:
            if not self.node_interval[child.name]:
                return

        # all children in cover set have function intervals
        child_intervals = []
        for child in self.node_cover_children[node.name]:
            # [lb(f(child)), ub(f(child))]
            _interval = self.node_interval[child.name]
            child_intervals.append(_interval)

        # Merge intervals, and find where lb/ub comes from
        function_interval, indices = self._1dinterval_merge(child_intervals)
        lb_from_child_idx, _ = indices
        lb_from_child = self.node_cover_children[node.name][lb_from_child_idx]
        self.node_interval_lb_from_child[node.name] = self.node_interval_lb_from_child[
            lb_from_child.name
        ]

        # Update the function interval of this node
        self.node_interval[node.name] = function_interval

        return

    def backprop_interval_from_leaf(self, node):
        if node.children:
            if self.verbose >= 3:
                print(
                    f"Warning: calling backprop interval from non-leaf node {node.name}"
                )
            return

        parent = node.parent
        while parent:
            self.update_node_function_interval_from_cover_set(parent)
            parent = parent.parent
        return

    def _init_local_opt(self, n_opt_local, **kwargs):
        self._box_global = self._list_to_box(self.lb, self.ub)
        self.local_optimizer = scipy_minimize

        self.local_opt_kwargs = {
            "method": "L-BFGS-B",
            "jac": self._grad,
            "options": {"maxfun": n_opt_local, "disp": False},
            "bounds": self._box_global,
        }

    def local_opt_from(self, x0):
        result_x = x0
        result_y = self.get_groundtruth(x0)
        if self.local_optimizer is not None:
            res = self.local_optimizer(
                self.get_groundtruth, x0, **self.local_opt_kwargs
            )
            result_x = res.x
            result_y = res.fun
        return result_y, result_x

    def node_local_optimize(self, node):
        x0 = node.X
        result_y, result_x = self.local_opt_from(x0)
        if self.verbose >= 2:
            print(f" run local optimization on node  {node.name}:")
            print(f"  y0 -> y*: {node.y} -> {result_y}")
        if self.verbose >= 3:
            print(f"  x0: {x0}")
            print(f"  x*: {result_x}")

        # update the node
        node.update_stat(result_y, result_x)
        node.visit_once()
        return
