import copy
from typing import Any

import numpy as np
import nlopt

import jax.numpy as jnp

# from jax import device_put, grad, jit, random, vmap
from jax import grad as jax_grad

from pyibex import Interval, IntervalVector

from .node import RealVectorTreeNode
from .linear import max_linear_weight
from .expression_operation import *


class MCMM:
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
        node_uct_improve=1.0,  # C_d  for node
        node_uct_n_improve=3,  # number of improves
        node_uct_explore=1.0,  # C_p  for node
        leaf_improve=100.0,  # C_d'' for leaf
        leaf_explore=10.0,  # C_p'' for leaf
        branch_explore=1.0,  # C_p' for branch
        function_variables=None,  # symbolic variables of the function
        function_expression=None,  # symbolic expression of the function
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
            self.n_intial = n_initial

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

        # batch
        self.batch_size = batch_size  # Not used yet

        # local optimization
        self.local_opt_eval_num = n_eval_local
        self.local_opt_eval_time = 10
        self.local_opt_algo = nlopt.LD_CCSAQ
        self.local_opt_ftol_rel = 1e-3
        self.local_opt_ftol_abs = 1e-3
        self.local_opt_xtol_rel = 1e-6
        self.local_opt_xtol_abs = 1e-6
        self.local_opt = None

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
        self.found_empty_box = False
        self.lb_expression = None
        self.lb_variables = None
        self.lb_ignore_min_dist = (
            1e-4  # ignore points with distance smaller than this value
        )
        self.lb_lower_by = 0.1  # lower the lower bound by this value
        self.lb_ignore_coeff = (
            1e-4  # ignore points with coefficient smaller than this value
        )

        # Update other parameters
        self._box_init = None
        self.set_parameters(**kwargs)

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
            self.create_root(self.root_anchor)

        tt = 0  # iteration counter
        while tt < self.max_iterations:
            if self.verbose >= 1:
                print(f"Running iteration: {tt+1} ...\n")

            # selection and expansion
            node = self.select()
            # when no node is selected
            # continue to next iteration
            if node is None:
                if self.verbose >= 2:
                    print(f"  No node is suggested. Rodo iteration {tt+1}.\n")
                continue

            # local optimization
            y_best, x_best = self.optimize_node(node)
            # save the best value to node best

            # back propagation
            self.backprop(node, x_best, y_best)

            # Information
            if self.verbose >= 1:
                print(f"  Best found value until iteration {tt+1} is: {self.root.y} \n")

            # next iteration
            tt += 1
        return self.root.y

    def set_parameters(self, **kwargs):
        for key, val in kwargs.items():
            if getattr(self, key, "Not Exist") != "Not Exist":
                setattr(self, key, val)
        return

    def select(self):
        # if the last 5 improves are less than 1e-5,
        # then try to find a box
        if len(self.root.improves) <= 5:
            return self.mctd_select()

        if np.sum(self.root.improves[-5:]) > 1e-5:
            return self.mctd_select()

        return self.mctd_select()

    def mctd_select(self, node=None):
        if node is None:
            node = self.root

        while node.children:
            if self.verbose >= 2:
                print(f"  Selecting best child for node : {node.identifier}")

            # choose best child by uct
            ucts = self.get_ucts(node)
            if node.visit > self.min_node_visit:  # at least visit by some times
                uct_explore = self.get_uct_explore(node)  # check branch exploration
                if self.verbose >= 2:
                    print(f"    Checking explore term: {uct_explore}")
                if uct_explore > max(ucts):
                    if self.verbose >= 2:
                        print(
                            f"  Selected: new exploration node is created at branch {node.identifier}"
                        )
                    return self.expand(node)  # expand at branch
            idx = np.argmax(ucts)
            node = node.children[idx]

            if self.verbose >= 2:
                print(f"  Selected: existing child node {node.identifier}")

        # # check leaf exploration
        # if node.visit > self.min_node_visit:
        #     visit = max(node.visit, 1)
        #     leaf_explore_thres = self.leaf_explore * np.sqrt(np.log(visit))
        #     leaf_explore_score = -node.y + self.leaf_improve * np.sum(
        #         node.improves[-self.node_uct_n_improve :]
        #     )
        #     if leaf_explore_score < leaf_explore_thres:
        #         return self.expand(node)
        if node.force_split:
            return self.expand(node)

        return node

    def expand(self, node, method="suggest"):
        parent_identifier = node.identifier

        # if it is a leaf node, use node.X/y as anchor for a copied child node at index 0
        if not node.children:
            copynode = self.create_node(node.X, node.y)
            copynode.set_parent(node)
            identifier = parent_identifier + "_0"
            copynode.identifier = identifier
            if node.force_split:
                copynode.set_force_split()
            copynode.visit = node.visit

        # create a new child node at index N

        if method == "random":
            child = self._expand_in_ring(node)

        elif method == "suggest":
            # Try to expand by suggest in box.
            # If no suggestion, it means the box is empty.
            # Then the node is the best around its neightbor already.
            # Never split it anymore and return None
            child = self._expand_by_suggest(node)
            if child is None:
                node.set_terminal()
                return None

        if self.verbose >= 2:
            print(f"  Expanded: new child node {child.identifier} is created.")
        return child

    def _expand_in_ring(self, node):
        # create a new child at the end of the children list on node
        # by placing an anchor point with the distance D from node.X
        # D is within [-dist_max, -dist_min] U [dist_min, dist_max]
        # sampled from a uniform distribution
        # x_orig = copy.deepcopy(node.X)
        x_orig = copy.deepcopy(node.anchor)

        dist_box = self.ub - self.lb
        dist_min = self.dist_min * (self.dist_decay**node.level) * dist_box
        dist_max = self.dist_max * (self.dist_decay**node.level) * dist_box

        x_anchor = self._sample_in_ring(x_orig, dist_min, dist_max)

        child = self.create_node(x_anchor)
        child.identifier = node.identifier + "_" + str(len(node.children))
        child.set_parent(node)
        return child

    def _expand_by_suggest(self, node):
        # create a new child at the end of the children list on node
        # by placing an anchor point in the box
        # where the lb_expression and fn_expression can match
        # x_orig = copy.deepcopy(node.X)
        x_orig = copy.deepcopy(node.anchor)

        # get the box near the node
        dist_box = self.ub - self.lb
        dist = (
            (self.dist_min + self.dist_max)
            / 2
            * (self.dist_decay**node.level)
            * dist_box
        )

        _lb = x_orig - dist
        _ub = x_orig + dist

        _lb = self._clip_point(_lb, clip_periodic=False)
        _ub = self._clip_point(_ub, clip_periodic=False)

        init_box = self._list_to_box(_lb, _ub)

        # find a box
        suggest_box = self.suggest_box(init_box)
        if self.verbose >= 2:
            print(f"    Initial box: {init_box}")
            print(f"    Suggested box: {suggest_box}")
        exist_empty_box = any([Interval.is_empty(intv) for intv in suggest_box])
        if exist_empty_box:
            return None

        # create a new point within the box
        x_anchor = self._sample_in_box(suggest_box)
        child = self.create_node(x_anchor)
        child.identifier = node.identifier + "_" + str(len(node.children))
        child.set_parent(node)
        return child

    def optimize_node(self, node):
        # local optimization
        x = node.X
        y_best, x_best, opt_status = self._local_opt(x)

        if opt_status < 0:
            # local optimization failed, use current best value
            print(
                f"  Local optimization failed on node {node.identifier}, using current best value. \n"
            )
            node.force_split = True
        elif opt_status > 1:
            # node is forcd to split because local optimization is stalled
            node.force_split = True
        return y_best, x_best

    def _local_opt(self, x):
        if self.local_opt is None:
            self._local_opt_init()

        assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)

        x_best = self.local_opt.optimize(x)
        y_best = self.local_opt.last_optimum_value()
        opt_status = self.local_opt.last_optimize_result()

        return y_best, x_best, opt_status

    def _nlopt_fn(self, x, grad):
        x = jnp.clip(x, self.lb, self.ub)
        y = self.get_groundtruth(x)

        if grad.size > 0:
            grad[:] = jax_grad(self.fn)(x)

        return y

    def backprop(self, node, x_best=None, y_best=None):
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

        while curr_node:
            curr_node.update_stat(y_best, x_best)
            curr_node.visit_once()
            curr_node = curr_node.parent
        return

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

    def _cache_expand(self, cache_array=None):
        if cache_array is None:
            cache_array = np.zeros((self._cache_expand_size, 1 + self.dims))
        else:
            cache_array = np.concatenate(
                (cache_array, np.zeros((self._cache_expand_size, 1 + self.dims))),
                axis=0,
            )
        return cache_array

    def get_ucts(self, node):
        ucts = []
        visit_parent = max(node.visit, 1)
        for child in node.children:
            # ternimal node never be selected
            if child.terminal:
                ucts.append(-np.inf)
                continue
            uct = child.score
            visit = max(child.visit, 1)
            uct += np.sqrt(np.log(visit_parent) / visit) * self.node_uct_explore
            ucts.append(uct)

            if self.verbose >= 2:
                print(
                    f"    Child {child.identifier}: score = {child.score}, num_visit = {child.visit}",
                )
        return ucts

    def get_uct_explore(self, node):
        uct_explore = 0
        visit_parent = max(node.visit, 1)
        if node.children:
            for ii in range(len(node.children)):
                child = node.children[ii]
                uct_explore += -child.y
            uct_explore /= len(node.children)
        uct_explore += np.sqrt(np.log(visit_parent)) * self.branch_explore
        return uct_explore

    def create_root(self, anchor=None):
        # create root node by anchor
        self.root = self.create_node(anchor)
        self.root.is_root = True
        self.root.set_level()
        self.root.identifier = "root"

        y_best, X_best = self.optimize_node(self.root)
        self.backprop(self.root, X_best, y_best)

        # Multiple starting nodes: put all nodes into the first level
        for ii in range(self.n_initial - 1):
            child = self.expand(self.root, method="random")
            y_best, X_best = self.optimize_node(child)
            self.backprop(child, X_best, y_best)
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

    def _local_opt_init(self):
        self.local_opt = nlopt.opt(self.local_opt_algo, self.dims)
        self.local_opt.set_min_objective(self._nlopt_fn)
        self.local_opt.set_lower_bounds(self.lb)
        self.local_opt.set_upper_bounds(self.ub)

        self.local_opt.set_maxeval(self.local_opt_eval_num)
        self.local_opt.set_maxtime(self.local_opt_eval_time)

        self.local_opt.set_ftol_rel(self.local_opt_ftol_rel)
        self.local_opt.set_ftol_abs(self.local_opt_ftol_abs)
        self.local_opt.set_xtol_rel(self.local_opt_xtol_rel)
        self.local_opt.set_xtol_abs(self.local_opt_xtol_abs)

        return

    def print(self):
        nodes = [self.root]

        print("Display entire tree info : node, best_y, num_visits, improves")
        while nodes:
            node = nodes.pop(0)
            print(
                node.identifier,
                node.y,
                node.visit,
                node.improves,
            )
            nodes.extend(node.children)
        print("\n")
        return

    @property
    def history(self):
        return self._history[: self.history_current_idx + 1, :]

    @property
    def node_best(self):
        return self._node_best[: self.node_best_current_idx + 1, :]

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

    def _quadra_lb(self, X, y):
        """
        Compute a quadratic lower bound function for the given data

        Return:
            variables: a list of variables
            expression: a sympy expression
        """

        # Get best point
        idx_best = np.argmin(y)
        y_best = y[idx_best]
        x_best = X[idx_best]

        # Learn a quadratic lower bound function based on X/y:
        # lb = k * (x - x_best)**2 + y_best <= y
        # where k is a non-negative constant
        # The above form is equivalent to:
        # x2 * k <= y1
        # in which x2 = (x - x_best)**2, y1 = y - y_best
        x2 = (X - x_best) ** 2
        y1 = y - y_best
        quadra_coeff = max_linear_weight(x2, y1)

        # Make the coefficient under-approximation
        # to guarantee the lower bound is always below the true function
        quadra_coeff *= 0.5

        # Update the lower bound function
        variables, expression = expression_quadratic(
            x_best, quadra_coeff, y_best, self.lb_ignore_coeff
        )

        if self.verbose >= 2:
            print(" Quadra LB center:", x_best)
            print(" Quadra LB coefficients:", quadra_coeff)
        return variables, expression

    def _distance(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        x12 = np.linalg.norm(x1, axis=1).reshape(-1, 1)
        x12 = x12**2

        x22 = np.linalg.norm(x2, axis=1).reshape(1, -1)
        x22 = x22**2

        dist = x12 + x22 - 2 * np.dot(x1, x2.T)

        return dist

    def node_best_lower_bound(self):
        # ignore min point neighbor
        if self.lb_ignore_min_dist > 0:
            mask = self._mask_min_neighbor(self.node_best, self.lb_ignore_min_dist)
        else:
            mask = np.ones(self.node_best.shape[0], dtype=bool)

        # Get history
        X = self.node_best[mask, 1:]
        y = self.node_best[mask, 0]

        # Get lower bound
        if self.verbose >= 2:
            print("  Finding lower bound function...")
        self.lb_variables, self.lb_expression = self._quadra_lb(X, y)

        # displacement for the lower bound
        if self.lb_lower_by != 0:
            self.lb_expression = expression_minus(
                self.lb_expression, f"({self.lb_lower_by})"
            )

        if self.verbose:
            print("  Lower bound function:", self.lb_expression)
        return

    def history_lower_bound(self):
        # ignore min point neighbor
        if self.lb_ignore_min_dist > 0:
            mask = self._mask_min_neighbor(self.history, self.lb_ignore_min_dist)
        else:
            mask = np.ones(self.history.shape[0], dtype=bool)

        # Get history
        X = self.history[mask, 1:]
        y = self.history[mask, 0]

        # Get lower bound
        if self.verbose >= 2:
            print("  Finding lower bound function...")
        self.lb_variables, self.lb_expression = self._quadra_lb(X, y)

        # displacement for the lower bound
        if self.lb_lower_by != 0:
            self.lb_expression = expression_minus(
                self.lb_expression, f"({self.lb_lower_by})"
            )

        if self.verbose:
            print("  Lower bound function:", self.lb_expression)
        return

    def _mask_min_neighbor(self, y_X, threshold):
        """
        Mask the points that are the close to the min point within distance**2 = dist.
        """

        assert isinstance(y_X, np.ndarray) or isinstance(y_X, jnp.ndarray)
        assert y_X.ndim == 2

        mask = np.ones(y_X.shape[0], dtype=bool)
        idx = np.argmin(y_X[:, 0])

        # Distance is computed by X^2 - 2 * X * Y.T + Y^2
        # We check only single point, so Y^2 is a constant
        distance1 = np.linalg.norm(y_X[idx, 1:]) ** 2
        distance2 = -2 * np.dot(y_X[:, 1:], y_X[idx, 1:].T)
        distance = distance1 + distance2

        mask[distance < threshold] = False
        mask[idx] = True  # always keep the min point

        return mask

    def suggest_box(self, init_box=None, lb_from="global" or "node"):
        """
        Suggest a box that is likely to have a point where the current lb function equals the objective function.
        """
        if self.function_expression is None:
            print("No function expression is provided. Cannot suggest box.")
            return None

        # Get the lower bound function
        if lb_from == "global":
            self.history_lower_bound()
        elif lb_from == "node":
            self.node_best_lower_bound()
        else:
            print(f"Unknown lb_from: {lb_from}")
            return None

        # Get the equality relationship
        assert self.lb_variables == self.function_variables
        expression_equality = expression_minus(
            self.function_expression, self.lb_expression
        )

        # Initial box is the entire space if not provided
        if init_box is None:
            if self._box_init is None:
                intervals = []
                for i in range(len(self.lb)):
                    _int = [self.lb[i], self.ub[i]]
                    intervals.append(_int)
                self._box_init = IntervalVector(intervals)
            init_box = self._box_init

        # Find a box
        suggest_box = root_box(
            self.lb_variables,
            expression_equality,
            init_box,
        )

        return suggest_box

    def _clip_point(self, x, clip_periodic=None):
        if clip_periodic is None:
            clip_periodic = self.clip_periodic
        if clip_periodic:
            box = self.ub - self.lb
            for ii in range(self.dims):
                while x[ii] > self.ub[ii]:
                    x[ii] -= box[ii]
                while x[ii] < self.lb[ii]:
                    x[ii] += box[ii]
        else:
            x = np.clip(x, self.lb, self.ub)
        return x

    def _sample_in_ring(self, x_center, dist_min, dist_max):
        # dist_min/max: either a non-negative float
        # or a list of non-negative floats or a numpy array
        # x_center: a numpy array
        # return: a sample as a numpy array

        # create a random ratio in (0, 1)
        # project (0.5, 1) to (dist_min, dist_max)
        # and (0, 0.5) to (-dist_max, -dist_min)
        ratio = self._cache_rand()
        ratio_mask = ratio < 0.5
        ratio = ratio * 2 - 1

        dist_min_plus_neg = np.ones(self.dims)
        dist_min_plus_neg[ratio_mask] = -1

        dist = dist_min * dist_min_plus_neg + (dist_max - dist_min) * ratio
        x_return = x_center + dist
        x_return = self._clip_point(x_return)
        return x_return

    def _sample_in_box(self, box):
        # box is an interval vector with length = dims
        # If there is empty interval, return None

        assert isinstance(box, IntervalVector)
        # check if the box contains empty
        contains_empty = any(Interval.is_empty(inv) for inv in box)
        if contains_empty:
            return None

        # create a new point with the box
        _lb, _ub = self._box_to_list(box)
        _lb = np.array(_lb)
        _ub = np.array(_ub)
        x_new = self.uniform_sample(_lb, _ub)
        x_new = self._clip_point(x_new)
        return x_new
