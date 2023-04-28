import copy
from typing import Any

import numpy as np
import nlopt

import jax.numpy as jnp

# from jax import device_put, grad, jit, random, vmap
from jax import grad as jax_grad

from src.node import RealVectorTreeNode


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
        dist_max=0.75,  # Max relative distance when create new node
        dist_min=0.25,  # Min relative distance when create new node
        dist_decay=0.5,  # decay factor of distance in every level
        dist_periodic=True,  # use periodic boundary when create new node
        node_uct_improve=1.0,  # C_d  for node
        node_uct_n_improve=3,  # number of improves
        node_uct_explore=1.0,  # C_p  for node
        leaf_improve=100.0,  # C_d'' for leaf
        leaf_explore=10.0,  # C_p'' for leaf
        branch_explore=1.0,  # C_p' for branch
        **kwargs,
    ):
        self.fn = fn
        self.lb = lb
        self.ub = ub
        self.log = log
        self.dims = len(lb)

        self.root = None
        self.verbose = verbose

        self.if_save_history = True
        self.history_cache = 1000
        self._history = np.zeros((self.history_cache, 1 + self.dims))
        self.history_current_idx = -1

        # iterations
        self.max_iterations = max_iterations

        # batch
        self.batch_size = batch_size  # Not used yet

        # local optimization
        self.local_opt_eval_num = n_eval_local
        self.local_opt_eval_time = 10
        self.local_opt_algo = nlopt.LD_CCSAQ
        self.local_opt_ftol_rel = 1e-6
        self.local_opt_ftol_abs = 1e-6
        self.local_opt_xtol_rel = 1e-6
        self.local_opt_xtol_abs = 1e-6
        self.local_opt = None

        # new node position
        self.dist_max = dist_max  # Max relative distance when create new node
        self.dist_min = dist_min  # Min relative distance when create new node
        self.dist_decay = dist_decay  # decay factor of distance in every level
        self.dist_periodic = dist_periodic  # use periodic boundary when create new node

        # exploration related
        self.node_uct_improve = node_uct_improve  # C_d  for node
        self.node_uct_n_improve = node_uct_n_improve  # number of improves
        self.node_uct_explore = node_uct_explore  # C_p  for node
        self.leaf_improve = leaf_improve  # C_d'' for leaf
        self.leaf_explore = leaf_explore  # C_p'' for leaf
        self.branch_explore = branch_explore  # C_d' for branch
        self.min_node_visit = 1  # minimum number of visits before expand

        self.set_parameters(**kwargs)

        return

    def optimize(self, **kwargs):
        self.set_parameters(**kwargs)
        self.create_root()

        for tt in range(self.max_iterations):
            if self.verbose >= 1:
                print(f"Running iteration: {tt+1} ...\n")

            # selection and expansion
            node = self.select()

            # local optimization
            y_best, x_best = self.optimize_node(node)

            # back propagation
            self.backprop(node, x_best, y_best)

            # Information
            if self.verbose >= 1:
                print(f"  Best found value until iteration {tt+1} is: {self.root.y} \n")
            if self.verbose >= 3:
                self.print()
        return self.root.y

    def set_parameters(self, **kwargs):
        for key, val in kwargs.items():
            if getattr(self, key, "Not Exist") != "Not Exist":
                setattr(self, key, val)
        return

    def select(self, node=None):
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

    def expand(self, node):
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
        # by placing an anchor point with the distance D from node.X
        # D is within [-dist_max, -dist_min] U [dist_min, dist_max]
        # sampled from a uniform distribution
        x_anchor = copy.deepcopy(node.X)

        dist_box = self.ub - self.lb
        dist_min = self.dist_min * (self.dist_decay**node.level) * dist_box
        dist_max = self.dist_max * (self.dist_decay**node.level) * dist_box

        # create a random ratio in (0, 1)
        # project (0.5, 1) to (dist_min, dist_max)
        # and (0, 0.5) to (-dist_max, -dist_min)
        ratio = self._cache_rand()
        ratio_mask = ratio < 0.5
        ratio = ratio * 2 - 1

        dist_min_plus_neg = np.ones(self.dims)
        dist_min_plus_neg[ratio_mask] = -1

        dist = dist_min * dist_min_plus_neg + (dist_max - dist_min) * ratio
        x_anchor += dist

        if self.dist_periodic:
            box = self.ub - self.lb
            for ii in range(x_anchor.shape[0]):
                while x_anchor[ii] > self.ub[ii]:
                    x_anchor[ii] -= box[ii]
                while x_anchor[ii] < self.lb[ii]:
                    x_anchor[ii] += box[ii]
        else:
            x_anchor = np.clip(x_anchor, self.lb, self.ub)

        child = self.create_node(x_anchor)
        child.identifier = parent_identifier + "_" + str(len(node.children))
        child.set_parent(node)

        if self.verbose >= 2:
            print(f"  Expanded: new child node {child.identifier} is created. \n")

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
        self.history_current_idx += 1

        if self.history_current_idx >= self._history.shape[0]:
            self._history = np.concatenate(
                (self._history, np.zeros((self.history_cache, 1 + self.dims))), axis=0
            )

        self._history[self.history_current_idx, 0] = y
        self._history[self.history_current_idx, 1:] = x
        return

    def get_ucts(self, node):
        ucts = []
        visit_parent = max(node.visit, 1)
        for child in node.children:
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
        self.root = self.create_node(anchor)
        self.root.is_root = True
        self.root.set_level()
        self.root.identifier = "root"
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

    def uniform_sample(self):
        ratio = self._cache_rand()
        sample = self.lb + (self.ub - self.lb) * ratio
        return sample

    def _cache_rand(self, cache_size=1000):
        """
        Cache uniform sampling from numpy.random.rand
        Store a pool of random samples as class variable for next call.

        Input:
            cache_size
        Output:
            A random sample
        """
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
