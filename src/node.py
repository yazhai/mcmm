import copy
import numpy as np


class SearchNodeABS:
    """
    This class defines an abstracted node in the search model.

    obj = SearchNodeABS(
                name = None,
                parent = None,
                children = None,
                possible_moves = None,
                is_root = False,
                )
    """

    def __init__(
        self,
        name=None,
        parent=None,
        children=None,
        possible_moves=None,
        is_root=False,
    ):
        self.name = name

        self.is_root = is_root
        self.parent = None
        self.set_parent(parent)
        self.level = -1
        if children is None:
            self.children = []
        else:
            assert isinstance(children, list)
            self.children = children

        self.possible_moves = []
        self.add_moves(possible_moves)

    def set_parent(self, parent):
        self.parent = parent
        if (self.parent is not None) and (self not in self.parent.children):
            self.parent.children.append(self)
        self.set_level()

    def set_level(
        self,
    ):
        if self.is_root:
            self.level = 0
        else:
            try:
                self.level = self.parent.level + 1
            except:
                self.level = -1
        return

    def set_parameters(self, **kwargs):
        for key, val in kwargs.items():
            if getattr(self, key, "Not Exist") != "Not Exist":
                setattr(self, key, val)
        return

    def add_moves(self, additional_possible_moves):
        if isinstance(additional_possible_moves, list):
            self.possible_moves = [*self.possible_moves, *additional_possible_moves]
        return

    def expand_moves(
        self,
    ):
        for mov in self.possible_moves:
            child = self.make_child(mov)
            self.child.append(child)
        return


class RealVectorTreeNode(SearchNodeABS):
    """
    Class: TreeNode with real vectors.


    obj = RealVectorTreeNode(
                input=None,
                name=None,
                is_root=False,
                children=None,
                parent=None,
                improve_factor=0.0,
                improve_count=3,
                )

    Properties:
        X: best sample coordinate in R^d
        y: best sample value in real
        saved_y_X: all saved samples
        score: score of this node
        level: level of the node in the tree
        visit: number of visits on this node

    Methods:
        add_samples(sample_vector): add samples in vector of nx[y, X_0, X_1, ...] to saved_y_X
        update_stat(y, X): update best found X and y
        visit_once()

    """

    def __init__(
        self,
        input=None,
        name=None,
        is_root=False,
        children=None,
        parent=None,
        improve_factor=0.0,
        improve_count=3,
        **kwargs,
    ):
        super().__init__(
            name=name,
            parent=parent,
            children=children,
            is_root=is_root,
        )

        self._X = None
        self._y = None  # current stats, best found (x, y) on the node
        self.saved_y_X = None
        self.add_samples(input)  # update the saved_y_X array

        # score related params
        self.visit = 0
        self.improve_factor = improve_factor
        self.improve_count = improve_count
        self.improves = []

        # force_split is a flag to force split the node
        self.force_split = False

        # terminal is a flag to indicate whether the node should not
        # be expanded anymore
        self.terminal = False

        self.set_parameters(**kwargs)

        return

    def __hash__(self) -> int:
        if self.name is None:
            return hash(0)
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, val):
        assert isinstance(val, np.ndarray)
        self._X = val

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = val

    @property
    def anchor(self):
        # use first node as anchor
        try:
            return self.saved_y_X[0, 1:]
        except:
            return None

    @property
    def score(self):
        score = 0
        try:
            score += -self.y
            score += np.sum(self.improves[-self.improve_count :]) * self.improve_factor
        except:
            pass

        # print(
        #     f"## Temporary: computing score on node {self.name} : best_y = {self.y} , improves = ",
        #     self.improves,
        # )

        return score

    def add_samples(self, sample_vector=None):
        if sample_vector is not None:
            assert isinstance(sample_vector, np.ndarray)
            try:
                self.saved_y_X = np.vstack((self.saved_y_X, sample_vector))
            except:
                self.saved_y_X = copy.deepcopy(sample_vector)
                if self.saved_y_X.ndim == 1:
                    self.saved_y_X = self.saved_y_X.reshape([1, -1])
        self.update_stat()
        return

    def update_stat(self, y=None, X=None):
        # if passing nothing, update the best saved value
        if (y is None) and (X is None):
            if self.saved_y_X is not None:
                idx_best = np.argmin(self.saved_y_X[:, 0])
                if (self.y is None) or (self.visit == 0):
                    self.y = self.saved_y_X[idx_best, 0]
                    self.X = self.saved_y_X[idx_best, 1:]
                else:
                    dy = self.y - self.saved_y_X[idx_best, 0]
                    self.y = self.saved_y_X[idx_best, 0]
                    self.X = self.saved_y_X[idx_best, 1:]
                    if dy > 0:
                        self.improves.append(dy)
                    else:
                        self.improves.append(0.0)
            return

        # if passing X and y, update the best saved value
        if y is not None:
            if (self.y is None) or (self.visit == 0):
                self.y = y
                self.X = copy.deepcopy(X)
            else:
                if self.y > y:
                    dy = self.y - y
                    self.y = y
                    self.X = copy.deepcopy(X)
                    self.improves.append(dy)
                else:
                    self.improves.append(0.0)
        return

    def visit_once(self):
        self.visit += 1

    def reset_visit(self):
        self.visit = 0

    def set_force_split(self):
        self.force_split = True
        return

    def set_terminal(self):
        self.terminal = True
        return

    def __hash__(self) -> int:
        if self.name is None:
            return hash(0)
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name
