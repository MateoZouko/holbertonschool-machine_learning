#!/usr/bin/env python3
"""
task 4 project decision tree:
Towards the predict function (2) : the update_bounds method
"""
import numpy as np


class Node:
    """ define a node"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.upper = dict()
        self.lower = dict()
        self.rec_node = dict()

    def update_bounds_below(self):
        """This method should recursively compute,
        for each node, two dictionaries"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            # get the parent node of left_child and right_child
            nodeprec = self
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

            # get the left_child
            if child == self.left_child:
                child.lower.update({nodeprec.feature: nodeprec.threshold})

            # get the right_child
            elif child == self.right_child:
                child.upper.update({nodeprec.feature: nodeprec.threshold})

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def get_leaves_below(self):
        """ get leaves of all the tree"""
        if self.is_root:
            global b
            b = []
        if self.left_child.is_leaf:
            b.append(self.left_child)
        if self.right_child.is_leaf:
            b.append(self.right_child)
        self.left_child.get_leaves_below()
        self.right_child.get_leaves_below()
        return b


class Leaf(Node):
    """define a leaf"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth
        self.upper = dict()
        self.lower = dict()

    def __str__(self):
        """print leaf caracteristics"""
        return (f"-> leaf [value={self.value}]")

    def update_bounds_below(self):
        """ pass when arrive at a leaf"""
        pass

    def get_leaves_below(self):
        """ return a leaf"""
        return [self]


class Decision_Tree():
    """define the classifier"""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def update_bounds(self):
        """ get every bounds"""
        self.root.update_bounds_below()

    def get_leaves(self):
        """ indicates all leaves"""
        return self.root.get_leaves_below()
