#!/usr/bin/env python3
"""
task 0
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

    def max_depth_below(self):
        """ computes the depth of a tree"""
        if not self.left_child.is_leaf:
            self.left_child.max_depth_below()

        if not self.right_child.is_leaf:
            self.right_child.max_depth_below()
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """ function computes a: nombuer of node and b: number of leaf"""
        if self.is_root:
            global a
            global b
            a = 1
            b = 0

        if not self.left_child.is_leaf:
            self.left_child.count_nodes_below()
            if not only_leaves:
                a = a + 1
        else:
            b = b + 1

        if not self.right_child.is_leaf:
            self.right_child.count_nodes_below()
            if not only_leaves:
                a = a + 1
        else:
            b = b + 1
        if not only_leaves:
            return b + a
        elif only_leaves:
            return b


class Leaf(Node):
    """define a leaf"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Define the position of a leaf in a tree"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ end of tree number"""
        return 1


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

    def depth(self):
        """linking to the depth max node"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ coount nodes, point to node class"""
        return self.root.count_nodes_below(only_leaves=only_leaves)
