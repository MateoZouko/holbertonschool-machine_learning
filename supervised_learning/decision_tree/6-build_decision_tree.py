#!/usr/bin/env python3
"""
Task 1
"""
import numpy as np


class Node:
    """
    Node class for a decision tree
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.depth = depth
        self.lower = dict()
        self.upper = dict()
        self.recent_node = dict()
        self.indicator = ""

    def max_depth_below(self):
        """
        Calculate the maximum depth below the node
        """
        if self.is_leaf:
            return self.depth
        else:
            return max(self.left_child.max_depth_below(),
                       self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below the node
        """
        if self.is_leaf:
            return 1
        else:
            if only_leaves:
                return self.left_child.\
                    count_nodes_below(only_leaves=True) + self.right_child.\
                    count_nodes_below(only_leaves=True)
            else:
                return 1 + self.left_child.\
                    count_nodes_below(only_leaves=False) + self.right_child.\
                    count_nodes_below(only_leaves=False)

    def __str__(self):
        """
        Return the string representation of the node
        """
        node_type = "root" if self.is_root else "-> node"
        node_repr = f"{node_type} [feature={self.feature},\
 threshold={self.threshold}]\n"
        if self.left_child:
            node_repr +=\
                self.left_child_add_prefix(self.left_child.__str__().strip())
        if self.right_child:
            node_repr +=\
                self.right_child_add_prefix(self.right_child.__str__().strip())
        return node_repr

    def left_child_add_prefix(self, text):
        """
        Add prefix to the left child
        """
        lines = text.split("\n")
        new_text = "    +--"+lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  "+x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """
        Add prefix to the right child
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def get_leaves_below(self):
        """
        Return the leaves below the node
        """
        if self.is_leaf:
            return [self]
        else:
            return self.left_child.get_leaves_below() +\
                self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """
        Update the bounds of the node
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:

            node = self
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

            if child == self.left_child:
                child.lower.update({node.feature: node.threshold})

            elif child == self.right_child:
                child.upper.update({node.feature: node.threshold})

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        Find in wich leaf where A(individual values) actually goes
        """
        def is_large_enough(A):
            """A is in the upper bracket"""
            arrSup = [np.greater(A[:, key], self.lower[key]) for
                      key in list(self.lower.keys())]
            arrSup_flatten = np.all(arrSup, axis=0)
            return arrSup_flatten

        def is_small_enough(A):
            """
            A is in the lower bracket
            """
            arrInf = [np.greater_equal(self.upper[key], A[:, key]) for
                      key in list(self.upper.keys())]
            arrInf_flatten = np.all(arrInf, axis=0)
            return arrInf_flatten

        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Alternative prediction (given)
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Leaf class for a decision tree
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth
        self.upper = dict()
        self.lower = dict()

    def max_depth_below(self):
        """
        Calculate the maximum depth below the node
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below the node
        """
        return 1

    def __str__(self):
        """
        Return the string representation of the leaf
        """
        return (f"-> leaf [value={self.value}] ")

    def get_leaves_below(self):
        """
        Return the leaves below the node
        """
        return [self]

    def update_bounds_below(self):
        """
        Update the bounds of the leaf
        """
        pass


class Decision_Tree():
    """
    Decision Tree class
    """
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
        """
        Calculate the maximum depth of the tree
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count the number of nodes in the tree
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Return the string representation of the tree
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Return the leaves of the tree
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Update the bounds of the tree
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """ fonction de prediction"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array([leaves[j].value for j in [np.array([leaf.indicator(A) for leaf in leaves])[:, i].nonzero()[0][0] for i in range(len(A))]])

    def pred(self, x):
        """ alternative prediction (given)"""
        return self.root.pred(x)
