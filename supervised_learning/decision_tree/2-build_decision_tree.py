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
        self.sub_population = None
        self.depth = depth

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
            node_repr += self.left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            node_repr += self.right_child_add_prefix(self.right_child.__str__())
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


class Leaf(Node):
    """
    Leaf class for a decision tree
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

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
