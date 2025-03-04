import numpy as np


class Node:
    """Class representing the nodes in a predictive clustering tree."""

    # Class properties
    leaf_count = 0 # Number of leaves made
    id = 0         # Number of nodes made (used for giving unique ID's to the nodes)

    def __init__(self, attribute_name=None, attribute_value=None, criterion_value=None, parent=None):
        """Makes a new node in the tree with given properties.

        @param attribute_name : The name of the attribute used to split in this node.
        @param attribute_value: The value or set of values on which the split was made.
        @param criterion_value: The value of the heuristic function reached for this split.
        @param parent: The parent node of this node.
        """
        Node.id  += 1
        self.prototype = None
        # self.children = [] # Max of length 2 because binary splits
        self.children = [None, None, None]  # Ternary splits: [like, unknown, dislike]
        self.parent = parent

        self.attribute_name = attribute_name
        self.attribute_value = attribute_value
        self.criterion_value = criterion_value
        self.proportion_like = None
        self.proportion_unknown = None
        self.proportion_dislike = None
        
        if attribute_name is not None:
            self.name = str(attribute_name) + "_" + str(self.id)

    def make_leaf(self, y, weights):
        """Turn this node into a leaf node, setting a prototype for later classification.

        @param y      : The target vector at this node in the tree.
        @type  y      : Pandas.DataFrame
        @param weights: The instance weights for each of the target entries in y.
        @type  weights: Pandas.DataFrame
        """
        self.y = y
        self.prototype, summed_weights = self.get_prototype(y, weights)
        self.name = "leaf_" + str(self.id) + "=" + str(self.prototype) + " (" + str(summed_weights) + ")"
        # self.prototype = [np.nan]
        
    def set_prototype(self, y, weights):
        """Sets the prototype for this node."""
        self.y = y
        self.prototype, _ = self.get_prototype(y, weights)
    
    def get_prototype(self, y, weights):
        """Returns the prototype for this node, along with the sum of given weights.

        @return: A tuple containing (prototype, summed_weights).
        """
        summed_weights = np.sum(weights.values)

        prototype = np.nansum(self.y.values * weights.values, axis=0)
        prototype /= np.sum(
            np.repeat(weights.values, repeats=self.y.shape[1], axis=1),
            axis=0,
            where=(~np.isnan(self.y.values))
        )
        prototype = np.round(prototype, 6)
        return prototype, summed_weights        

    def make_leaf_prune(self, node1, node2):
        """
        Make this node a pruned leaf, by combining the properties of its children.
        Used in classification for combining leaves with the same prediction prototype.
        """
        self.y = np.vstack((node1.y,node2.y))
        self.prototype, summed_weights = self.get_prototype(y, weights)
        self.children = [None, None, None]  # Ternary splits: [like, unknown, dislike]
        # Node.leaf_count += 1
        self.name = "leaf_" + str(self.id) + "=" + str(self.prototype) + " (" + str(summed_weights) + ")"

    @property
    def is_leaf(self):
        """Returns true if and only if this node has no children."""
        return len(self.children) == 0

    def set_proportion(self, proportion_like, proportion_unknown, proportion_dislike):
        """Sets the proportion of instances going to each child node in a ternary split."""
        total = proportion_like + proportion_unknown + proportion_dislike
        if total > 0:
            self.proportion_like = proportion_like / total
            self.proportion_unknown = proportion_unknown / total
            self.proportion_dislike = proportion_dislike / total
        else:
            self.proportion_like = 0
            self.proportion_unknown = 0
            self.proportion_dislike = 0


    def print(self):
        print(self.attribute_name)
        print(self.attribute_value)
        print(self.criterion_value)
        print('-------')
