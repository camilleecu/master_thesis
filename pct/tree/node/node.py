import numpy as np


class Node:
    """Class representing the nodes in a predictive clustering tree."""

    # Class properties
    leaf_count = 0 # Number of leaves made
    id = 0         # Number of nodes made (used for giving unique ID's to the nodes)

    def __init__(self, attribute_name=None,  criterion_value=None, parent=None, depth=0): #attribute_value=None, (deleted)
        """Creates a new node with attribute_name as the best item from tree.py 
        and criterion_value from tree.py.
        
        @param attribute_name : The best feature (item) chosen for splitting.
        @param criterion_value: The criterion value used to evaluate the split.
        @param parent: The parent node of this node.
        @param depth: Depth of the node in the tree.
        """

        Node.id  += 1
        # self.prototype = None
        # self.children = [] # Max of length 2 because binary splits
        self.children = [None, None, None]  # Ternary splits: [like, unknown, dislike]
        self.parent = parent
        self.depth = depth

        # Assign values from tree.py
        self.attribute_name = attribute_name  # Best feature for splitting
        self.criterion_value = criterion_value  # Split quality metric

        self.lovers_count = 0    # Count of users who love the item
        self.haters_count = 0    # Count of users who hate the item
        self.unknowns_count = 0  # Count of users with unknown rating for the item
        

        if attribute_name is not None:
            self.name = f"{attribute_name}_{Node.id}"

    def make_leaf(self, y, weights):
        """Turn this node into a leaf node, setting a prototype for later classification.
        If the tree depth limit is reached (depth > self.max_depth), a leaf node is created using Node.make_leaf().
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
            
    def set_split_info(self, split_item, user_rating, lovers, haters, unknowns):
        """Store information about the split, user rating, and distribution."""
        self.split_item = split_item
        self.user_rating = user_rating
        self.lovers_count = len(lovers)
        self.haters_count = len(haters)
        self.unknowns_count = len(unknowns)


    def print(self):
        print(self.attribute_name)
        # print(self.attribute_value)
        print(self.criterion_value)
        print('-------')
