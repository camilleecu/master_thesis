import numpy as np
import pandas as pd
# import pygraphviz as pgv
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from collections import OrderedDict
from pct.tree.node.node import Node
from pct.tree.splitter.splitter import Splitter
import pct.tree.utils as utils
# from pct.tree.ftest.ftest import FTest
import matplotlib.pyplot as plt


class Tree:
    """Main class containing the general functionality of predictive clustering trees.
    get_prediction
    Main source for HMC:
        VENS, Celine, et al. Decision trees for hierarchical multi-label classification. 
        Machine learning, 2008, 73.2: 185.
    """

    VERBOSE = False  # Verbosity level
    INITIAL_WEIGHT = 1.0  # The initial weight used for a sample.

    def __init__(self, *, min_instances=7, max_depth=2):  # , ftest=0.01
        """Constructs a new predictive clustering tree (PCT).

        @param min_instances: The minimum number of (weighted) samples in a leaf node (stopping criterion).
        @param ftest: The p-value (in [0,1]) used in the F-test for the statistical significance of a split.
        """
        # self.ftest = ftest
        self.min_instances = min_instances
        self.max_depth = max_depth
        self.splitter = None
        self.x = None
        self.y = None
        self.target_weights = None
        self.root = None
        self.size = {"node_count": 0, "leaf_count": 0}
        self.categorical_attributes = None
        self.numerical_attributes = None
        self.pruning_strat = None



    def create_rI_rU(self, x, y):
        """Dynamically generate rI and rU dictionaries based on current subset."""
        rI = {}
        rU = {}

        for user_id in x.index: # x is a DataFrame
            for item_id, rating in zip(x.columns, x.loc[user_id]):
                if rating > 0:  # Only consider rated items
                    if item_id not in rI:
                        rI[item_id] = []
                    rI[item_id].append((user_id, rating))

                    if user_id not in rU:
                        rU[user_id] = []
                    rU[user_id].append((item_id, rating))
        
        return rI, rU


    def fit(self, x, y, target_weights=None):
        """
        Fit the predictive clustering tree on the given dataset and store rI and rU.
        """
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        print("✅ Converted x and y to DataFrame")

        self.x = x
        self.y = y
        print("✅ Assigned x and y")

        if utils.learning_task(y) == "classification":
            print("✅ Applying classification preprocessing...")
            self.y = utils.create_prototypes(y)

        print("✅ Creating target weights...")
        self.target_weights = target_weights
        if target_weights is None:
            self.target_weights = utils.get_target_weights(self.y)

        print("✅ Identifying numerical and categorical attributes...")
        self.numerical_attributes = x.select_dtypes(include=np.number).columns
        self.categorical_attributes = x.select_dtypes(exclude=np.number).columns

        print("✅ Creating Splitter...")
        self.splitter = self.make_splitter()

        if self.splitter is None:
            raise ValueError("🚨 Splitter was not properly initialized!")

        print("✅ Calling build()...")
        instance_weights = pd.DataFrame(np.full(x.shape[0], Tree.INITIAL_WEIGHT), index=x.index)
        self.root = self.build(self.x, self.y, instance_weights, None)
        print("✅ Tree built successfully!")

        return self

    def build(self, x, y, instance_weights, parent_node, depth=0):
        """Recursively build this predictive clustering tree with updated rI and rU per subset."""

        if depth > self.max_depth:
            print(f"🍃 Reached max depth at node {parent_node}. Stopping recursion.")
            self.size["leaf_count"] += 1
            node = Node(parent=parent_node)
            node.make_leaf(y, instance_weights)
            return node

        print("🌲 Building predictive clustering tree...")

        # Select the best item (feature) for splitting
        best_item, criterion_value = self.splitter.find_best_split_item(x, y, instance_weights)
        print("🔍 Best item for splitting: ", best_item)

        if best_item is None:
            print("🍃 Creating leaf node (no valid split found)...")
            self.size["leaf_count"] += 1
            node = Node(parent=parent_node)
            node.make_leaf(y, instance_weights)
            return node

        # Create a node for this item split
        self.size["node_count"] += 1
        node = Node(best_item, criterion_value, parent_node)

        print(f"Type of x: {type(x)}")
        print(f"Shape of x: {x.shape}")

        # Create rI and rU dynamically based on current subset
        rI_subset, rU_subset = self.create_rI_rU(x, y)
        print("rI_subset: ", rI_subset)
        print("rU_subset: ", rU_subset)

        # Get all users who rated this item in the **current subset**
        users_rated_item = set(user for user, _ in rI_subset.get(best_item, []))
        print("👥 Users who rated item {}: {}".format(best_item, len(users_rated_item)))

        # Classify users into three groups: Lovers, Haters, Unknowns
        lovers = [u for u in users_rated_item if dict(rU_subset[u]).get(best_item, 0) >= 4]
        haters = [u for u in users_rated_item if dict(rU_subset[u]).get(best_item, 0) <= 3]
        unknowns = [u for u in x.index if u not in users_rated_item]

        print("❤️ Lovers:", len(lovers))
        print("💔 Haters:", len(haters))
        print("❓ Unknowns:", len(unknowns))

        # Extract subsets based on filtered indices
        x_lovers, y_lovers = x.loc[lovers].copy(), y.loc[lovers].copy()
        x_haters, y_haters = x.loc[haters].copy(), y.loc[haters].copy()
        x_unknowns, y_unknowns = x.loc[unknowns].copy(), y.loc[unknowns].copy()

        instance_weights_lovers = instance_weights.loc[lovers].copy()
        instance_weights_haters = instance_weights.loc[haters].copy()
        instance_weights_unknowns = instance_weights.loc[unknowns].copy()



        # Recursively build the tree for each group with updated rI and rU
        print("🔄 Recursively building tree for subsets...")
        node.children = [
            self.build(x_lovers, y_lovers, instance_weights_lovers, node, depth + 1),
            self.build(x_haters, y_haters, instance_weights_haters, node, depth + 1),
            self.build(x_unknowns, y_unknowns, instance_weights_unknowns, node, depth + 1)
        ]

        return node


    def make_splitter(self):
        """
        Constructs a splitter object for this tree. This function was abstracted because
        it is often the changing point for a new PCT method (RF, SSL, HMC, PBCT ...).
        """
        return Splitter(
            self.min_instances,
            self.numerical_attributes,
            self.categorical_attributes,
            # self.ftest, 
            self.target_weights
        )

    @staticmethod
    def is_acceptable(criterion_value):
        """Returns true if and only if the given value is an acceptable splitting value."""
        return criterion_value != -np.inf

    @staticmethod
    def get_new_weights(instance_weights, missing_index, subset_index):
        """Returns the weight multipliers to be used for an instance that wants to pass through
        a node, in case that instance has a missing value for the splitting attribute.

        @param instance_weights: The weights of the instances for the current node.
        @param missing_index: Indices of the missing values for the splitting variable.
        @param subset_index: Indices of the values going to the left branch of the node.
        @return: Tuple (weight for left child, weight for right child), summing to 1.
        """
        total_weight = np.sum(instance_weights)
        missing_weight = np.sum(instance_weights.loc[missing_index])
        subset_weight = np.sum(instance_weights.loc[subset_index])
        weight1 = (subset_weight / (total_weight - missing_weight)).values[0]
        weight2 = 1 - weight1
        return (weight1, weight2)

    def update_instance_weights(self, node_instance_weights, missing_index, partition_size, total_size):
        # TODO unused?
        proportion = (partition_size - len(missing_index)) / (total_size - len(missing_index))
        # subset (total - missing)
        node_instance_weights.loc[missing_index] = node_instance_weights.loc[missing_index] * proportion
        # print (proportion)
        return node_instance_weights

    def postProcess(self, node):
        """
        For classification tasks, recursively prunes leaves under the given node, if they have
        the same parent and same prediction prototype.

        @param node: The 'root' node for this operation.
        """
        # Recursively applies L{postProcessSamePrediction} on each (in)direct child node of 
        # the given node. The children are handled in postfix order, i.e. all children are 
        # handled before the parent. This is important to correctly prune leaves iteratively.
        for n in node.children:
            self.postProcess(n)
        self.postProcessSamePrediction(node)

    def postProcessSamePrediction(self, node, pruning_strat=None):
        """
        If the children of the given node are leaves which give the same output for
        L{get_prediction}, prunes them away with Node's L{make_leaf_prune}.

        @param node: The node whose children should be considered.
        @param pruning_strat: TODO unused.
        """
        if (~node.is_leaf
                and node.children[0].is_leaf
                and self.get_prediction(node.children[0]) == self.get_prediction(node.children[1])
        ):
            node.make_leaf_prune(node.children[0], node.children[1])

    def get_prediction(self, leaf):
        """Returns the prediction prototype for the given leaf."""
        if leaf.prototype is not None:
            return np.argmax(leaf.prototype)  ## gimmick to get the same output as clus
        return -1

    def predict(self, x, single_label=False):
        """Predicts the labels for each instance in the given dataset.

        @param x: Dataframe containing instances and features (rows and columns).
        @param single_label: For classification problems, whether to return the target (class)
            containing the highest score instead of the prediction probabilities.
        @return: Target predictions (regression) or prediction probabilities (classification).
        """
        x = pd.DataFrame(x)
        predictions = np.array([
            self.predict_instance(instance, self.root)
            for _, instance in x.iterrows()
        ])
        return np.argmax(predictions, axis=1) if single_label else predictions

    def predict_instance(self, instance, node):
        """Recursively predict the given instance over the nodes of this tree.
        
        @param instance: The single instance to predict.
        @param node: The current node in the recursive process.
        """
        # When in a leaf, return the prototype
        if node.is_leaf:
            prototype = np.mean(node.prototype)  # Camille
            # Sometimes there are missing target values in the prototype.
            while np.isnan(prototype).any() and node.parent is not None:
                node = node.parent
                prototype = node.prototype
            return prototype

        # Call this function recursively
        value = np.array(instance[node.attribute_name])
        if utils.is_missing(value):
            left_prediction = node.proportion_left * self.predict_instance(instance, node.children[0])
            right_prediction = node.proportion_right * self.predict_instance(instance, node.children[1])
            return left_prediction + right_prediction
        else:
            # 0 (false) if left, 1 (true) if right
            child = ~utils.is_in_left_branch(value, node.attribute_value)
            return self.predict_instance(instance, node.children[child])

    def decision_path(self, x):
        """Returns the decision path for each instance in the given dataset.

        The decision path is a binary vector (taking the role of a feature representation),
        representing the nodes that an instance passes through (1). To deal with missing values,
        we turn this into a real vector representing the proportion of the instance that
        passes through each node. The tree is traversed with a breadth-first search.
        
        @param x: Dataframe containing instances and features (rows and columns).
        @return: The decision path for each instance in the input dataset.
        """
        return np.array([
            self.decision_path_instance(instance) for index, instance in x.iterrows()
        ])

    def decision_path_instance(self, x):
        """Build the decision path for the given instance.
        
        @param x: The (single) instance to make a decision path of.
        @param node: The current node in the recursive process.
        """
        decision_path = OrderedDict()  # (key, value) = (visited node, decision path value)
        decision_path[self.root] = 1  # Root is always fully visited
        queue = [self.root]  # Stopping criterion

        while len(queue) != 0:
            # Loop management
            node = queue.pop(0)
            if len(node.children) == 0:
                continue  # Nothing left to set here
            queue.extend(node.children)

            # Set the decision path values
            value = np.array(x[node.attribute_name])
            if utils.is_missing(value):
                decision_path[node.children[0]] = decision_path[node] * node.proportion_left
                decision_path[node.children[1]] = decision_path[node] * node.proportion_right
            else:
                goLeft = utils.is_in_left_branch(value, node.attribute_value)
                goLeft = bool(goLeft)  # Fix for weird python.bool vs numpy.bool stuff
                decision_path[node.children[0]] = decision_path[node] if goLeft else 0
                decision_path[node.children[1]] = decision_path[node] if ~goLeft else 0

        return list(decision_path.values())

    def nodes(self):
        nodes = []
        queue = [self.root]
        while len(queue) != 0:
            node = queue.pop(0)
            queue.extend(node.children)
            nodes.append(node)
        return nodes

    def leaves(self):
        leaves = []
        queue = [self.root]
        while len(queue) != 0:
            node = queue.pop(0)
            queue.extend(node.children)
            if node.is_leaf:
                leaves.append(node)
        return leaves

    def print_tree_structure(self, node=None, level=0):
        """Prints the tree structure for debugging."""
        if level > self.max_depth:
            return
        if node is None:
            node = self.root  # Start from the root if no node is passed

        # Indentation based on the level in the tree
        indent = " " * (level * 4)

        # Print basic information about the node
        print(f"{indent}Node: {node.name}")
        print(f"{indent}Attribute: {node.attribute_name}")
        # print(f"{indent}Value: {node.attribute_value}")
        print(f"{indent}Criterion: {node.criterion_value}")

        if node.is_leaf:
            print(f"{indent}Leaf Node: Yes")
            print(f"{indent}Prediction: {node.prototype}")  # If the leaf node has a prediction (prototype)
        else:
            print(f"{indent}Leaf Node: No")
            print(f"{indent}Children:")
            # Recursively print information for each child node
            for child in node.children:
                self.print_tree_structure(child, level + 1)  # Increase the level for deeper indentation

    def visualize_tree(self):
        """Creates a graphical representation of the decision tree using networkx."""
        graph = nx.DiGraph()
        self._add_edges(self.root, graph)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph, seed=42)  # Positioning of nodes
        labels = {node: data["label"] for node, data in graph.nodes(data=True)}

        nx.draw(graph, pos, with_labels=True, labels=labels, node_size=3000, node_color="lightblue", edge_color="gray")
        plt.title("Decision Tree Structure")
        plt.show()

    def _add_edges(self, node, graph, parent=None):
        """Recursively adds edges to the graph for visualization."""
        if node is None:
            return
        
        # Assign a label for the node
        node_label = f"{node.attribute_name}\nCriterion: {node.criterion_value}"
        graph.add_node(node.name, label=node_label)

        # Add an edge from the parent to the current node
        if parent:
            graph.add_edge(parent, node.name)

        # Recursively add children
        for child in node.children:
            if child is not None:
                self._add_edges(child, graph, node.name)