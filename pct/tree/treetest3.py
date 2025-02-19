import numpy as np
import pandas as pd
import networkx as nx
from collections import OrderedDict
from pct.tree.node.node import Node
from pct.tree.splitter.splittertest3 import Splitter3  # Updated to Splitter3
import pct.tree.utils as utils


class Tree3:
    """Main class containing the general functionality of predictive clustering trees."""
    VERBOSE = False
    INITIAL_WEIGHT = 1.0

    def __init__(self, *, min_instances=5, ftest=0.01):
        """Constructs a new predictive clustering tree (PCT)."""
        self.ftest = ftest
        self.min_instances = min_instances
        self.splitter = None
        self.x = None
        self.y = None
        self.target_weights = None
        self.root = None
        self.size = {"node_count": 0, "leaf_count": 0}
        self.categorical_attributes = None
        self.numerical_attributes = None
        self.pruning_strat = None

    def fit(self, x, y, target_weights=None):
        """Fits this PCT to the given dataset."""
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        self.x = x
        self.y = y
        if utils.learning_task(y) == "classification":
            self.y = utils.create_prototypes(y)
        self.target_weights = target_weights
        if target_weights is None:
            self.target_weights = utils.get_target_weights(self.y)

        instance_weights = pd.DataFrame(np.full(x.shape[0], Tree3.INITIAL_WEIGHT), index=x.index)
        self.numerical_attributes = x.select_dtypes(include=np.number).columns
        self.categorical_attributes = x.select_dtypes(exclude=np.number).columns
        self.splitter = self.make_splitter()

        self.size = {"node_count": 0, "leaf_count": 0}
        self.root = self.build(self.x, self.y, instance_weights, None)
        return self

    def make_splitter(self):
        """Constructs a splitter object for this tree."""
        return Splitter3(
            self.min_instances, 
            self.numerical_attributes, self.categorical_attributes,
            self.ftest, 
            self.target_weights
        )

    def build(self, x, y, instance_weights, parent_node):
        """Recursively build this predictive clustering tree."""
        attribute_name, criterion_value, attribute_value = self.splitter.find_split(x, y, instance_weights)
    
        # If we didn't find an acceptable split, make the current node a leaf
        if ~self.is_acceptable(criterion_value):
            self.size["leaf_count"] += 1
            node = Node(parent=parent_node)
            node.make_leaf(y, instance_weights)
            return node

        if Tree3.VERBOSE > 0:
            print(attribute_name, attribute_value, criterion_value)

        missing_ind = utils.is_missing(x[attribute_name])
        subset_ind_left = utils.is_in_left_branch(x[attribute_name], attribute_value)
        subset_ind_right = utils.is_in_right_branch(x[attribute_name], attribute_value)
        subset_ind_middle = ~(subset_ind_left | subset_ind_right)  # Middle is everything else

        weight_left, weight_middle, weight_right = self.get_new_weights_for_three(
            instance_weights, missing_ind, subset_ind_left, subset_ind_right
        )

        self.size["node_count"] += 1
        node = Node(attribute_name, attribute_value, criterion_value, parent_node)
        node.set_proportion(weight_left, weight_middle, weight_right)
        node.set_prototype(y, instance_weights)

        # Split the data into three partitions
        x_left = x.loc[subset_ind_left | missing_ind]
        x_middle = x.loc[subset_ind_middle | missing_ind]
        x_right = x.loc[subset_ind_right | missing_ind]
        
        y_left = y.loc[x_left.index]
        y_middle = y.loc[x_middle.index]
        y_right = y.loc[x_right.index]
        
        instance_weights_left = instance_weights.loc[x_left.index]
        instance_weights_middle = instance_weights.loc[x_middle.index]
        instance_weights_right = instance_weights.loc[x_right.index]
        
        # Adjust instance weights for missing values
        instance_weights_left.loc[missing_ind] *= weight_left
        instance_weights_middle.loc[missing_ind] *= weight_middle
        instance_weights_right.loc[missing_ind] *= weight_right

        # Call recursively for three children
        node.children = [None, None, None]
        node.children[0] = self.build(x_left, y_left, instance_weights_left, node)  # Left child
        node.children[1] = self.build(x_middle, y_middle, instance_weights_middle, node)  # Middle child
        node.children[2] = self.build(x_right, y_right, instance_weights_right, node)  # Right child

        return node

    @staticmethod
    def is_acceptable(criterion_value):
        """Returns true if and only if the given value is an acceptable splitting value."""
        return criterion_value != -np.inf

    @staticmethod
    def get_new_weights_for_three(instance_weights, missing_index, subset_left, subset_right):
        """Returns the weight multipliers for the three child branches."""
        total_weight = np.sum(instance_weights)
        missing_weight = np.sum(instance_weights.loc[missing_index])
        subset_left_weight = np.sum(instance_weights.loc[subset_left])
        subset_right_weight = np.sum(instance_weights.loc[subset_right])
        subset_middle_weight = total_weight - missing_weight - subset_left_weight - subset_right_weight
        
        weight_left = subset_left_weight / (total_weight - missing_weight)
        weight_right = subset_right_weight / (total_weight - missing_weight)
        weight_middle = subset_middle_weight / (total_weight - missing_weight)
        
        return weight_left, weight_middle, weight_right

    def print_tree_structure(self, node=None, level=0):
        """Prints the tree structure for debugging."""
        if node is None:
            node = self.root  
        
        indent = " " * (level * 4) 
        print(f"{indent}Node: {node.name}")
        print(f"{indent}Attribute: {node.attribute_name}")
        print(f"{indent}Value: {node.attribute_value}")
        print(f"{indent}Criterion: {node.criterion_value}")
        
        if node.is_leaf:
            print(f"{indent}Leaf Node: Prototype {node.prototype}")
        else:
            print(f"{indent}Children:")
            for i, child in enumerate(node.children):
                self.print_tree_structure(child, level + 1)

        
    def update_instance_weights(self, node_instance_weights, missing_index, partition_size, total_size):
        # TODO unused?
        proportion = (partition_size - len(missing_index))/(total_size - len(missing_index))
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
         
    def postProcessSamePrediction(self, node, pruning_strat = None):
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
            node.make_leaf_prune(node.children[0],node.children[1])            
            
    def get_prediction(self,leaf):
        """Returns the prediction prototype for the given leaf."""
        if leaf.prototype is not None:
            return np.argmax(leaf.prototype) ## gimmick to get the same output as clus
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
            for _,instance in x.iterrows()
        ])
        return np.argmax(predictions, axis=1) if single_label else predictions

    def predict_instance(self, instance, node):
        """Recursively predict the given instance over the nodes of this tree.
        
        @param instance: The single instance to predict.
        @param node: The current node in the recursive process.
        """
        # When in a leaf, return the prototype
        if node.is_leaf:
            prototype = node.prototype
            # Sometimes there are missing target values in the prototype.
            while np.isnan(prototype).any() and node.parent is not None:
                node = node.parent
                prototype = node.prototype
            return prototype

        # Call this function recursively
        value = np.array(instance[node.attribute_name])
        if utils.is_missing(value):
            left_prediction  = node.proportion_left  * self.predict_instance(instance, node.children[0])
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
        decision_path = OrderedDict() # (key, value) = (visited node, decision path value)
        decision_path[self.root] = 1  # Root is always fully visited
        queue = [self.root]           # Stopping criterion

        while len(queue) != 0:
            # Loop management
            node = queue.pop(0)
            if len(node.children) == 0:
                continue # Nothing left to set here
            queue.extend(node.children)

            # Set the decision path values
            value = np.array(x[node.attribute_name])
            if utils.is_missing(value):
                decision_path[node.children[0]] = decision_path[node] * node.proportion_left
                decision_path[node.children[1]] = decision_path[node] * node.proportion_right
            else:
                goLeft = utils.is_in_left_branch(value, node.attribute_value)
                goLeft = bool(goLeft) # Fix for weird python.bool vs numpy.bool stuff
                decision_path[node.children[0]] = decision_path[node] if  goLeft else 0
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
        queue  = [self.root]
        while len(queue) != 0:
            node = queue.pop(0)
            queue.extend(node.children)
            if node.is_leaf:
                leaves.append(node)
        return leaves

    def draw_tree(self, fileName):
        """Renders this tree as a networkx graph, stored in a png for the given filename."""
        G = nx.DiGraph()
        self.__convert_to_graph(self.root, G)

        for _, _, d in G.edges(data=True):
            d['label'] = d.get('value','')
        # print (G.nodes())
        A = nx.drawing.nx_agraph.to_agraph(G)
        A.layout(prog='dot')
        A.draw(fileName + '.png')

    def __convert_to_graph(self, node, G):
        """Recursive function setting the node and edge information from this tree into G."""
        # print (node.attribute_name)
        if node.is_leaf:
            return
        else:
            G.add_node(node.name)
            if node.attribute_name in self.numerical_attributes:
                G.add_edge(node.name, node.children[1].name, value = "<=" + str(node.attribute_value))
                self.__convert_to_graph(node.children[1], G)
                G.add_edge(node.name, node.children[0].name, value = ">" + str(node.attribute_value))
                self.__convert_to_graph(node.children[0], G)

            elif node.attribute_name in self.categorical_attributes:
                G.add_edge(node.name, node.children[0].name, value = "in" + str(node.attribute_value))
                self.__convert_to_graph(node.children[0], G)
                G.add_edge(node.name, node.children[1].name, value = "not in" + str(node.attribute_value))
                self.__convert_to_graph(node.children[1], G)
                



 

