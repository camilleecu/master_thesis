import numpy as np
import pandas as pd
# import pygraphviz as pgv
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from collections import OrderedDict
from pct.tree.node.node import Node
from pct.tree.splitter.splitter import Splitter
import pct.tree.utils as utils

class Tree:
    """Main class containing the general functionality of predictive clustering trees.

    Main source for HMC:
        VENS, Celine, et al. Decision trees for hierarchical multi-label classification. 
        Machine learning, 2008, 73.2: 185.
    """

    VERBOSE = False # Verbosity level
    INITIAL_WEIGHT = 1.0 # The initial weight used for a sample.

    def __init__(self, *, min_instances=5, ftest=0.01):
        """Constructs a new predictive clustering tree (PCT).

        @param min_instances: The minimum number of (weighted) samples in a leaf node (stopping criterion).
        @param ftest: The p-value (in [0,1]) used in the F-test for the statistical significance of a split.
        """
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
        """Fits this PCT to the given dataset.

        @param x: Pandas dataframe holding the descriptive variables.
        @param y: Pandas dataframe holding the target variables (classification XOR regression).
        @param target_weights: Weights given to the target variables (mainly used for HMC)
            (default is None: the weights are then automatically generated based on target variances)
        @return: This Tree object, trained on the given dataset.
        """
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        # assert x.shape[0] == y.shape[0], "Number of instances and labels are mismatched"
        self.x = x
        self.y = y
        if utils.learning_task(y) == "classification":
            self.y = utils.create_prototypes(y)
        self.target_weights = target_weights
        if target_weights is None:
            self.target_weights = utils.get_target_weights(self.y)

        instance_weights = pd.DataFrame(np.full(x.shape[0], Tree.INITIAL_WEIGHT), index=x.index)
        self.numerical_attributes   = x.select_dtypes(include=np.number).columns
        self.categorical_attributes = x.select_dtypes(exclude=np.number).columns
        self.splitter = self.make_splitter()

        self.size = {"node_count": 0, "leaf_count": 0}
        self.root = self.build(self.x, self.y, instance_weights, None)
        # self.postProcess(self.root) #### this is needed only for classification tasks
        return self

    def make_splitter(self):
        """
        Constructs a splitter object for this tree. This function was abstracted because
        it is often the changing point for a new PCT method (RF, SSL, HMC, PBCT ...).
        """
        return Splitter(
            self.min_instances, 
            self.numerical_attributes, self.categorical_attributes,
            self.ftest, 
            self.target_weights
        )

    def build(self, x, y, instance_weights, parent_node): ### extra parameter y_parent??
        """Recursively build this predictive clustering tree.
        
        @param x: Pandas dataframe holding the descriptive variables at this point of the process.
        @param y: Pandas dataframe holding the target variables at this point of the process.
        @param instance_weights: Pandas dataframe holding the instance weights at this point.
        @param parent_node: Parent of the current node.
        @return: The current node.
        @postcondition: A new node will be constructed, increasing the size of this tree by 1.
        """
        attribute_name, criterion_value = self.splitter.find_best_split_item(x, y, instance_weights)

        # If no valid split is found, return a leaf node  ##Camille
        if attribute_name is None:
            self.size["leaf_count"] += 1
            node = Node(parent=parent_node)
            node.make_leaf(y, instance_weights)
            return node

        # Extract attribute values for splitting
        attribute_value = x[attribute_name]

    
        # If we didn't find an acceptable split, make the current node a leaf
        if ~self.is_acceptable(criterion_value):
            self.size["leaf_count"] += 1
            node = Node(parent=parent_node)
            node.make_leaf(y, instance_weights)
            return node

        # Print the output similar to Clus:
        if Tree.VERBOSE > 0:
            print (attribute_name, attribute_value, criterion_value)

        # Retrieve some convenience variables
        missing_ind = utils.is_missing(x[attribute_name])
        subset_ind  = utils.is_in_left_branch(x[attribute_name], attribute_value)
        weight1, weight2 = self.get_new_weights(instance_weights, missing_ind, subset_ind)

        # Init the current node
        self.size["node_count"] += 1
        node = Node(attribute_name, attribute_value, criterion_value, parent_node)
        node.set_proportion(weight1, weight2)
        node.set_prototype(y, instance_weights)

        # Retrieve the datasets and instance weights for the children
        # x_left  = x.loc[ subset_ind | missing_ind]
        # x_right = x.loc[~subset_ind | missing_ind]
        # y_left  = y.loc[x_left.index]
        # y_right = y.loc[x_right.index]
        # instance_weights_left  = instance_weights.loc[x_left.index]
        # instance_weights_right = instance_weights.loc[x_right.index]
        # instance_weights_left.loc[missing_ind]  *= weight1
        # instance_weights_right.loc[missing_ind] *= weight2

        # Define three groups based on ratings
l       like_users = x[attribute_name] > 50
        dislike_users = x[attribute_name] <= 50
        unknown_users = x[attribute_name].isna()

        # Create subsets for each split
        x_like = x.loc[like_users.index]
        x_dislike = x.loc[dislike_users.index]
        x_unknown = x.loc[unknown_users.index]

        y_like = y.loc[x_like.index]
        y_dislike = y.loc[x_dislike.index]
        y_unknown = y.loc[x_unknown.index]

        instance_weights_like = instance_weights.loc[x_like.index]
        instance_weights_dislike = instance_weights.loc[x_dislike.index]
        instance_weights_unknown = instance_weights.loc[x_unknown.index]


        # Call this function recursively on the node's children
        # node.children = [None,None]
        # node.children[0] = self.build( x_left , y_left , instance_weights_left , node )
        # node.children[1] = self.build( x_right, y_right, instance_weights_right, node )
        # return node
        # Recursively build the tree with three branches
        node.children = [ ##Camille
            self.build(x_like, y_like, instance_weights_like, node),
            self.build(x_dislike, y_dislike, instance_weights_dislike, node),
            self.build(x_unknown, y_unknown, instance_weights_unknown, node)
]


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
        total_weight   = np.sum(instance_weights)
        missing_weight = np.sum(instance_weights.loc[missing_index])
        subset_weight  = np.sum(instance_weights.loc[subset_index])
        weight1 = (subset_weight/(total_weight - missing_weight)).values[0]
        weight2 = 1 - weight1
        return (weight1, weight2)

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
            prototype = np.mean(node.prototype) #Camille
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
                
