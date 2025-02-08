import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import OrderedDict

import pct.tree.utils as utils
from pct.tree.tree import Tree
from pct.tree.node.node import Node
from pct.tree.splitter.splitter import Splitter

class BiClusteringTree(Tree): 
    """Class of PBCTs (for now: lookahead-PBCT of depth one for HMC).
    
    Source:
        Bruna Santos, Felipe Nakano, Ricardo Cerri, Celine Vens
        Predictive Bi-Clustering Trees for Hierarchical Multi-label Classification
        https://lirias.kuleuven.be/retrieve/580599
    """

    CLUS = False # Whether to apply some hacks to get the same output as CLUS

    def fit(self, x, y, vert_features, target_weights=None):
        """Fits this PBCT to the given dataset.

        @param vert_features: The feature representation to use for vertical splits.
            (V^F in the PBCT paper; see also PCT/doc/pbct_hmc.png)
        @type vert_features: Pandas DataFrame with index equal to the columns of y.
        @return: This PBCT, trained on the given dataset.
        """
        self.size["vert_node_count"] = 0
        # This part is very similar to super().fit
        self.x = x
        self.y = y
        if utils.learning_task(y) == "classification":
            self.y = utils.create_prototypes(y)
        self.vert_features = vert_features
        self.target_weights = target_weights
        if target_weights is None:
            self.target_weights = utils.get_target_weights(self.y)

        self.numerical_attributes   = x.select_dtypes(include=['int', 'float']).columns
        self.categorical_attributes = x.select_dtypes(exclude=['int', 'float']).columns

        # Setting the 4 different weights used for this learning problem
        H_instance_weights = pd.DataFrame(np.full(x.shape[0], Tree.INITIAL_WEIGHT))
        V_instance_weights = pd.DataFrame(np.ones(y.shape[1]), index=y.columns)   # Works for CLUS
        # V_instance_weights = pd.DataFrame(self.target_weights, index=y.columns) # What I would say
        self.H_target_weights = self.target_weights # hierarchical class weights
        self.V_target_weights = np.ones(self.y.shape[0])
        # self.V_target_weights = H_instance_weights.values[:,0] # What I would say

        # For the first vertical split, Clus seems to use these weights instead?
        if self.CLUS:
            self.V_target_weights = utils.get_target_weights(self.y.transpose())

        self.make_splitter()
        # maybe replace all of these parameters by boolean masks idx_instance and idx_target?
        # well except for the instance weights though (the missing values there have to be
        # multiplied with a weight, which is not the same for each branch in the tree)
        self.root = self.build(
            self.x, self.y, self.vert_features, H_instance_weights, V_instance_weights, 
            self.H_splitter.target_weights, self.V_splitter.target_weights, None
        )
        # might want to reset F-test and target weights of the splitters here?
        return self

    def make_splitter(self):
        """Constructs the horizontal and vertical splitter objects for this tree."""
        self.H_splitter = Splitter( # on [x, y]
            min_instances=self.min_instances,
            numerical_attributes=self.numerical_attributes, 
            categorical_attributes=self.categorical_attributes,
            ftest=self.ftest,
            target_weights=self.H_target_weights
        )
        self.V_splitter = Splitter( # on [vert_features, y.transpose()]
            min_instances=self.min_instances,
            numerical_attributes=self.vert_features.select_dtypes(include=np.number).columns,
            categorical_attributes=self.vert_features.select_dtypes(exclude=np.number).columns,
            ftest=self.ftest,
            target_weights=self.V_target_weights
        )
        return None

    def build(
            self, x, y, vert_features, H_instance_weights, V_instance_weights, 
            H_target_weights, V_target_weights, parent_node
        ):
        """Recursively build this predictive biclustering tree.

        @note: All parameters are intended to represent this point of the process, i.e. this 
            function finds a split on the given dataset and weights.
        @param x: See L{Tree#fit}.
        @param y: See L{Tree#fit}.
        @param vert_features: Pandas dataframe holding the V^F matrix.
        @param H_instance_weights: Pandas dataframe holding the instance weights for [x,y].
        @param V_instance_weights: Pandas dataframe holding the instance weights for [vert_features, y.transpose()].
        @param H_target_weights: Numpy array holding the target weights for y.
        @param V_target_weights: Numpy array holding the target weights for y.transpose().
        @param parent_node: See L{Tree#fit}.
        @return: The current node.
        """
        # For subsequent vertical splits, Clus, seems to use the normal weights
        if (self.CLUS and parent_node is not None): # (unorthodox hack to check for second V-split)
            V_target_weights = np.ones(len(V_target_weights))

        # Setting some vars
        self.H_splitter.target_weights = H_target_weights
        self.V_splitter.target_weights = V_target_weights
        self.H_splitter.ftest.set_value(self.ftest * len(y.columns) / len(self.y.columns))

        # Finding the best horizontal and vertical split
        H_attr_name, H_heur, H_attr_val = self.H_splitter.find_split( x, y, H_instance_weights )
        V_attr_name, V_heur, V_attr_val = self.V_splitter.find_split( vert_features, y.transpose(), V_instance_weights )
        # If there is a vertical split, use lookahead to find its heuristic
        if V_heur != -np.inf:
            V_heur_old = V_heur # For printing the heuristic value later
            V_heur, splits_info = self.lookahead(
                x, y, vert_features, H_instance_weights, H_target_weights, V_attr_name, V_attr_val
            )

        # Delegate the work to the correct split handler
        if H_heur >= V_heur:
            return self.handle_H_split(
                x, y, vert_features, H_instance_weights, V_instance_weights, H_target_weights, 
                V_target_weights, parent_node, H_attr_name, H_heur, H_attr_val
                )
        else:
            return self.handle_V_split(
                x, y, vert_features, H_instance_weights, V_instance_weights, H_target_weights, 
                V_target_weights, parent_node, V_attr_name, V_heur_old, V_attr_val, splits_info
                )


    def lookahead(
            self, x, y, vert_features, H_instance_weights, H_target_weights, V_attr_name, V_attr_val
        ):
        """Evaluates a vertical split by looking ahead to the 2 subsequent horizontal splits.

        @param V_attr_name: Name of the (vertical) splitting feature.
        @param V_attr_val : Splitting value for the given splitting feature.
        @return: A tuple containing 
            - The heuristic value of the vertical split, as a weighted combination
              of the heuristic values of the 2 subsequent horizontal splits.
            - A list of splitting info for the horizontal splits. Each entry contains
              a tuple with the return value of self.H_splitter.find_split, i.e.
              (splitting attribute name, heuristic value, splitting value)
        """
        # Retrieve the dataset for each side
        target_left = utils.is_in_left_branch(vert_features[V_attr_name], V_attr_val).values
        y_left  = y.iloc[:,  target_left]
        y_right = y.iloc[:, ~target_left]

        # Find the resulting horizontal splits
        ftest_level0 = self.ftest
        splits_info = [("",-np.inf,""), ("",-np.inf,"")]
        # Left
        self.H_splitter.target_weights = H_target_weights[ target_left]
        if self.CLUS: # For some reason, Clus seems to use these target weights for lookahead splits
            self.H_splitter.target_weights = np.ones(sum(target_left))
        self.H_splitter.ftest.set_value(ftest_level0 * len(y_left.columns)/len(self.y.columns))
        splits_info[0] = self.H_splitter.find_split(x, y_left , H_instance_weights)
        # Right
        self.H_splitter.target_weights = H_target_weights[~target_left]
        if self.CLUS: # For some reason, Clus seems to use these target weights for lookahead splits
            self.H_splitter.target_weights = np.ones(sum(~target_left))
        self.H_splitter.ftest.set_value(ftest_level0 * len(y_right.columns)/len(self.y.columns))
        splits_info[1] = self.H_splitter.find_split(x, y_right, H_instance_weights)
        # Reset the variables
        self.H_splitter.target_weights = H_target_weights
        self.H_splitter.ftest.set_value(ftest_level0)

        # Combined heuristic
        H_heur_left  = splits_info[0][1]
        H_heur_right = splits_info[0][1]
        V_heur = (
            max(0, y_left.shape[1]  / y.shape[1] * H_heur_left) +
            max(0, y_right.shape[1] / y.shape[1] * H_heur_right)
        )
        return V_heur, splits_info

    def handle_H_split(
            self, x, y, vert_features, H_instance_weights, V_instance_weights, H_target_weights, 
            V_target_weights, parent_node, attribute_name, criterion_value, attribute_value
        ):
        """Handles node construction and recursive calls in the case of a horizontal split."""
        # If we didn't find an acceptable split, make the current node a leaf
        if ~self.is_acceptable(criterion_value):
            self.size["leaf_count"] += 1
            node = Node(parent=parent_node)
            node.is_vertical = False
            node.make_leaf(y, H_instance_weights)
            return node

        # Print the output similar to Clus:
        if Tree.VERBOSE > 0:
            print(attribute_name, attribute_value, criterion_value, "(Horizontal)")

        # Retrieve some convenience variables
        missing_ind = utils.is_missing(x[attribute_name])
        subset_ind  = utils.is_in_left_branch(x[attribute_name], attribute_value)
        weight1, weight2 = self.get_new_weights(H_instance_weights, missing_ind, subset_ind)

        # Init the current node
        self.size["node_count"] += 1
        node = Node(attribute_name, attribute_value, criterion_value, parent_node)
        node.is_vertical = False
        node.set_proportion(weight1, weight2)
        node.set_prototype(y, H_instance_weights)

        # Retrieve the datasets and instance weights for the children
        x_left  = x.loc[ subset_ind | missing_ind]
        x_right = x.loc[~subset_ind | missing_ind]
        y_left  = y.loc[x_left.index]
        y_right = y.loc[x_right.index]
        H_instance_weights_left  = H_instance_weights.loc[x_left.index]
        H_instance_weights_right = H_instance_weights.loc[x_right.index]
        H_instance_weights_left.loc[missing_ind]  *= weight1
        H_instance_weights_right.loc[missing_ind] *= weight2
        V_target_weights_left  = V_target_weights[ subset_ind | missing_ind]
        V_target_weights_right = V_target_weights[~subset_ind | missing_ind]

        # Build the node's children in a recursive fashion
        node.children = [None,None]
        node.children[0] = self.build( x_left , y_left , vert_features, H_instance_weights_left , V_instance_weights, H_target_weights, V_target_weights_left, node )
        node.children[1] = self.build( x_right, y_right, vert_features, H_instance_weights_right, V_instance_weights, H_target_weights, V_target_weights_right, node )
        return node

    def handle_V_split(
            self, x, y, vert_features, H_instance_weights, V_instance_weights, H_target_weights, 
            V_target_weights, parent_node, V_attr_name, V_heur, V_attr_val, H_splits_info
        ):
        """Handles node construction and recursive calls in the case of a vertical split."""
        # Print the output similar to Clus:
        # NOTE The heuristic printed here is the heuristic of the split in the
        # (V^F, V^T) space, not the one resulting from the 2 horizontal splits!
        if Tree.VERBOSE > 0:
            print(V_attr_name, V_attr_val, V_heur, "(Vertical)")
        L_attr_name, L_crit_val, L_attr_val = H_splits_info[0]
        R_attr_name, R_crit_val, R_attr_val = H_splits_info[1]

        # Init the current node
        self.size["vert_node_count"] += 1
        node = Node(V_attr_name, V_attr_val, V_heur, parent_node)
        # node.set_proportion(1.0, 1.0)
        node.set_prototype(y, H_instance_weights)
        node.is_vertical = True

        # Retrieve the datasets for the children
        target_left = utils.is_in_left_branch(vert_features[V_attr_name], V_attr_val).values
        y_left  = y.iloc[:, target_left]
        y_right = y.iloc[:,~target_left]
        vert_features_left  = vert_features.loc[ target_left]
        vert_features_right = vert_features.loc[~target_left]
        V_instance_weights_left  = V_instance_weights.iloc[ target_left]
        V_instance_weights_right = V_instance_weights.iloc[~target_left]
        H_target_weights_left  = H_target_weights[ target_left]
        H_target_weights_right = H_target_weights[~target_left]
        
        # Build the node's children in a recursive fashion
        node.children = [None, None]
        node.children[0] = self.handle_H_split(x, y_left , vert_features_left , H_instance_weights, V_instance_weights_left , H_target_weights_left , V_target_weights, parent_node, L_attr_name, L_crit_val, L_attr_val)
        node.children[1] = self.handle_H_split(x, y_right, vert_features_right, H_instance_weights, V_instance_weights_right, H_target_weights_right, V_target_weights, parent_node, R_attr_name, R_crit_val, R_attr_val)
        return node


    def predict(self, x, single_label=False):
        classes = np.full(self.y.shape[1], True)
        if isinstance(x,pd.DataFrame):
            predictions = np.array([
                self.predict_instance(instance, classes, self.root)
                for index,instance in x.iterrows()
            ])
        else:
            predictions = np.array([self.predict_instance(x, classes, self.root)])
        return predictions

    def predict_instance(self, instance, classes, node):
        # When in a leaf, return the prototype
        if node.is_leaf:
            prototype = node.prototype
            # Sometimes there are missing target values in the prototype.
            while np.isnan(prototype).any():
                if ~node.parent.is_vertical:
                    # Proceed as usual
                    node = node.parent
                    prototype = node.prototype
                else:
                    # TODO test this -- trying to take the indices to go to the left here
                    isLeftChild = (node == node.parent.children[0])
                    target_left = utils.is_in_left_branch(
                        self.vert_features[node.parent.attribute_name], node.parent.attribute_value
                    )
                    if isLeftChild:
                        ind = (classes &  target_left.values)
                    else:
                        ind = (classes & ~target_left.values)
                    node = node.parent
                    prototype = node.prototype[ind[classes]]
            return prototype

        # Call this function recursively
        if node.is_vertical: # (can also be checked with node.attribute_name in self.categorical_attributes or self.numerical_attributes)
            # Find the part of the prediction that goes to the left child
            target_left = utils.is_in_left_branch(self.vert_features[node.attribute_name], node.attribute_value)
            L_ind = (classes &  target_left.values)
            R_ind = (classes & ~target_left.values)

            pred = np.full(sum(classes), np.nan)
            pred[L_ind[classes]] = self.predict_instance(instance, L_ind, node.children[0])
            pred[R_ind[classes]] = self.predict_instance(instance, R_ind, node.children[1])
            return pred

        # For horizontal nodes -- as before
        if utils.is_missing(instance[node.attribute_name]):
            left_prediction  = node.proportion_left  * self.predict_instance(instance, classes, node.children[0])
            right_prediction = node.proportion_right * self.predict_instance(instance, classes, node.children[1])
            return left_prediction + right_prediction
        else:
            # 0 (false) if left, 1 (true) if right
            child = ~utils.is_in_left_branch(instance[node.attribute_name], node.attribute_value)
            return self.predict_instance(instance, classes, node.children[child])


    def decision_path_instance(self, x):
        """Build the decision path for the given instance."""
        decision_path = OrderedDict() # (key, value) = (visited node, decision path value)
        decision_path[self.root] = 1  # Root is always fully visited
        queue = [self.root]           # Stopping criterion

        while len(queue) != 0:
            # Loop management
            node = queue.pop(0)
            if len(node.children) == 0:
                continue # Nothing left to set here
            queue.extend(node.children)

            if node.is_vertical:
                # Vertical splits don't affect decision path
                decision_path[node.children[0]] = decision_path[node]
                decision_path[node.children[1]] = decision_path[node]
                del decision_path[node] # Remove the vertical node from the path
            else:
                # Continue as usual
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
