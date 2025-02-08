import random
import numpy as np
import pandas as pd
import pct.tree.utils as utils
from pct.tree.tree import Tree
from pct.tree.node.node import Node
from pct.tree.splitter.splitter import Splitter
from pct.forest.forest_splitter import RandomForestSplitter

class RandomForestTree(Tree):

    def __init__(
        self,
        min_instances, ftest,
        num_sub_instances, num_sub_features,
        random_state=None
    ):
        """Construct a new PCT to use in a random forest.
        
        @param num_sub_instances: Number of instances to bootstrap.
        @param num_sub_features: Number of features to use for each split.
        @param random_state: Seed for the random number generator.
            (default is None: some pseudorandom seed is then used)
        """
        super().__init__(min_instances=min_instances, ftest=ftest)
        self.num_sub_instances = num_sub_instances
        self.num_sub_features = num_sub_features
        self.random = random.Random()
        self.random.seed(random_state) # If None, uses some pseudoRNG seed

    def fit(self, x, y, target_weights=None):
        """Fits this PCT to a bootstrapped version of the given data, using
        random feature selection at each split.
        """
        self.task = utils.learning_task(y)

        # Sample the instances (with replacement) and delete the indices (because of duplicates)
        num_instances = x.shape[0]
        self.ind_sub_instances = self.random.choices(range(num_instances), k=self.num_sub_instances)
        x_sub = x.iloc[self.ind_sub_instances,:].reset_index(drop=True)
        y_sub = y.iloc[self.ind_sub_instances,:].reset_index(drop=True)

        # Proceed as usual
        super().fit(x_sub, y_sub, target_weights=target_weights)

        # Reset the datasets (for later error calculation)
        # (also so there aren't a bunch of "_sub" copies in memory when fit is done)
        self.x = x
        self.y = y
        if self.task == "classification":
            self.y = utils.create_prototypes(y)
        return self

    def make_splitter(self):
        """Constructs a splitter object for this tree."""
        return RandomForestSplitter(
            self.min_instances, self.numerical_attributes, self.categorical_attributes, 
            self.ftest, self.target_weights, self.num_sub_features, self.random
        )

    def get_feature_importances(self, feat_imp_dataframe):
        """Aggregate feature heuristics from each node in the tree.
        
        @param feat_imp_dataframe: Pandas dataframe to write the results in.
        """
        # Uses breadth-first search
        queue = [self.root]
        while len(queue) != 0:
            node = queue[0]
            queue = queue[1:] # Pop the first entry
            if len(node.children) != 0:
                feat_imp_dataframe.loc[node.attribute_name] += node.criterion_value
                # Because node.children is usually of length 2, append is faster than extend
                for child in node.children:
                    queue.append(child)

    def compute_oob_error(self, oob_error_dataframe):
        """Computes the out-of-bag error, i.e. the error on samples outside the bootstrapped dataset.

        @param oob_error_dataframe: Pandas dataframe to write the results in.
        @param metric: TODO
            The metric to use. Defaults to MSE for regression problems and 
            (adjusted) accuracy for classification.
        """
        # TODO use sklearn metrics (maybe as parameter)
        # Problem: we want to have a per-instance error, so using sklearn metrics can only be done per row.
        # + Have to check if NaN handling still works as expected
        num_instances = self.x.shape[0]
        # num_outputs   = self.y.shape[1]
        ind_oob_instances = list(set(range(num_instances)) - set(self.ind_sub_instances))
        x_oob = self.x.iloc[ind_oob_instances, :].reset_index(drop=True)
        y_oob = self.y.iloc[ind_oob_instances, :].reset_index(drop=True) # True labels
        y_pred = self.predict(x_oob, single_label=(self.task=="classification"))

        if self.task == "regression":
            oob_error = (y_oob.values - y_pred)**2
        elif self.task == "classification":
            y_oob = np.argmax(y_oob.values, axis=1) # Map prototypes to actual classes
            y_oob -= 1 # '?' is class 0 for y_oob, but unused for y_pred
            oob_error = (y_oob != y_pred).reshape(len(ind_oob_instances), oob_error_dataframe.shape[1])
            oob_error[y_oob == -1] = np.NaN # Missing values set similarly to regression case
            #from sklearn.metrics import confusion_matrix
            #confusion_matrix(y_oob, y_pred) # First column is zero, because unwanted part!

        oob_error_dataframe[ind_oob_instances,:] += oob_error

    def compute_itb_error(self, itb_error_dataframe):
        """Computes the "in-the-bag" error. See L{compute_oob_error}."""
        # TODO implement changes made in compute_oob_error (once that's ready with metric as argument)

        ind_itb_instances = list(set(self.ind_sub_instances)) # Only grab unique instances
        x_itb = self.x.iloc[ind_itb_instances, :].reset_index(drop=True)
        y_itb = self.y.iloc[ind_itb_instances, :].reset_index(drop=True) # True labels
        y_pred = self.predict(x_itb, single_label=(self.task=="classification"))
        
        if self.task == "regression":
            # MSE 
            itb_error = ((y_itb-y_pred)**2)
        elif self.task == "classification":
            # Mapping the true labels to format similar to y_pred
            y_itb = np.argmax(y_itb.values, axis=1) # Map prototypes to actual classes
            y_itb -= 1 # '?' is class 0 for y_itb, but unused for y_pred
            oob_error = (y_itb != y_pred).reshape(len(ind_itb_instances), itb_error_dataframe.shape[1])
            oob_error[y_itb == -1] = np.NaN # Missing values set similarly to regression case
            
        itb_error_dataframe.loc[ind_itb_instances,:] += itb_error

