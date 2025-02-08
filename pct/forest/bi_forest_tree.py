import random
import numpy as np
import pandas as pd
import pct.tree.utils as utils
from pct.tree.bi_tree import BiClusteringTree
from pct.tree.node.node import Node
from pct.tree.splitter.splitter import Splitter
from pct.forest.forest_tree import RandomForestTree
from pct.forest.forest_splitter import RandomForestSplitter

class RandomBiClusteringTree(BiClusteringTree, RandomForestTree):
    
    def __init__(
            self, 
            min_instances, ftest, 
            num_sub_instances, num_sub_features, num_sub_V_features, 
            random_state=None
    ):
        """Initializes this PBCT as a RandomForestTree with a given number of vertical
        features to use at each split.
        """
        super().__init__(
            min_instances, ftest, num_sub_instances, num_sub_features, random_state
        )
        self.num_sub_V_features = num_sub_V_features

    def fit(self, x, y, vert_features, target_weights=None):
        """Fits this PCT to a bootstrapped version of the given data, using random
        feature selection at each split, in both horizontal and vertical features.
        """
        # NOTE Didn't change much w.r.t. RandomForestTree#fit
        # i.e. extra argument vert_features, and now using PBCT#fit instead of PCT#Fit

        self.task = utils.learning_task(y)

        # Sample the instances (with replacement) and delete the indices (because of duplicates)
        num_instances = x.shape[0]
        self.ind_sub_instances = self.random.choices(range(num_instances), k=self.num_sub_instances)
        x_sub = x.iloc[self.ind_sub_instances,:].reset_index(drop=True)
        y_sub = y.iloc[self.ind_sub_instances,:].reset_index(drop=True)

        # Proceed as usual (PBCT fit)
        super().fit(x_sub, y_sub, vert_features, target_weights=target_weights)

        # Reset the datasets (for later error calculation)
        self.x = x
        self.y = y
        if self.task == "classification":
            self.y = utils.create_prototypes(y)
        self.vert_features = vert_features
        return self

    def make_splitter(self):
        """Initializes the splitter objects for this PBCT forest."""
        self.H_splitter = RandomForestSplitter( # on [x, y]
            min_instances=self.min_instances,
            numerical_attributes=self.numerical_attributes, 
            categorical_attributes=self.categorical_attributes,
            ftest=self.ftest,
            target_weights=self.H_target_weights,
            num_sub_features=self.num_sub_features,
            rng_engine=self.random
        )
        self.V_splitter = RandomForestSplitter( # on [vert_features, y.transpose()]
            min_instances=self.min_instances,
            numerical_attributes=self.vert_features.select_dtypes(include=np.number).columns,
            categorical_attributes=self.vert_features.select_dtypes(exclude=np.number).columns,
            ftest=self.ftest,
            target_weights=self.V_target_weights,
            num_sub_features=self.num_sub_V_features,
            rng_engine=self.random
        )
        return None
