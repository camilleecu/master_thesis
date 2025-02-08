import numpy as np
import pandas as pd
import random
from pct.forest.forest import RandomForest
from pct.forest.bi_forest_tree import RandomBiClusteringTree
from multiprocessing import Pool

class RandomBiClusteringForest(RandomForest):

    def fit(
        self, x, y, vert_features,
        target_weights=None,  # Mostly used for HMC
        num_sub_instances=-1,
        num_sub_features=-1, # Number of features to use for each tree
        num_sub_V_features=-1, # Number of vertical features to use for each tree
        n_jobs=None
    ):
        """Fits this random forest of PBCTs to the given data.

        @param vert_features: The feature representation to use for the targets.
        @param num_sub_V_features: Number of vertical features to use for each split.
            (default is the square root of the number of the number of columns in vert_features)
        """
        # Input validation
        num_V_features = vert_features.shape[1]
        if num_sub_V_features == -1:
            num_sub_V_features = round(np.sqrt(num_V_features))
        
        # Bookkeeping
        self.vert_features = vert_features
        self.num_sub_V_features = num_sub_V_features

        # Proceed as usual
        return super().fit(x, y, target_weights, num_sub_instances, num_sub_features)

    def make_tree(self):
        """Initializes and returns a PBCT-random forest tree to be used in this forest."""
        return RandomBiClusteringTree(
            self.min_instances, self.ftest,
            self.num_sub_instances, self.num_sub_features, self.num_sub_V_features,
            random_state=random.random()
        )

    def fit_tree(self, i):
        """Fits and returns the i-th tree in this forest."""
        return self.trees[i].fit(
            self.x, self.y, self.vert_features, self.target_weights
        )

    @property
    def feature_importances_(self):
        """Returns the aggregated feature heuristic, averaged over all trees in the forest."""
        features = pd.concat( (self.x, self.vert_features) )
        feat_imp = pd.DataFrame(np.zeros(len(features.columns)), index=features.columns)
        for i in range(self.num_trees):
            self.trees[i].get_feature_importances(feat_imp)
        feat_imp /= self.num_trees
        return feat_imp
