import numpy as np
import pandas as pd
from pct.tree.tree import Tree
from pct.tree.splitter.semi_splitter import SemiSupervisedSplitter

class SemiSupervisedTree(Tree):
    """Class of semi-supervised PCTs for multi-target regression.
    
    Source:
        Jurica Levatic, Dragi Kocev, Michelangelo Ceci, Saso Dzeroski, 
        Semi-supervised trees for multi-target regression, Information 
        Sciences (2018), doi:10.1016/j.ins.2018.03.033
    """
    def __init__(self, inout_weight, min_instances=5, ftest=0.01, feature_weights=None):
        """Constructs a new semi-supervised PCT.

        @param inout_weight: 
            The weight (∈ [0,1]) used for output (=target) versus input (=descriptors).
        @param feature_weights:
            The normalized importance scores used as weights for the feature variances.
        """
        super().__init__(min_instances=min_instances, ftest=ftest)
        self.inout_weight = inout_weight
        self.feature_weights = feature_weights
        # --> If not provided, then in the fit function maybe train a random forest on the
        # labeled part of the data and extract the feature weights and then normalize them

        # Unlabeled instances are given by the logical vector (np.sum(~np.isnan(y), axis=1) == 0)

    def make_splitter(self):
        """Constructs a splitter object for this semi-supervised PCT."""
        return SemiSupervisedSplitter(
            min_instances=self.min_instances,
            numerical_attributes=self.numerical_attributes, 
            categorical_attributes=self.categorical_attributes,
            inout_weight=self.inout_weight,
            ftest=self.ftest,
            target_weights=self.target_weights,
            feature_weights=self.feature_weights
        )
