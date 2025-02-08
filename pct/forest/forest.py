import numpy as np
import pandas as pd
import random
from pct.forest.forest_tree import RandomForestTree
from multiprocessing import Pool
from sklearn.base import BaseEstimator

class RandomForest(BaseEstimator):
    """Class containing functionality relevant to random forests of PCTs.

    Interesting functions for extending to forests of other trees are
    L{make_tree} and L{fit_tree}.
    """
    VERBOSE = False

    def __init__(
        self, *, # Enforces use of keyword arguments 
        min_instances=5, 
        ftest=0.01,
        num_trees=100,
        num_sub_instances=None,
        num_sub_features =None,
        random_state=None,
    ):
        """Constructs a new random forest of PCTs.

        @param min_instances: See tree/tree.py/Tree/__init__
        @param ftest        : See tree/tree.py/Tree/__init__
        @param num_trees    : The amount of trees to plant in the forest
        @param random_state : Set to an integer seed if you want reproducible results.
        """
        self.min_instances = min_instances
        self.ftest = ftest
        self.num_trees = num_trees
        self.trees = [None for i in range(num_trees)] # List of references to the PCTs
        random.seed(random_state) # If None, uses some pseudoRNG seed
        self.random_state = random_state
        self.num_sub_instances = num_sub_instances
        self.num_sub_features  = num_sub_features
    
    def fit(
        self, x, y, 
        target_weights=None,  # Mostly used for HMC
        num_sub_instances=-1,
        num_sub_features =-1,
        n_jobs=None
    ):
        """Fits this random forest to the given data.

        @param x                : See tree/tree.py/Tree/fit. L{Tree#fit}
        @param y                : See tree/tree.py/Tree/fit.
        @param target_weights   : See tree/tree.py/Tree/fit.
        @param num_sub_instances: Number of instances to bootstrap for each tree.
            (default is the total number of instances i.e. #rows in x)
        @param num_sub_features: Number of features to use for each split.
            (default is the square root of the number of features i.e. #columns of x)
        @param n_jobs: How much multiprocessing to use. None is sequential computations,
            -1 automatically selects a number of processes (=Python interpreters) based
            on the available CPU cores, or supplying a positive integer fixes the number
            of processes to that number.
        @return: This RandomForest object, trained on the given dataset.
        @see Tree#fit
        """
        if self.num_sub_features:
            num_sub_features = self.num_sub_features
        if self.num_sub_instances:
            num_sub_instances = self.num_sub_instances

        # Input validation
        num_instances = x.shape[0]
        num_features  = x.shape[1]
        if num_sub_instances == -1:
            num_sub_instances = num_instances
        if num_sub_features == -1:
            num_sub_features  = round(np.sqrt(num_features)) # General rule of thumb
        assert 1 <= num_sub_features  <= num_features , "Invalid number of features"
        assert 1 <= num_sub_instances <= num_instances, "Invalid number of instances"

        # Some bookkeeping and convenience variables
        self.x = pd.DataFrame(x)
        self.y = pd.DataFrame(y)
        self.num_sub_instances = num_sub_instances
        self.num_sub_features  = num_sub_features
        self.target_weights = target_weights

        # Planting the trees in the forest
        # (careful adding multiprocessing here, might mess up the RNG)
        for i in range(self.num_trees):
            self.trees[i] = self.make_tree()
            self.trees[i].VERBOSE = RandomForest.VERBOSE

        # Fitting the trees to the given dataset
        if n_jobs is None:
            for i in range(self.num_trees):
                self.fit_tree(i)
        
        else:
            # Initialize a pool with the given number of jobs
            # NOTE sklearn uses joblib: https://joblib.readthedocs.io/en/latest/
            if n_jobs == -1:
                pool = Pool()
            else:
                assert n_jobs > 0, "Number of processes must be positive!"
                pool = Pool(processes=n_jobs)
            
            # Fit the trees to the data
            for i in range(self.num_trees):
                # Assigning to the return value here (in fit, we do "return self")
                # because otherwise the tree is destroyed after the process finishes
                # (as this runs in a different Python interpreter)
                self.trees[i] = pool.apply_async(
                    func=self.fit_tree, args=[i]
                    ).get()

            # Close off the processes
            pool.close()
            pool.join()

        return self

    def make_tree(self):
        """Initializes and returns a tree to be used in this forest."""
        return RandomForestTree(
            self.min_instances, self.ftest,
            self.num_sub_instances, self.num_sub_features,
            random_state=random.random()
        )
    
    def fit_tree(self, i):
        """Fits and returns the i-th tree in this forest."""
        return self.trees[i].fit(
            self.x, self.y, target_weights=self.target_weights
        )

    def predict(self, x, single_label=False):
        """Returns the average prediction scores of all trees in the forest."""
        # TODO Maybe return as pandas dataframe? (columns = attribute names)
        predictions = self.trees[0].predict(x)
        for i in range(1, self.num_trees):
            predictions += self.trees[i].predict(x)
        predictions /= self.num_trees
        return np.argmax(predictions, axis=1) if single_label else predictions

    @property
    def feature_importances_(self):
        """Returns the aggregated feature heuristic, averaged over all trees in the forest."""
        feat_imp  = pd.DataFrame(np.zeros(len(self.x.columns)), index=self.x.columns)
        for i in range(self.num_trees):
            self.trees[i].get_feature_importances(feat_imp)
        feat_imp /= self.num_trees
        return feat_imp

    @property
    def oob_error_(self): 
        """Returns the out-of-bag error, averaged over all trees in the forest."""
        # NOTE sklearn's API specifies oob_score_ instead of error
        #      also, sklearn already computes this in the fit function
        # TODO also change names of these functions in forestTree? 
        #      (not properties though because of passed dataframe)
        # TODO what about trees without any OOB samples?
        oob_error = np.zeros(self.y.shape)
        for i in range(self.num_trees):
            self.trees[i].compute_oob_error(oob_error) # (pass by reference)
        oob_error = pd.DataFrame(oob_error, index=self.y.index, columns=self.y.columns)
        oob_error = oob_error.mean(skipna=True) # Aggregate the instances
        oob_error /= self.num_trees
        return oob_error
    
    @property
    def itb_error_(self):
        """Returns the 'in-the-bag' error, averaged over all trees in the forest."""
        itb_error = pd.DataFrame(np.zeros(self.y.shape), index=self.y.index, columns=self.y.columns)
        for i in range(self.num_trees):
            self.trees[i].compute_itb_error(itb_error) # (dataframe is passed by reference)
        itb_error = pd.DataFrame(itb_error, index=self.y.index, columns=self.y.columns)
        itb_error = itb_error.mean(skipna=True)
        itb_error /= self.num_trees
        return itb_error

    def decision_path(self, x):
        """Returns the decision path in this forest for each instance in the given dataset.
        
        The decision path of a forest consists of the decision paths of each of its trees
        stacked onto one another.
        """
        return np.hstack(
            tuple([self.trees[i].decision_path(x) for i in range(self.num_trees)])
        )
