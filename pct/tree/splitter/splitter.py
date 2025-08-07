import numpy as np
# from splitterThread import parallelSplitter
# from threading import Thread
# from pct.tree.ftest.ftest import FTest
# from pct.tree.heuristic.NumericHeuristic import NumericHeuristic
# from pct.tree.heuristic.CategoricalHeuristic import CategoricalHeuristic
from pct.tree.heuristic.Heuristic import Heuristic5
from pct.tree.heuristic.NumericHeuristic_yahoo import NumericHeuristic5


class Splitter:
    def __init__(
        self,
        min_instances,
        numerical_attributes,
        categorical_attributes,
        # ftest,
        # target_weights # Mostly used for HMC
    ):
        """Constructs this splitter object with the given parameters.
        
        @param min_instances: The minimum number of (weighted) samples in a leaf node (stopping criterion).
        @param numerical_attributes  : Iterable holding the names of the numerical   splitting attributes.
        @param categorical_attributes: Iterable holding the nmaes of the categorical splitting attributes.
        @param ftest: The p-value (in [0,1]) used in the F-test for the statistical significance of a split.
        @param target_weights: The weights corresponding to the target variables.
        """
        self.criterion = "Squared Error"
        self.worst_performance = -1
        print("Initializing Splitter...")
        # self.ftest = FTest(ftest)

        self.min_instances = min_instances
        self.numerical_attributes = numerical_attributes
        self.categorical_attributes = categorical_attributes
        # self.target_weights = target_weights


    

    ## This function is implemented by Camille
    def find_best_split_item(self, x, y): #, instance_weights
        """Finds the most informative item to split users based on squared error reduction.

        @param x: User-item interaction matrix (rows = users, columns = items).
        @param y: Target variable (ratings).
        @param instance_weights: Weights assigned to each user.
        @return: (best_item_id, criterion_value) - Item with highest variance in ratings.
        """

        best_item = None
        lowest_error = np.inf  # We want to minimize squared error

        # Iterate over each item (column) in the user-item matrix
        for item_id in x.columns:
            item_ratings = x[item_id].values.reshape(-1, 1)  # Convert column to 2D array
           

            # Skip items with too few ratings (ensure enough users per item)
            if np.count_nonzero(~np.isnan(item_ratings)) < self.min_instances:
                continue  
            
            # Compute squared error for this item using NumericHeuristic5
            heuristic = NumericHeuristic5(
                self.criterion, self.min_instances, x, y
            )
            
            total_error = heuristic.squared_error_total(item_id)  # Compute total squared error
            # print item_id and total_error
            # print(item_id, total_error)
            
      

            
            # Select the item with the lowest squared error
            if total_error < lowest_error:
                best_item = item_id
                lowest_error = total_error
            
            
        # If no valid item is found, return None (no split possible)
        return best_item, lowest_error if best_item is not None else (-np.inf)
       


    