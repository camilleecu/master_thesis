import numpy as np
# from splitterThread import parallelSplitter
# from threading import Thread
# from pct.tree.ftest.ftest import FTest
# from pct.tree.heuristic.NumericHeuristic import NumericHeuristic
# from pct.tree.heuristic.CategoricalHeuristic import CategoricalHeuristic
from pct.tree.heuristic import NumericHeuristictest5

class Splitter:
    def __init__(
        self,
        min_instances,
        numerical_attributes,
        categorical_attributes,
        ftest,
        target_weights # Mostly used for HMC
    ):
        """Constructs this splitter object with the given parameters.
        
        @param min_instances: The minimum number of (weighted) samples in a leaf node (stopping criterion).
        @param numerical_attributes  : Iterable holding the names of the numerical   splitting attributes.
        @param categorical_attributes: Iterable holding the nmaes of the categorical splitting attributes.
        @param ftest: The p-value (in [0,1]) used in the F-test for the statistical significance of a split.
        @param target_weights: The weights corresponding to the target variables.
        """
        self.criterion = "variance"
        self.worst_performance = -1
        self.ftest = FTest(ftest)

        self.min_instances = min_instances
        self.numerical_attributes = numerical_attributes
        self.categorical_attributes = categorical_attributes
        self.target_weights = target_weights


    def find_split(self, x, y, instance_weights):
        """Finds the best split in the given dataset."""

        possible_attributes = x.columns
        criteria = [[None, None] for _ in range(len(possible_attributes))]
        for i, attribute in enumerate(possible_attributes):
            if attribute in self.numerical_attributes:
                self.numerical_split(x, y, instance_weights, attribute, criteria, i)
            elif attribute in self.categorical_attributes:
                self.categorical_split(x, y, instance_weights, attribute, criteria, i)
        highest_split = np.argmax([a[0] for a in criteria])
        return possible_attributes[highest_split], criteria[highest_split][0], criteria[highest_split][1]
    

    ## This function is implemented by Camille
    def find_best_split_item(self, x, y, instance_weights):
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
            heuristic = NumericHeuristictest5(
                self.criterion, self.target_weights, self.min_instances, self.ftest,
                instance_weights, item_ratings, y
            )
            
            total_error = heuristic.squared_error_total()  # Compute total squared error
            
            # Select the item with the lowest squared error
            if total_error < lowest_error:
                best_item = item_id
                lowest_error = total_error

        # If no valid item is found, return None (no split possible)
        return best_item, lowest_error if best_item is not None else (-np.inf)


    def numerical_split(self, x, y, instance_weights, attribute_name, criteria, return_index):
        """Finds the best split for the given numerical attribute."""
        # Sort by the values of the attribute (for efficient splits!)
        x = x.sort_values(attribute_name, ascending=False)

        # Extract the attribute
        x_attribute = x[attribute_name]
        y = y.loc[x_attribute.index].values
        instance_weights = instance_weights.loc[x_attribute.index].values

        # Construct the heuristic object
        heuristic = NumericHeuristic(
            self.criterion, 
            self.target_weights, 
            self.min_instances, 
            self.ftest, 
            instance_weights, 
            x_attribute, 
            y
        )
        heuristic.update_missing()
        heuristic.variance_current_criterion()
        
        # Inner loop, specific to numerical splits. We proceed by taking each value of x as a threshold
        # value, and checking where the best split would lie.
        highest_criterion = -np.inf
        new_criterion = -np.inf
        splitting_value = None
        previous_value = None 
        for i,v in enumerate(x_attribute):
            # Only proceed if x changes value (and is not missing)
            if v != previous_value and not np.isnan(v):
                if heuristic.stop_criteria():
                    new_criterion = -np.inf 
                else:    
                    new_criterion = heuristic.measure_heuristic()
            if new_criterion > highest_criterion:
                highest_criterion = new_criterion
                splitting_value = v
            previous_value = v
            heuristic.update(i)
        criteria[return_index] = [highest_criterion, splitting_value]

    def categorical_split(self, x, y, instance_weights, attribute_name, criteria, return_index):
        """Finds the best (in a greedy sense) split for the given categorical attribute."""

        # Extract the attribute
        x_attribute = x[attribute_name]
        y = y.values
        instance_weights = instance_weights.values

        # Construct the heuristic object
        heuristic = CategoricalHeuristic(
            self.criterion,
            self.target_weights, 
            self.min_instances, 
            self.ftest, 
            instance_weights, 
            x_attribute, 
            y
        )
        heuristic.update(x_attribute, y, instance_weights.reshape(-1))
        heuristic.variance_current_criterion()
        
        # Inner loop, specific to categorical splits. We proceed using a greedy search. First, the
        # best attribute is added. Then we check if adding any of the remaining attributes would
        # result in a higher heuristic value, and take the best one of these (if any). We proceed
        # iteratively.
        nb_attributes_values = len(heuristic.indexing)
        attributes_available = list(range(nb_attributes_values))
        attributes_included = []
        
        highest_criterion = -np.inf
        cardinality = 0
        best_attribute_index = 0
        possible_values = []

        while best_attribute_index != -1 and cardinality + 1 < nb_attributes_values:        
            best_attribute_index = -1
            for i in attributes_available:
                if heuristic.stop_criteria(attributes_included, i):
                    new_criterion = -np.inf 
                else:
                    new_criterion = heuristic.heuristic_subset(i, attributes_included)
                    if new_criterion > highest_criterion:
                        highest_criterion = new_criterion
                        best_attribute_index = i
            if best_attribute_index != -1:
                attributes_available.remove(best_attribute_index)
                attributes_included.append(best_attribute_index)
                possible_values.append(heuristic.get_attribute_name(best_attribute_index))
            cardinality += 1
        criteria[return_index] = [highest_criterion, possible_values]
