import numpy as np
# from splitterThread import parallelSplitter
# from threading import Thread
from pct.tree.ftest.ftest import FTest
from pct.tree.heuristic.NumericHeuristic import NumericHeuristic
from pct.tree.heuristic.CategoricalHeuristic import CategoricalHeuristic

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

        # # Alternative way of doing this (one less argument)
        # criteria = {}
        # for attr in set(x.columns) - set(self.categorical_attributes)
        #     self.numerical_split(x, y, instance_weights, attr, criteria)
        # for attr in set(x.columns) - set(self.numerical_attributes)
        #     self.categorical_split(x, y, instance_weights, attr, criteria)
        # best_attr = max( criteria.keys(), key=(lambda key: criteria[key][0]) )
        # return best_attr, (*criteria[best_attr])

        possible_attributes = x.columns
        # print (x.shape)
        criteria = [[None,None]] * len(possible_attributes)
        for i, attribute in enumerate(possible_attributes):
            if attribute in self.numerical_attributes:
                self.numerical_split(x, y, instance_weights, attribute, criteria,i)
            elif attribute in self.categorical_attributes:
                self.categorical_split(x, y, instance_weights, attribute, criteria,i)
        highest_split = np.argmax([a[0] for a in criteria])
        # print([criteria[i] if ~np.isneginf(criteria[i][0]) else [] for i in range(len(criteria))])
        return possible_attributes[highest_split], criteria[highest_split][0], criteria[highest_split][1]

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
                    # print(f"Didn't measure heuristic for i={i}  \t(out of total {len(x_attribute)})")
                    new_criterion = -np.inf 
                else:    
                    # print ("measure ")
                    new_criterion = heuristic.measure_heuristic()
            if new_criterion > highest_criterion:
                highest_criterion = new_criterion
                splitting_value = v
            previous_value = v
            heuristic.update(i) # 24/07/'20 At the moment this is the most expensive call
        criteria[return_index] = [highest_criterion,splitting_value]
        # criteria[attribute_name] = [highest_criterion, splitting_value]

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
        
        highest_criterion = -np.inf # Contains the best heuristic value so far
        cardinality = 0             # Used for checking if we have only one variable left
        best_attribute_index = 0    # Used for checking if there's any variable left worth adding
        possible_values = []        # Contains the attribute values in the split

        while best_attribute_index != -1 and cardinality + 1 < nb_attributes_values:        
            best_attribute_index = -1
            # Search the best attribute that can be added at this moment
            for i in attributes_available:
                if heuristic.stop_criteria(attributes_included,i):
                    # Don't split on this attribute if it would violate the min_instances restriction
                    new_criterion = -np.inf 
                else:
                    # Otherwise, check if including this variable in the split is better than the
                    # currently highest criterion (from a previous while- or for-loop)
                    new_criterion = heuristic.heuristic_subset(i,attributes_included)
                    if new_criterion > highest_criterion:
                        highest_criterion = new_criterion
                        best_attribute_index = i
            if best_attribute_index != -1:
                # We found a better split, so update the bookkeeping lists
                attributes_available.remove(best_attribute_index)
                attributes_included.append(best_attribute_index)
                possible_values.append(heuristic.get_attribute_name(best_attribute_index))
            cardinality += 1
        criteria[return_index] = [highest_criterion, possible_values]
        # criteria[attribute_name] = [highest_criterion, possible_values]
