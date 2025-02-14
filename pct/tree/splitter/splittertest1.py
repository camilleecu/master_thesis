import numpy as np
# from splitterThread import parallelSplitter
# from threading import Thread
from pct.tree.ftest.ftest import FTest
from pct.tree.heuristic.NumericHeuristictest1 import NumericHeuristic1  
from pct.tree.heuristic.CategoricalHeuristictest1 import CategoricalHeuristic1  

class Splitter1:
    def __init__(
        self,
        min_instances,
        numerical_attributes,
        categorical_attributes,
        ftest,
        target_weights  # Mostly used for HMC
    ):
        """Constructs this splitter object with the given parameters."""
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
        criteria = [[None, None]] * len(possible_attributes)
        
        for i, attribute in enumerate(possible_attributes):  
            if attribute in self.numerical_attributes:
                self.numerical_split(x, y, instance_weights, attribute, criteria, i)
            elif attribute in self.categorical_attributes:
                self.categorical_split(x, y, instance_weights, attribute, criteria, i)
        
        highest_split = np.argmax([a[0] for a in criteria])
        return possible_attributes[highest_split], criteria[highest_split][0], criteria[highest_split][1]

    def numerical_split(self, x, y, instance_weights, attribute_name, criteria, return_index):
        """Finds the best split for the given numerical attribute."""
        # Sorting by attribute values for efficient splits!
        x = x.sort_values(attribute_name, ascending=False)
        x_attribute = x[attribute_name]
        y = y.loc[x_attribute.index].values
        instance_weights = instance_weights.loc[x_attribute.index].values

        # Initialize heuristic
        splitting_value1 = 50  # Example threshold for LIKE / DISLIKE
        splitting_value2 = 0   # Example threshold for UNKNOWN

        # Pass splitting_value1 and splitting_value2 to NumericHeuristic1
        heuristic = NumericHeuristic1(self.criterion, self.target_weights, self.min_instances, self.ftest, 
                                      instance_weights, x_attribute, y, splitting_value1, splitting_value2)

        heuristic.update_missing()
        heuristic.variance_current_criterion()

        highest_criterion = -np.inf
        previous_value = None 

        # Now consider 'LIKE', 'DISLIKE', and 'UNKNOWN' groups
        for i, v in enumerate(x_attribute):
            if v > splitting_value1:
                group = 'LIKE'  
            elif v <= splitting_value1 and v > splitting_value2:
                group = 'DISLIKE'  
            else:
                group = 'UNKNOWN'  

            heuristic.update(i)

        # Return the best split thresholds
        criteria[return_index] = [highest_criterion, [splitting_value1, splitting_value2]]


    
    def categorical_split(self, x, y, instance_weights, attribute_name, criteria, return_index):
        """Finds the best (in a greedy sense) split for the given categorical attribute."""
        x_attribute = x[attribute_name]
        y = y.values
        instance_weights = instance_weights.values

        # Construct the heuristic object
        heuristic = CategoricalHeuristic1(self.criterion, self.target_weights, self.min_instances, self.ftest, instance_weights, x_attribute, y)
        heuristic.update(x_attribute, y, instance_weights.reshape(-1))
        heuristic.variance_current_criterion()

        highest_criterion = -np.inf
        splitting_value1 = None  # First threshold for splitting
        splitting_value2 = None  # Second threshold for splitting

        # Loop over the unique category values and calculate the heuristic for splitting them
        for category_value in x_attribute.unique():
            new_criterion = heuristic.heuristic_subset(category_value)
            if new_criterion > highest_criterion:
                highest_criterion = new_criterion
                splitting_value1 = category_value 

        criteria[return_index] = [highest_criterion, [splitting_value1, splitting_value2]]  
