import numpy as np
from pct.tree.ftest.ftest import FTest
from pct.tree.heuristic.NumericHeuristictest3 import NumericHeuristic3  # Updated to NumericHeuristic3
from pct.tree.heuristic.CategoricalHeuristic import CategoricalHeuristic

import numpy as np
from pct.tree.ftest.ftest import FTest
from pct.tree.heuristic.NumericHeuristictest3 import NumericHeuristic3  # Updated to NumericHeuristic3
from pct.tree.heuristic.CategoricalHeuristic import CategoricalHeuristic

class Splitter3:
    def __init__(
        self,
        min_instances,
        numerical_attributes,
        categorical_attributes,
        ftest,
        target_weights
    ):
        """Constructs this splitter object with the given parameters."""
        self.criterion = "variance"
        self.worst_performance = -1
        self.ftest = ftest

        self.min_instances = min_instances
        self.numerical_attributes = numerical_attributes
        self.categorical_attributes = categorical_attributes
        self.target_weights = target_weights

    def find_split(self, x, y, instance_weights):
        """Finds the best split in the given dataset."""
        possible_attributes = x.columns
        criteria = [[None, None]] * len(possible_attributes)
        
        # Loop through each attribute to find the best split
        for i, attribute in enumerate(possible_attributes):
            print(f"Trying to split on attribute: {attribute}")  # Output the attribute being considered
            if attribute in self.numerical_attributes:
                self.numerical_split(x, y, instance_weights, attribute, criteria, i)
            elif attribute in self.categorical_attributes:
                self.categorical_split(x, y, instance_weights, attribute, criteria, i)

        highest_split = np.argmax([a[0] for a in criteria])
        best_attribute = possible_attributes[highest_split]
        best_criterion = criteria[highest_split][0]
        best_split = criteria[highest_split][1]

        # Output the best split found
        print(f"Best split found on attribute: {best_attribute} with criterion: {best_criterion}")
        return best_attribute, best_criterion, best_split

    def numerical_split(self, x, y, instance_weights, attribute_name, criteria, return_index):
        """Finds the best split for the given numerical attribute."""
        print(f"Performing numerical split on {attribute_name}")  # Output when performing numerical split

        x_attribute = x[attribute_name]
        y = y.loc[x_attribute.index].values
        instance_weights = instance_weights.loc[x_attribute.index].values

        # Calculate dynamic thresholds based on y (ratings)
        threshold_1, threshold_2 = self.calculate_thresholds(y)

        # Output thresholds
        print(f"Thresholds for splitting: {threshold_1}, {threshold_2}")

        # Initialize the heuristic with the thresholds
        heuristic = NumericHeuristic3(
            self.criterion, 
            self.target_weights, 
            self.min_instances, 
            self.ftest, 
            instance_weights, 
            x_attribute, 
            y,
            split_thresholds=[threshold_1, threshold_2]  # Pass the thresholds
        )
        heuristic.update_missing()
        heuristic.variance_current_criterion()

        highest_criterion = -np.inf
        splitting_value = None
        previous_value = None 

        for i, v in enumerate(x_attribute):
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

        # Output the best split found
        print(f"Best numerical split found with criterion: {highest_criterion} at value: {splitting_value}")

        # Core modification for ternary split based on y (ratings):
        left_split = (y < threshold_1) | np.isnan(y)  # Left split: y < threshold_1 or missing
        middle_split = (y >= threshold_1) & (y < threshold_2)  # Middle split: threshold_1 <= y < threshold_2
        right_split = (y >= threshold_2)  # Right split: y >= threshold_2

        # Output partition sizes
        print(f"Left split size: {np.sum(left_split)}, Middle split size: {np.sum(middle_split)}, Right split size: {np.sum(right_split)}")

        # Update split information in criteria
        criteria[return_index] = [highest_criterion, [left_split, middle_split, right_split]]

    def calculate_thresholds(self, y):
        """Calculate the thresholds for ternary split based on the provided ratings."""
        threshold_1 = np.percentile(y, 33)  # 33rd percentile for the first threshold (middle split starts)
        threshold_2 = np.percentile(y, 66)  # 66th percentile for the second threshold (right split starts)
        return threshold_1, threshold_2
    
    def categorical_split(self, x, y, instance_weights, attribute_name, criteria, return_index):
        """Finds the best (in a greedy sense) split for the given categorical attribute."""
        print(f"Performing categorical split on {attribute_name}")  # Output when performing categorical split
        
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

        highest_criterion = -np.inf
        cardinality = 0
        best_attribute_index = 0
        possible_values = []

        while best_attribute_index != -1 and cardinality + 1 < len(heuristic.indexing):        
            best_attribute_index = -1
            for i in range(len(heuristic.indexing)):
                if heuristic.stop_criteria([], i):
                    new_criterion = -np.inf 
                else:
                    new_criterion = heuristic.heuristic_subset(i, [])
                    if new_criterion > highest_criterion:
                        highest_criterion = new_criterion
                        best_attribute_index = i

            if best_attribute_index != -1:
                possible_values.append(heuristic.get_attribute_name(best_attribute_index))
            cardinality += 1
        
        # Divide the possible values into three sets for the ternary split
        thresholds = possible_values[:len(possible_values) // 3]
        left_split = x_attribute.isin(thresholds[0])
        right_split = x_attribute.isin(thresholds[1])
        middle_split = ~left_split & ~right_split

        # Output partition sizes
        print(f"Left split size: {np.sum(left_split)}, Middle split size: {np.sum(middle_split)}, Right split size: {np.sum(right_split)}")

        # Update split information in criteria
        criteria[return_index] = [highest_criterion, [left_split, middle_split, right_split]]

    def categorical_split(self, x, y, instance_weights, attribute_name, criteria, return_index):
        """Finds the best (in a greedy sense) split for the given categorical attribute."""
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

        highest_criterion = -np.inf
        cardinality = 0
        best_attribute_index = 0
        possible_values = []

        while best_attribute_index != -1 and cardinality + 1 < len(heuristic.indexing):        
            best_attribute_index = -1
            for i in range(len(heuristic.indexing)):
                if heuristic.stop_criteria([], i):
                    new_criterion = -np.inf 
                else:
                    new_criterion = heuristic.heuristic_subset(i, [])
                    if new_criterion > highest_criterion:
                        highest_criterion = new_criterion
                        best_attribute_index = i

            if best_attribute_index != -1:
                possible_values.append(heuristic.get_attribute_name(best_attribute_index))
            cardinality += 1
        
        # Divide the possible values into three sets for the ternary split
        thresholds = possible_values[:len(possible_values) // 3]
        left_split = x_attribute.isin(thresholds[0])
        right_split = x_attribute.isin(thresholds[1])
        middle_split = ~left_split & ~right_split

        # Update split information in criteria
        criteria[return_index] = [highest_criterion, [left_split, middle_split, right_split]]
