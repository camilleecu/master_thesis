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
        # Sort by the values of the attribute (for efficient splits!)
        x = x.sort_values(attribute_name, ascending=False)

        # Extract the attribute
        x_attribute = x[attribute_name]
        y = y.loc[x_attribute.index].values
        instance_weights = instance_weights.loc[x_attribute.index].values

        # Construct the heuristic object for ternary splits
        # We now calculate thresholds dynamically
        threshold_1, threshold_2 = self.calculate_thresholds(x_attribute)

        # Initialize the heuristic with the thresholds
        heuristic = NumericHeuristic3(  # Use NumericHeuristic3
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

        # Proceed with ternary split logic
        highest_criterion = -np.inf
        new_criterion = -np.inf
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

        # Core modification for ternary split
        left_split = x_attribute < threshold_1
        right_split = x_attribute > threshold_2
        middle_split = ~(left_split | right_split)

        # Update split information in criteria
        criteria[return_index] = [highest_criterion, [left_split, middle_split, right_split]]

    def calculate_thresholds(self, x_attribute):
        """Calculate the thresholds for ternary split."""
        # Here, we are using percentiles to calculate the thresholds
        threshold_1 = np.percentile(x_attribute, 33)  # 33rd percentile
        threshold_2 = np.percentile(x_attribute, 66)  # 66th percentile
        return threshold_1, threshold_2
    
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

        # Greedy search with changes for three-way split
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
        
        # Core modification for ternary split
        # Divide the possible values into three sets for the ternary split
        thresholds = possible_values[:len(possible_values) // 3]
        left_split = x_attribute.isin(thresholds[0])
        right_split = x_attribute.isin(thresholds[1])
        middle_split = ~left_split & ~right_split

        # Update split information in criteria
        criteria[return_index] = [highest_criterion, [left_split, middle_split, right_split]]
