import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder
from pct.tree.heuristic.NumericHeuristic import NumericHeuristic
from pct.tree.heuristic.CategoricalHeuristic import CategoricalHeuristic
from pct.tree.splitter.splitter import Splitter
import pct.tree.utils as utils

# idea: make a measure_criterion (num/cat) function here, and make 
# heuristic a property of this class. then we only need to
# overwrite measure_criterion oh yeah and of course init the 
# heuristic_attrib here

class SemiSupervisedSplitter(Splitter):
    def __init__(
        self, min_instances, 
        numerical_attributes, categorical_attributes, 
        inout_weight,
        ftest=0.01,
        target_weights=None,
        feature_weights=None
    ):
        super().__init__(
            min_instances=min_instances, 
            numerical_attributes=numerical_attributes, 
            categorical_attributes=categorical_attributes, 
            ftest=ftest, 
            target_weights=target_weights
        )
        self.inout_weight = inout_weight
        self.feature_weights = feature_weights
        if feature_weights is None:
            feature_weights = np.ones(len(numerical_attributes) + len(categorical_attributes))

    def numerical_split(self, x, y, instance_weights, attribute_name, criteria, return_index):
        """Finds the best split for the given numerical attribute."""
        # Sort by the values of the attribute (for efficient splits!)
        x = x.sort_values(attribute_name, ascending=False)

        # Extract the attribute
        x_attribute = x[attribute_name]
        y = y.loc[x_attribute.index].values
        instance_weights = instance_weights.loc[x_attribute.index].values

        # Construct the heuristic object
        heuristic_target = NumericHeuristic(
            self.criterion, self.target_weights, self.min_instances, self.ftest, 
            instance_weights, x_attribute, y
        )
        catprot = utils.create_prototypes(x[self.categorical_attributes])
        catprot.index = x.index
        x = x[self.numerical_attributes]
        # x = x.drop(attribute_name, axis=1) # THIS LINE
        x = pd.concat( (x, catprot), axis=1 )
        x = x.values
        # TODO make utils.get_target_weights(x) a class instance // argument
        heuristic_attrib = NumericHeuristic(
            self.criterion, utils.get_target_weights(x), self.min_instances, self.ftest, 
            instance_weights, x_attribute, x
        )
        heuristic_target.update_missing()
        heuristic_attrib.update_missing()
        heuristic_target.variance_current_criterion()
        heuristic_attrib.variance_current_criterion()
       
        # Inner loop, specific to numerical splits
        highest_criterion = -np.inf
        new_criterion = -np.inf
        splitting_value = None
        previous_value = None 
        for i,v in enumerate(x_attribute):
            if v != previous_value and not np.isnan(v):
                if heuristic_target.stop_criteria():
                    new_criterion = -np.inf 
                else:    
                    ### CHANGE HERE
                    new_criterion = (
                        self.inout_weight * heuristic_target.measure_heuristic()
                        + (1 - self.inout_weight) * max(0, heuristic_attrib.measure_heuristic())
                    )
            if new_criterion > highest_criterion:
                highest_criterion = new_criterion
                splitting_value = v
            previous_value = v
            heuristic_target.update(i)
            heuristic_attrib.update(i)
        criteria[return_index] = [highest_criterion,splitting_value]
        # criteria[attribute_name] = [highest_criterion, splitting_value]

    def categorical_split(self, x, y, instance_weights, attribute_name, criteria, return_index):
        """Finds the best (in a greedy sense) split for the given categorical attribute."""

        # Extract the attribute
        x_attribute = x[attribute_name]
        y = y.values
        instance_weights = instance_weights.values

        # Construct the heuristic objects
        heuristic_target = CategoricalHeuristic(
            self.criterion, self.target_weights, self.min_instances, self.ftest, 
            instance_weights, x_attribute, y
        )
        # x = x.drop(attribute_name, axis=1) # THIS LINE
        catprot = utils.create_prototypes(x[self.categorical_attributes])
        catprot.index = x.index
        x = pd.concat( (x[self.numerical_attributes], catprot), axis=1 )
        x = x.values
        # TODO make utils.get_target_weights(x) a class instance // argument
        heuristic_attrib = CategoricalHeuristic(
            self.criterion, utils.get_target_weights(x), self.min_instances, self.ftest, 
            instance_weights, x_attribute, x
        )
        heuristic_target.update(x_attribute, y, instance_weights.reshape(-1))
        heuristic_attrib.update(x_attribute, x, instance_weights.reshape(-1))
        heuristic_target.variance_current_criterion()
        heuristic_attrib.variance_current_criterion()
        
        # Inner loop, specific to categorical splits
        nb_attributes_values = len(heuristic_target.indexing)
        attributes_available = list(range(nb_attributes_values))
        attributes_included = []
        
        highest_criterion = -np.inf
        cardinality = 0
        best_attribute_index = 0
        possible_values = []
        # Greedy search: best attribute first, then add every other attribute and check if its better etc
        while best_attribute_index != -1 and cardinality + 1 < nb_attributes_values:        
            best_attribute_index = -1
            for i in attributes_available:
                if heuristic_target.stop_criteria(attributes_included,i):
                    new_criterion = -np.inf 
                else:        
                    ### CHANGE HERE
                    new_criterion = ( 
                        self.inout_weight       * heuristic_target.heuristic_subset(i, attributes_included) +
                        (1 - self.inout_weight) * heuristic_attrib.heuristic_subset(i, attributes_included)
                    )
                    if new_criterion > highest_criterion:
                        highest_criterion = new_criterion
                        best_attribute_index = i
            if best_attribute_index!=-1:
                attributes_available.remove(best_attribute_index)
                attributes_included.append(best_attribute_index)
                possible_values.append(heuristic_target.get_attribute_name(best_attribute_index))
            cardinality+=1 
        criteria[return_index] = [highest_criterion, possible_values]
        # criteria[attribute_name] = [highest_criterion, possible_values]
