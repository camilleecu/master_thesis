from pct.tree.splitter.splitter import Splitter

class RandomForestSplitter(Splitter):

    def __init__(
            self, min_instances, numerical_attributes, categorical_attributes, 
            ftest, target_weights, num_sub_features, rng_engine
        ):
        """Constructs this splitter object with the given parameters.
        
        @param num_sub_features: Number of randomly selected features to use at each split.
        @param rng_engine: Random number generator for feature selection.
        @type rng_engine: Python built-in random object
        """
        super().__init__(
            min_instances, numerical_attributes, categorical_attributes, ftest, target_weights
        )
        self.num_sub_features = num_sub_features
        self.random = rng_engine

    def find_split(self, x, y, instance_weights):
        """Finds the best split in the given dataset, using only a subset of the features in x."""
        # Select a random subset of the features
        num_features = len(x.columns)
        self.ind_sub_features = self.random.sample(range(num_features) , self.num_sub_features)
        x_sub = x.iloc[:,self.ind_sub_features]

        # Proceed as usual
        return super().find_split(x_sub, y, instance_weights)
