import numpy as np
from pct.tree.heuristic.Heuristictest3 import Heuristic3

class NumericHeuristic3(Heuristic3):
    def __init__(
        self,
        name,
        weights,
        min_instances,
        ftest,
        instance_weights,
        x,  # Sorted (!) descriptive variable on which we are splitting
        y,
        split_thresholds  # Add this parameter to receive the thresholds for the split
    ):
        super().__init__(name, weights, min_instances, ftest, instance_weights, x, y)
        self.measure_heuristic = self.variance_total
        self.update = self.update_variance
        self.update_missing = self.update_variance_missing

        # Initialize arrays for three partitions: left, middle, and right
        self.n_pos = np.zeros(3)  # Number of instances in each partition (left, middle, right)
        self.k_pos = np.zeros((3, self.targets_number))  # Target counts for each partition
        self.sv_pos = np.zeros(3)  # Sum of the targets in each partition
        self.ss_pos = np.zeros(3)  # Sum of squared targets in each partition

        self.y[self.target_is_nan] = 0  # Handle missing target values
        self.split_thresholds = split_thresholds  # Store the split thresholds

    def update_variance(self, i):
        """Updates the variance using the i'th instance for each of the three partitions."""
        weight = self.instance_weights[i]
        target = self.y[i]

        # Determine which partition this instance belongs to based on the thresholds
        if target < self.split_thresholds[0] or np.isnan(target):  # left partition
            idx = 0
        elif target >= self.split_thresholds[1]:  # right partition
            idx = 2
        else:  # middle partition
            idx = 1

        # Update the variance for the appropriate partition
        self.n_pos[idx] += weight
        self.k_pos[idx] += weight * ~self.target_is_nan[i]
        self.sv_pos[idx] += weight * target
        self.ss_pos[idx] += weight * target**2

    def update_variance_missing(self):
        """Updates the variance using the instances where the feature is missing for each partition."""
        ind_missing = np.isnan(self.x)
        weights = self.instance_weights[ind_missing]
        targets = self.y[ind_missing]

        missing_n_tot = np.sum(weights)
        missing_k_tot = np.sum(weights * ~self.target_is_nan[ind_missing], axis=0)
        missing_sv_tot = np.sum(weights * targets, axis=0)
        missing_ss_tot = np.sum(weights * targets ** 2, axis=0)

        # Subtract missing values from the total counts for each partition
        self.n_tot -= missing_n_tot
        self.k_tot -= missing_k_tot
        self.sv_tot -= missing_sv_tot
        self.ss_tot -= missing_ss_tot

    def variance_total(self):
        """Returns the total variance if it was accepted by the F-test (-np.inf otherwise)."""
        # Calculate the variance for each of the three partitions
        variances = np.zeros(3)
        for i in range(3):
            # Calculate variance considering positive and negative splits within each partition
            n_pos, k_pos, sv_pos, ss_pos = self.n_pos[i], self.k_pos[i], self.sv_pos[i], self.ss_pos[i]
            n_neg, k_neg, sv_neg, ss_neg = self.get_neg(n_pos, k_pos, sv_pos, ss_pos)
            variances[i] = self.variance(n_pos, k_pos, sv_pos, ss_pos) + self.variance(n_neg, k_neg, sv_neg, ss_neg)

        # Combine the variances from each partition
        total_variance = np.sum(variances)
        return self.ftest.test(self.n_tot, self.current_criterion, total_variance)

    def stop_criteria(self):
        """True if and only if the current partition violates the min_instances restriction."""
        # Ensure the stopping criterion accounts for the size of all three partitions
        partition_size = np.sum(self.n_pos)
        return partition_size < self.min_instances or (self.n_tot - partition_size) < self.min_instances
