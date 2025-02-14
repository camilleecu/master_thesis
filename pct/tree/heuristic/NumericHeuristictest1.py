import numpy as np
from pct.tree.heuristic.Heuristic import Heuristic

class NumericHeuristic1(Heuristic):
    def __init__(
        self,
        name,
        weights,
        min_instances,
        ftest,
        instance_weights,
        x,  # Sorted (!) descriptive variable on which we are splitting
        y,
        splitting_value1=None,  # Modifoed
        splitting_value2=None   # Modifoed
    ):
        super().__init__(name, weights, min_instances, ftest, instance_weights, x, y)
        self.splitting_value1 = splitting_value1  # Modifoed
        self.splitting_value2 = splitting_value2  # Modifoed
        self.measure_heuristic = self.variance_total
        self.update = self.update_variance
        self.update_missing = self.update_variance_missing

        self.n_pos = 0
        self.k_pos = np.zeros(self.targets_number)
        self.sv_pos = np.zeros(self.targets_number)
        self.ss_pos = np.zeros(self.targets_number)

        # Initialize for three groups (LIKE, DISLIKE, UNKNOWN)
        self.n_dislike = 0
        self.k_dislike = np.zeros(self.targets_number)
        self.sv_dislike = np.zeros(self.targets_number)
        self.ss_dislike = np.zeros(self.targets_number)

        self.n_unknown = 0
        self.k_unknown = np.zeros(self.targets_number)
        self.sv_unknown = np.zeros(self.targets_number)
        self.ss_unknown = np.zeros(self.targets_number)

        self.y[self.target_is_nan] = 0

    def update_variance(self, i):
        """Updates the variance using the i'th instance."""
        weight = self.instance_weights[i]
        target = self.y[i]
        value = self.x.iloc[i]  # Get the value of the splitting variable (x)

        # Determine the group for this instance
        if value > self.splitting_value1:  # modified: LIKE group
            self.n_pos += weight
            self.k_pos += weight * ~self.target_is_nan[i]
            self.sv_pos += weight * target
            self.ss_pos += weight * target**2
        elif value <= self.splitting_value1 and value > self.splitting_value2:  # modified: DISLIKE group
            self.n_dislike += weight  # modified
            self.k_dislike += weight * ~self.target_is_nan[i]  # modified
            self.sv_dislike += weight * target  # modified
            self.ss_dislike += weight * target**2  # modified
        else:  # modified: UNKNOWN group
            self.n_unknown += weight  # modified
            self.k_unknown += weight * ~self.target_is_nan[i]  # modified
            self.sv_unknown += weight * target  # modified
            self.ss_unknown += weight * target**2  # modified

    def update_variance_missing(self):
        """Updates the variance using the instances where the feature is missing."""
        ind_missing = np.isnan(self.x)
        weights = self.instance_weights[ind_missing]
        targets = self.y[ind_missing]
        missing_n_tot = np.sum(weights)
        missing_k_tot = np.sum(weights * ~self.target_is_nan[ind_missing], axis=0)
        missing_sv_tot = np.sum(weights * targets, axis=0)
        missing_ss_tot = np.sum(weights * targets ** 2, axis=0)

        self.n_tot -= missing_n_tot
        self.k_tot -= missing_k_tot
        self.sv_tot -= missing_sv_tot
        self.ss_tot -= missing_ss_tot

    def variance(self, n_pos, k_pos, sv_pos, ss_pos):
        """Returns the variance for the given partition variables.
        
        @param n_pos : Sum of instance weights for each partition.
        @param k_pos : Sum of instance weights where target is not missing for each partition.
        @param sv_pos: Summed values for each partition.
        @param ss_pos: Sum of squares for each partition.
        """
        n_dislike, k_dislike, sv_dislike, ss_dislike = self.get_dislike(n_pos, k_pos, sv_pos, ss_pos)  # modified
        n_unknown, k_unknown, sv_unknown, ss_unknown = self.get_unknown(n_pos, k_pos, sv_pos, ss_pos)  # modified
        n_neg, k_neg, sv_neg, ss_neg = self.get_neg(n_pos, k_pos, sv_pos, ss_pos)
        return (self.__var_inner(n_pos, k_pos, sv_pos, ss_pos) +
                self.__var_inner(n_dislike, k_dislike, sv_dislike, ss_dislike) +  # modified
                self.__var_inner(n_unknown, k_unknown, sv_unknown, ss_unknown) +  # modified
                self.__var_inner(n_neg, k_neg, sv_neg, ss_neg))

    def variance_total(self):
        """Returns the total variance if it was accepted by the F-test (-np.inf otherwise)."""
        variance = self.variance(self.n_pos, self.k_pos, self.sv_pos, self.ss_pos)
        return self.ftest.test(self.n_tot, self.current_criterion, variance)

    def stop_criteria(self):
        """True if and only if the current partition violates the min_instances restriction."""
        partition_size = self.n_pos + self.n_dislike + self.n_unknown  # modified: total size for three groups
        return partition_size < self.min_instances or (self.n_tot - partition_size) < self.min_instances

    def get_dislike(self, n_pos, k_pos, sv_pos, ss_pos):
        """Returns the partition variables for the DISLIKE group."""
        return (
            self.n_dislike,
            self.k_dislike,
            self.sv_dislike,
            self.ss_dislike
        )  # modified

    def get_unknown(self, n_pos, k_pos, sv_pos, ss_pos):
        """Returns the partition variables for the UNKNOWN group."""
        return (
            self.n_unknown,
            self.k_unknown,
            self.sv_unknown,
            self.ss_unknown
        )  # modified
