import numpy as np
from pct.tree.heuristic.Heuristic import Heuristic

class NumericHeuristic2(Heuristic):
    def __init__(
        self,
        name,
        weights,
        min_instances,
        ftest,
        instance_weights,
        x,  # Sorted (!) descriptive variable on which we are splitting
        y,
        splitting_value1=None,  # Modified
        splitting_value2=None   # Modified
    ):
        super().__init__(name, weights, min_instances, ftest, instance_weights, x, y)
        self.splitting_value1 = splitting_value1  # Modified
        self.splitting_value2 = splitting_value2  # Modified
        self.measure_heuristic = self.squared_error_total  # Changed to squared error
        self.update = self.update_squares_error  # Changed to squared error
        self.update_missing = self.update_squares_error_missing  # Changed to squared error

        # Initialize for three groups (LIKE, DISLIKE, UNKNOWN)
        self.n_pos = 0
        self.k_pos = np.zeros(self.targets_number)
        self.sv_pos = np.zeros(self.targets_number)
        self.ss_pos = np.zeros(self.targets_number)

        self.n_dislike = 0
        self.k_dislike = np.zeros(self.targets_number)
        self.sv_dislike = np.zeros(self.targets_number)
        self.ss_dislike = np.zeros(self.targets_number)

        self.n_unknown = 0
        self.k_unknown = np.zeros(self.targets_number)
        self.sv_unknown = np.zeros(self.targets_number)
        self.ss_unknown = np.zeros(self.targets_number)

        self.y[self.target_is_nan] = 0

        # Initialize current_criterion
        self.current_criterion = -np.inf  # Start with a default invalid criterion value

    def update_squares_error(self, i):
        """Updates the squared error using the i'th instance."""
        weight = self.instance_weights[i]
        target = self.y[i]
        value = self.x.iloc[i]  # Get the value of the splitting variable (x)

        # Determine the group for this instance
        if value > self.splitting_value1:  # LIKE group
            self.n_pos += weight
            self.k_pos += weight * ~self.target_is_nan[i]
            self.sv_pos += weight * target
            self.ss_pos += weight * target**2
        elif value <= self.splitting_value1 and value > self.splitting_value2:  # DISLIKE group
            self.n_dislike += weight
            self.k_dislike += weight * ~self.target_is_nan[i]
            self.sv_dislike += weight * target
            self.ss_dislike += weight * target**2
        else:  # UNKNOWN group
            self.n_unknown += weight
            self.k_unknown += weight * ~self.target_is_nan[i]
            self.sv_unknown += weight * target
            self.ss_unknown += weight * target**2

    def update_squares_error_missing(self):
        """Updates the squared error using the instances where the feature is missing."""
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

    def squared_error(self, n_pos, k_pos, sv_pos, ss_pos):
        """Returns the squared error for the given partition variables."""
        error_pos = self.__squared_error_inner(n_pos, k_pos, sv_pos, ss_pos)
        error_dislike = self.__squared_error_inner(self.n_dislike, self.k_dislike, self.sv_dislike, self.ss_dislike)
        error_unknown = self.__squared_error_inner(self.n_unknown, self.k_unknown, self.sv_unknown, self.ss_unknown)
        return error_pos + error_dislike + error_unknown  # Only three groups now

    def __squared_error_inner(self, n, k, sv, ss):
        """Calculates the squared error for a partition."""
        if n == 0:
            return 0  # If partition is empty, return 0
        mean = sv / k  # Mean value
        return ss - 2 * mean * sv + mean * mean * k

    def squared_error_total(self):
        """Returns the total squared error if it was accepted by the F-test (-np.inf otherwise)."""
        error = self.squared_error(self.n_pos, self.k_pos, self.sv_pos, self.ss_pos)
        self.current_criterion = error  # Update current_criterion with the latest error
        return self.ftest.test(self.n_tot, self.current_criterion, error)

    def stop_criteria(self):
        """True if and only if the current partition violates the min_instances restriction."""
        partition_size = self.n_pos + self.n_dislike + self.n_unknown  # Total size for three groups
        return partition_size < self.min_instances or (self.n_tot - partition_size) < self.min_instances

    def get_dislike(self, n_pos, k_pos, sv_pos, ss_pos):
        """Returns the partition variables for the DISLIKE group."""
        return (
            self.n_dislike,
            self.k_dislike,
            self.sv_dislike,
            self.ss_dislike
        )

    def get_unknown(self, n_pos, k_pos, sv_pos, ss_pos):
        """Returns the partition variables for the UNKNOWN group."""
        return (
            self.n_unknown,
            self.k_unknown,
            self.sv_unknown,
            self.ss_unknown
        )
