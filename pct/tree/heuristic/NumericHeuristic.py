import numpy as np
from pct.tree.heuristic.Heuristic import Heuristic5



class NumericHeuristic5(Heuristic5):
    def __init__(self, name, weights, min_instances, instance_weights, x, y):
        # Initialize the heuristic with given parameters and compute statistics
        super().__init__(name, weights, min_instances, instance_weights, x, y)
        self.measure_heuristic = self.squared_error_total  # Use squared error for splitting

        # Compute statistics
        self.sum_t, self.sum2_t, self.n_t = self.compute_statistics()
        self.sum_L, self.sum2_L, self.n_L, self.sum_H, self.sum2_H, self.n_H, self.sum_U, self.sum2_U, self.n_U = self.compute_statistics_for_groups()

    def squared_error(self):
        """Compute squared error for each group: Lovers, Haters, and Unknowns"""
    
        # Lovers Group Error
        e_L = np.zeros_like(self.sum_L)  # Initialize with zeros
        mask_L = self.n_L > 0  # Find non-zero elements in the Lovers group    
        e_L[mask_L] = self.sum2_L[mask_L] - (self.sum_L[mask_L] ** 2) / self.n_L[mask_L]
        
        # Haters Group Error
        e_H = np.zeros_like(self.sum_H)  # Initialize with zeros
        mask_H = self.n_H > 0  # Find non-zero elements in the Haters group
        e_H[mask_H] = self.sum2_H[mask_H] - (self.sum_H[mask_H] ** 2) / self.n_H[mask_H]
        
        # Unknowns Group Error (using subtraction as requested)
        e_U = np.zeros_like(self.sum_U)  # Initialize with zeros
        mask_U = self.n_U > 0  # Find non-zero elements in the Unknowns group
        e_U[mask_U] = self.sum2_U[mask_U] - (self.sum_U[mask_U] ** 2) / self.n_U[mask_U]
        
        # Return squared error arrays for each group
        return e_L, e_H, e_U

    def compute_statistics_for_groups(self):
        """Compute statistics for each group: Lovers, Haters, and Unknowns"""
        
        # Lovers: ratings >= 4
        mask_L = self.x >= 4
        sum_L = np.nansum(self.x * mask_L, axis=0)  # Sum of ratings for Lovers, per column
        sum2_L = np.nansum(self.x**2 * mask_L, axis=0)  # Sum of squares for Lovers, per column
        n_L = np.count_nonzero(mask_L, axis=0)  # Number of ratings in Lovers group, per column

        # Haters: ratings <= 3
        mask_H = self.x <= 3
        sum_H = np.nansum(self.x * mask_H, axis=0)  # Sum of ratings for Haters, per column
        sum2_H = np.nansum(self.x**2 * mask_H, axis=0)  # Sum of squares for Haters, per column
        n_H = np.count_nonzero(mask_H, axis=0)  # Number of ratings in Haters group, per column

        # Unknowns: ratings that are neither Lovers nor Haters
        sum_U = self.sum_t - sum_L - sum_H  # Total sum minus Lovers and Haters
        sum2_U = self.sum2_t - sum2_L - sum2_H  # Total sum of squares minus Lovers and Haters
        n_U = self.n_t - n_L - n_H  # Total count minus Lovers and Haters

        return sum_L, sum2_L, n_L, sum_H, sum2_H, n_H, sum_U, sum2_U, n_U

    def squared_error_total(self):
        """Return the total squared error based on Lovers, Haters, and Unknowns groups"""
        e_L, e_H, e_U = self.squared_error()  # Get squared errors for each group
        squared_error = e_L + e_H + e_U  # Sum squared errors for all groups (item-wise)
        return squared_error

    def stop_criteria(self):
        """Check if the stopping criteria are met based on partition sizes"""
        partition_sizes = [
            len(self.n_L) if isinstance(self.n_L, (list, np.ndarray)) else self.n_L,
            len(self.n_H) if isinstance(self.n_H, (list, np.ndarray)) else self.n_H,
            len(self.n_U) if isinstance(self.n_U, (list, np.ndarray)) else self.n_U
        ]
        return super().stop_criteria(partition_sizes)
