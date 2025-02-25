import numpy as np
from pct.tree.heuristic.Heuristic import Heuristic5

class NumericHeuristic5(Heuristic5):
    def __init__(
        self,
        name,
        weights,
        min_instances,
        ftest,
        instance_weights,
        x,  # Sorted descriptive variable (user-item ratings matrix) on which we are splitting
        y
    ):
        super().__init__(name, weights, min_instances, ftest, instance_weights, x, y)
        self.measure_heuristic = self.squared_error_total  # Focus on squared error for splitting


    def squared_error(self, sum_L, sum_H, sum_U, sum2_L,sum2_H, sum2_U, n_L, n_H, n_U):
        """Compute the squared error for each group: Lovers, Haters, and Unknowns."""
        e_L = sum2_L - (np.sum(sum_L) ** 2) / n_L if n_L > 0 else 0
        e_H = sum2_H - (np.sum(sum_H) ** 2) / n_H if n_H > 0 else 0
        e_U = sum2_U - (np.sum(sum_U) ** 2) / n_U if n_U > 0 else 0
        return e_L + e_H + e_U

    def compute_statistics_for_groups(self):
        """Compute statistics for the three groups: Lovers, Haters, and Unknowns."""
        
        # Use vectorized operations to classify the ratings
        lovers = self.x >= 4  # Lovers: ratings >= 4
        haters = self.x <= 3  # Haters: ratings <= 3
        unknowns = np.isnan(self.x) | (self.x == 0)  # Unknowns: NaN or 0

        # Calculate the sums for Lovers and Haters groups
        sum_L = np.sum(self.x * lovers, axis=0)  # Sum for Lovers
        sum_H = np.sum(self.x * haters, axis=0)  # Sum for Haters

        sum2_L = np.sum(self.x**2 * lovers, axis=0)  # Sum of squares for Lovers
        sum2_H = np.sum(self.x**2 * haters, axis=0)  # Sum of squares for Haters

        n_L = np.count_nonzero(lovers, axis=0)  # Number of Lovers
        n_H = np.count_nonzero(haters, axis=0)  # Number of Haters

        # Now calculate the total sums and sum squares
        sum_t, sum2_t, n_t = self.compute_statistics()

        # Calculate Unknowns group by subtraction from total statistics
        sum_U = sum_t - sum_L - sum_H
        sum2_U = sum2_t - sum2_L - sum2_H
        n_U = n_t - n_L - n_H

        return sum_L, sum2_L, n_L, sum_H, sum2_H, n_H, sum_U, sum2_U, n_U
        
      def squared_error_total(self):
          """Returns the total squared error based on Lovers, Haters, and Unknowns groups."""
          sum_L, sum2_L, n_L, sum_H, sum2_H, n_H, sum_U, sum2_U, n_U = self.compute_statistics_for_groups(sum_t, sum2_t, n_t)

          # Calculate squared error using the method from Heuristic5
          squared_error = self.squared_error(sum_L, sum_H, sum_U, sum2_L,sum2_H, sum2_U, n_L, n_H, n_U)
          return squared_error

      def stop_criteria(self, n_L, n_H, n_U):
          partition_sizes = [n_L, n_H, n_U]
          return super().stop_criteria(partition_sizes)

     
