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

    def compute_statistics_for_groups(self, sum_t, sum2_t, n_t):
        """Compute statistics for the three groups: Lovers, Haters, and Unknowns."""
        sum_L, sum2_L, n_L = np.zeros_like(sum_t), np.zeros_like(sum2_t), np.zeros_like(n_t)
        sum_H, sum2_H, n_H = np.zeros_like(sum_t), np.zeros_like(sum2_t), np.zeros_like(n_t)
        sum_U, sum2_U, n_U = np.zeros_like(sum_t), np.zeros_like(sum2_t), np.zeros_like(n_t)
        
        # Loop through all users to classify them into Lovers, Haters, or Unknowns
        for i in range(self.x.shape[0]):  # Iterate over each user
            for j in range(self.x.shape[1]):  # Iterate over each item
                rating = self.x[i, j]
                # Classify the user based on their rating
                if rating >= 4:  # Lovers
                    sum_L[j] += rating
                    sum2_L[j] += rating**2
                    n_L[j] += 1
                elif rating <= 3:  # Haters
                    sum_H[j] += rating
                    sum2_H[j] += rating**2
                    n_H[j] += 1
                else: # Unknowns
                    sum_U[j] += rating
                    sum2_U[j] += rating**2
                    n_U[j] += 1

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

     
