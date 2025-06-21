import numpy as np
import pandas as pd
from pct.tree.heuristic.Heuristic import Heuristic5

class NumericHeuristic5(Heuristic5):
    def __init__(self, name, min_instances, x, y): #, numerical_attributes
        super().__init__(name, min_instances, x, y)
        #self.numerical_attributes = numerical_attributes  # added to handle numerical attributes after item_type in yahoo data
        self.measure_heuristic = self.squared_error_total  # Use squared error for splitting

        # Compute statistics
        self.sum_t, self.sum2_t, self.n_t = self.compute_statistics()
        
    def find_split_groups(self, split_item):
        """Determine which users belong to Lovers, Haters, and Unknowns based on the rating of the split item."""
        ratings = self.x[split_item].values  # Extract column values from DataFrame

        indices_L = np.where(ratings == 1)[0]  # Lovers: rating >= 50
        indices_H = np.where((ratings == 0.1) & (ratings > 0))[0]  # Haters: rating <= 50 and nonzero
        return indices_L, indices_H

    def compute_statistics_for_groups(self, indices_L, indices_H):
        """Compute statistics (sum, squared sum, count) for each group based on user indices."""
        #rating_cols = self.numerical_attributes #  # Filter numeric rating columns (exclude itemtype columns)
        #x_data = self.x[rating_cols] # Compute stats only for rating columns
        x_data = self.x

        # Compute for Lovers group
        sum_L = np.nansum(x_data.iloc[indices_L, :].values, axis=0)
        sum2_L = np.nansum(x_data.iloc[indices_L, :].values ** 2, axis=0)
        n_L = np.count_nonzero((x_data.iloc[indices_L, :].values != 0) & (~np.isnan(x_data.iloc[indices_L, :].values)), axis=0)

        # Compute for Haters group
        sum_H = np.nansum(x_data.iloc[indices_H, :].values, axis=0)
        sum2_H = np.nansum(x_data.iloc[indices_H, :].values ** 2, axis=0)
        n_H = np.count_nonzero((x_data.iloc[indices_H, :].values != 0) & (~np.isnan(x_data.iloc[indices_H, :].values)), axis=0)

        # Compute for Unknowns group (total - Lovers - Haters)
        sum_U = self.sum_t - sum_L - sum_H
        sum2_U = self.sum2_t - sum2_L - sum2_H
        n_U = self.n_t - n_L - n_H

        return sum_L, sum2_L, n_L, sum_H, sum2_H, n_H, sum_U, sum2_U, n_U

    def squared_error(self, sum_G, sum2_G, n_G):
        """Compute squared error for a given group."""
        squared_error = np.zeros_like(sum_G, dtype=float)  # Ensure float output
        mask = n_G > 0  # Avoid division by zero
        squared_error[mask] = sum2_G[mask] - (sum_G[mask] ** 2) / n_G[mask]
        return squared_error

    def squared_error_total(self, split_item):
        """Compute total squared error for the current split."""
        # Step 1: Get user groups based on split item
        indices_L, indices_H = self.find_split_groups(split_item)

        # Step 2: Compute statistics for each group
        sum_L, sum2_L, n_L, sum_H, sum2_H, n_H, sum_U, sum2_U, n_U = \
            self.compute_statistics_for_groups(indices_L, indices_H)

        # Step 3: Compute squared error for each group
        e_L = self.squared_error(sum_L, sum2_L, n_L)
        e_H = self.squared_error(sum_H, sum2_H, n_H)
        e_U = self.squared_error(sum_U, sum2_U, n_U)

        # Step 4: Compute total squared error (sum over all items)
        squared_error_total = np.sum(e_L + e_H + e_U)
        return squared_error_total

    def stop_criteria(self):
        """Check if the stopping criteria are met based on partition sizes."""
        partition_sizes = [len(self.n_t)]
        return super().stop_criteria(partition_sizes)
