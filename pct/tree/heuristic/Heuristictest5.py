import numpy as np

# # -----------------------------------------------
# # It's cheaper to have the warnings than to manually check for np.nan.
# # See https://github.com/scikit-learn/scikit-learn/issues/5870#issuecomment-159409438
# # for an explanation why.
# # -----------------------------------------------
# # Ignoring warnings
# import warnings
# class ignore_warnings(object):
#     def __init__(self, f):
#         self.f = f

#     def __call__(self, args):
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             self.f(args)
# # -----------------------------------------------

import numpy as np

class Heuristic5:
    def __init__(self, name, weights, min_instances, ftest, instance_weights, x, y):
        self.name = name
        self.weights = weights
        self.min_instances = min_instances
        self.ftest = ftest
        self.instance_weights = instance_weights
        self.x = x  # User-item ratings matrix
        self.y = y  # Target variable (ratings or other)
        self.n_tot = np.sum(self.instance_weights)  # Total weight
        self.current_criterion = None  # Placeholder for the splitting criterion

    def compute_statistics(self):
        """Compute global statistics for the entire dataset."""
        sum_t = np.sum(self.x, axis=0)  # Total ratings for each item
        sum2_t = np.sum(self.x**2, axis=0)  # Sum of squared ratings for each item
        n_t = np.count_nonzero(~np.isnan(self.x), axis=0)  # Number of users who rated each item
        return sum_t, sum2_t, n_t

     def stop_criteria(self):
        """True if and only if the current partition violates the min_instances restriction."""
        partition_size = self.n_tot
        return partition_size < self.min_instances or (self.n_tot - partition_size) < self.min_instances

   