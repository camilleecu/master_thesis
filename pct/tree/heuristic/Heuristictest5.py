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

    def compute_statistics(self, threshold):
        """Compute global statistics for the entire dataset, using thresholding."""
        # Apply thresholding to the ratings matrix (same as thresholded matrix in NumericHeuristic5)
        rating_matrix_thresholded = np.where(self.x > threshold, 1, np.where(self.x > 0, -1, 0))

        # Now calculate sum, sum2, and n for the thresholded matrix
        sum_t = np.sum(rating_matrix_thresholded, axis=0)
        sum2_t = np.sum(rating_matrix_thresholded**2, axis=0)
        n_t = np.count_nonzero(rating_matrix_thresholded != 0, axis=0)  # Number of non-zero ratings
        return sum_t, sum2_t, n_t

    
    def stop_criteria(self, partition_sizes):
        return any(size < self.min_instances for size in partition_sizes)

   
