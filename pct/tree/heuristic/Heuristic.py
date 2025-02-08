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

class Heuristic():
    """Class containing tools for splitting heuristics in the PCT."""

    initial_weight = 1.0
    missing_value_categorical = '?'
    missing_value_numeric = np.nan #### not used in code, left here for references 

    def __init__(
        self,
        name,			  # Name of the criterion used ("variance" or "gini")
        target_weights,   # For normalizing the ranges of the features
        min_instances,    # Minimal number of instances in a leaf
        ftest,            # FTest object
        instance_weights, # For instances that have passed through a node with a missing feature
        x,                # Descriptive variable on which we are splitting
        y,                # Target variable(s)
    ):
        self.name = name
        self.target_weights = target_weights
        self.targets_number = y.shape[1]
        self.min_instances = min_instances
        self.ftest = ftest
        self.instance_weights = instance_weights
        self.variance_list = np.zeros(len(self.target_weights)) # Preallocated variance vector

        self.n_tot = np.sum(instance_weights)
        self.k_tot = np.sum(
            np.repeat(instance_weights, repeats=y.shape[1], axis=1),
            axis=0,
            where=(~np.isnan(y))
        )
        self.sv_tot = np.nansum(self.instance_weights * y, axis=0)
        self.ss_tot = np.nansum(self.instance_weights * y**2, axis=0)

        self.x = x
        self.y = y
        self.instance_weights = instance_weights
        self.target_is_nan = np.isnan(self.y)
        # self.y[self.target_is_nan] = 0 # Incorrect for CategoricalHeuristic (because overwrites splitter-y)
        # self.y = np.nan_to_num(y)      # Has negative impact on performance

    def variance_current_criterion(self):
        """Recomputes the current variance (using internal variables)."""
        # sv_tot = Total summed values
        # ss_tot = Total sum of squares
        # n_tot  = Total weight
        # k_tot  = Total weight of instances with non-missing targets
        self.current_criterion = np.mean(
            ( self.ss_tot * (self.n_tot - 1)/(self.k_tot - 1)
             - self.n_tot * (self.sv_tot / self.k_tot)**2 )
            *self.target_weights
        )

    # def gini_single(self, values):
    # 	values = np.sum(values,axis=0)
    # 	annotations = np.sum(values,axis=0)
    # 	gini = ((1.0 - np.sum([(a/annotations) ** 2 for a in values]))  * self.target_weights * annotations) / self.targets_number
    # 	return gini

    def variance(self, n_pos, k_pos, sv_pos, ss_pos):
        """Returns the variance for the given partition variables.

        @param n_pos : Sum of instance weights for each partition.
        @param k_pos : Sum of instance weights where target is not missing for each partition.
        @param sv_pos: Summed values for each partition.
        @param ss_pos: Sum of squares for each partition.
        """
        n_neg, k_neg, sv_neg, ss_neg = self.get_neg(n_pos, k_pos, sv_pos, ss_pos)
        # Could remove the "abs" if we are sure that k_pos >= 1
        if (any(abs(k_pos - 1) < 1e-9) or any(abs(k_neg - 1) < 1e-9)
            or (n_pos < 1e-9) or (n_neg < 1e-9)):
            return np.inf
        return (
            self.__var_inner(n_pos, k_pos, sv_pos, ss_pos) + 
            self.__var_inner(n_neg, k_neg, sv_neg, ss_neg)
        )
        # # Older version, doesn't work in some special cases (i.e. k = n = 1)
        # if any(k_pos == 1) or any(k_neg == 1) or any(k_pos == 0) or any(k_neg == 0):
        # 	return np.inf # (checking for infinities saves some operations)
        # return np.mean(
        # 	self.target_weights *
        # 	( ss_pos * (n_pos-1)/(k_pos-1) - n_pos*(sv_pos/k_pos)**2
        # 	+ ss_neg * (n_neg-1)/(k_neg-1) - n_neg*(sv_neg/k_neg)**2 )
        # )

    # @ignore_warnings
    def __var_inner(self, n, k, sv, ss):
        # if k == n: var = weights * (ss - sv² / n)
        # if k != n: var = weights * (ss * (n - 1)/(k - 1) - n * (sv / k)²)
        np.multiply(
            ss - sv**2 / n, self.target_weights, 
            out = self.variance_list, 
            where = (k == n) 
        )
        np.multiply(
            ss * (n-1)/(k-1) - n*(sv/k)**2, self.target_weights, 
            out = self.variance_list, 
            where = (k != n) 
        )
        return np.mean(self.variance_list)

    def get_neg(self, n_pos, k_pos, sv_pos, ss_pos):
        """Returns the partition variables for the other side of the split."""
        return (
            self.n_tot  - n_pos,
            self.k_tot  - k_pos,
            self.sv_tot - sv_pos,
            self.ss_tot - ss_pos
        )

