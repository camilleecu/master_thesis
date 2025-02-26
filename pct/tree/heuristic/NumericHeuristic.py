import numpy as np
from pct.tree.heuristic.Heuristic import Heuristic

class NumericHeuristic(Heuristic):
	def __init__(
		self,
		name,
		weights,
		min_instances,
		ftest,
		instance_weights,
		x,  # Sorted (!) descriptive variable on which we are splitting
		y
	):
		super().__init__(name, weights, min_instances, ftest, instance_weights, x, y)
		self.measure_heuristic = self.variance_total
		# self.measure_current_heuristic = self.variance_current_criterion
		self.update = self.update_variance
		self.update_missing = self.update_variance_missing

		self.n_pos = 0
		self.k_pos = np.zeros(self.targets_number)
		self.sv_pos = np.zeros(self.targets_number)
		self.ss_pos = np.zeros(self.targets_number)

		self.y[self.target_is_nan] = 0

		# Track rating categories separately
		self.like_n = 0
		self.dislike_n = 0
		self.unknown_n = 0

		self.like_sv = 0
		self.dislike_sv = 0
		self.unknown_sv = 0

		self.like_ss = 0
		self.dislike_ss = 0
		self.unknown_ss = 0

	def variance(self, n, sv, ss):
		"""Compute variance using the standard formula: (sum of squares - (sum squared / count)) / count"""
		if n == 0:
			return 0  # Prevent division by zero
		return (ss - (sv ** 2) / n) / n

	def update_variance(self, i):
		# Updates variance separately for like, dislike, and unknown ratings.
		weight = self.instance_weights[i]
		rating = self.y[i]  # User rating (1-100)

		if np.isnan(rating):  # Unknown ratings
			self.unknown_n += weight
			self.unknown_sv += weight * 50  # Default midpoint rating
			self.unknown_ss += weight * 50 ** 2
		elif rating > 50:  # Like category
			self.like_n += weight
			self.like_sv += weight * rating
			self.like_ss += weight * rating ** 2
		else:  # Dislike category
			self.dislike_n += weight
			self.dislike_sv += weight * rating
			self.dislike_ss += weight * rating ** 2

	def debug_total(self):
		print(self.n_tot)
		print(self.k_tot)
		print(self.sv_tot)
		print(self.ss_tot)
		print("------")

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

	# def gini_total(self, current_criterion):
	#     total_annotations = np.sum(self.total_classes)
	#     positive_annotations = np.sum(self.partition, axis=0)
	#     positive_gini = ((1.0 - np.sum([(a / positive_annotations) ** 2 for a in self.partition])) * self.target_weights * positive_annotations) / self.targets_number
	#     difference = total_annotations - positive_annotations
	#     gini_difference = ((1.0 - np.sum([((v - self.partition[i]) / difference) ** 2 for i, v in enumerate(self.total_classes)])) * self.target_weights * difference) / self.targets_number
	#     return self.ftest.test(total_annotations, self.current_criterion, gini_difference + positive_gini)

	def variance_total(self):
		# Computes variance for ratings within the three groups (like, dislike, unknown).

		# Compute variance for each rating category
		var_like = self.variance(self.like_n, self.like_sv, self.like_ss) if self.like_n > 0 else 0
		var_dislike = self.variance(self.dislike_n, self.dislike_sv, self.dislike_ss) if self.dislike_n > 0 else 0
		var_unknown = self.variance(self.unknown_n, self.unknown_sv, self.unknown_ss) if self.unknown_n > 0 else 0

		# Compute weighted total variance
		total_variance = var_like + var_dislike + var_unknown
		return self.ftest.test(self.n_tot, self.current_criterion, total_variance)

	# def stop_criteria(self):
	#     # True if and only if the current partition violates the min_instances restriction.
	#     partition_size = self.n_pos
	#     return partition_size < self.min_instances or (self.n_tot - partition_size) < self.min_instances

	def stop_criteria(self):
		# Stops splitting if any rating group has too few users.
		return (
			self.like_n < self.min_instances or
			self.dislike_n < self.min_instances or
			self.unknown_n < self.min_instances
		)
