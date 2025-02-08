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
		x, # Sorted (!) descriptive variable on which we are splitting
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

	# def update_gini(self, params):
	# 	self.partition_size += 1
	# 	self.partition+=params["y_current"]

	def update_variance(self, i):
		"""Updates the variance using the i'th instance."""
		weight = self.instance_weights[i]
		target = self.y[i]
		self.n_pos  += weight
		self.k_pos  += weight * ~self.target_is_nan[i]
		# temp = weight*target
		# self.sv_pos += temp
		# self.ss_pos += temp * target
		self.sv_pos += weight * target
		self.ss_pos += weight * target**2

	def debug_total(self):
		print (self.n_tot)
		print (self.k_tot)
		print (self.sv_tot)
		print (self.ss_tot)
		print ("------")

	def update_variance_missing(self):
		"""Updates the variance using the instances where the feature is missing."""
		ind_missing = np.isnan(self.x)
		weights = self.instance_weights[ind_missing]
		targets = self.y[ind_missing]
		missing_n_tot  = np.sum( weights )
		missing_k_tot  = np.sum( weights *~self.target_is_nan[ind_missing], axis=0 )
		missing_sv_tot = np.sum( weights * targets            , axis=0 )
		missing_ss_tot = np.sum( weights * targets ** 2       , axis=0 )

		self.n_tot  -= missing_n_tot
		self.k_tot  -= missing_k_tot
		self.sv_tot -= missing_sv_tot 
		self.ss_tot -= missing_ss_tot


	# def gini_total(self,current_criterion):
	# 	total_annotations = np.sum(self.total_classes)
	# 	positive_annotations = np.sum(self.partition,axis=0)
	# 	positive_gini = ((1.0 - np.sum([(a/positive_annotations) ** 2 for a in self.partition]))  * self.target_weights * positive_annotations) / self.targets_number 
	# 	difference = total_annotations - positive_annotations
	# 	gini_difference = ((1.0 - np.sum([((v - self.partition[i]) / difference ) ** 2 for i,v in enumerate(self.total_classes)]) ) * self.target_weights * difference) / self.targets_number 
	# 	return self.ftest.test(total_annotations, self.current_criterion,  gini_difference + positive_gini)

	def variance_total(self):
		"""Returns the total variance if it was accepted by the F-test (-np.inf otherwise)."""
		variance = self.variance( self.n_pos, self.k_pos, self.sv_pos, self.ss_pos )
		return self.ftest.test(self.n_tot, self.current_criterion,  variance)

	def stop_criteria(self):
		"""True if and only if the current partition violates the min_instances restriction."""
		partition_size = self.n_pos
		return partition_size < self.min_instances  or (self.n_tot - partition_size) < self.min_instances
