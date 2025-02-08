import numpy as np
from pct.tree.heuristic.Heuristic import Heuristic
		
class CategoricalHeuristic(Heuristic):
	def __init__(
		self,
		name,
		weights,
		min_instances,
		ftest,
		instance_weights,
		x,
		y
	):
		super().__init__(name, weights, min_instances, ftest, instance_weights, x, y )
		self.measure_heuristic = self.variance_total
		# self.measure_current_heuristic = self.variance_current_criterion
		self.heuristic_subset = self.variance_subset
		self.update = self.update_variance

		self.indexing = {}
		self.partition_sizes = []
		self.partitions = []

		# Preallocate these other variables as well
		self.n_pos  = np.zeros(len(set(x)))
		self.k_pos  = np.zeros((len(set(x)), self.k_tot.size))
		self.sv_pos = np.zeros((len(set(x)), self.sv_tot.size))
		self.ss_pos = np.zeros((len(set(x)), self.ss_tot.size))

	# def update_gini(self, params):
	# 	for i,value in enumerate(params["values"]):
	# 		if value not in self.indexing :
	# 			self.indexing[value] = len(self.indexing)
	# 			self.partition_sizes.append(1.0)
	# 			self.partitions.append(params["targets"][i])
	# 		else:
	# 			self.partition_sizes[self.indexing[value]]+=1.0
	# 			self.partitions[self.indexing[value]] += params["targets"][i]

	def update_variance(self, x, y, instance_weights):
		"""
		Updates the variance using the given dataset. For categorical heuristics, we can
		do this all at once (as opposed to numeric heuristics) because the variables to
		be included in the split are computed in a different way (greedy search instead 
		of incremental thresholding).
		"""
		# TODO can use self.x, self.y, self.instance_weights instead actually?
		target = np.nan_to_num(y)
		# Not len(set(x))-1 because not all datasets have missing values
		self.partition_sizes = np.zeros(len(set(x) - set(self.missing_value_categorical)))

		# Build the n_pos, k_pos, sv_tot, ss_tot arrays
		ind_missing = (x == self.missing_value_categorical)
		instance_weights = instance_weights[..., np.newaxis]
		for i, value in enumerate( set(x) - set(self.missing_value_categorical) ):
			self.indexing[value] = len(self.indexing)
			ind = (~ind_missing) & (x == value)
			self.partition_sizes[i] = np.sum( instance_weights[ind], axis=0 )
			self.n_pos[i]  = np.sum( instance_weights[ind] )
			self.k_pos[i]  = np.sum( instance_weights[ind] * ~self.target_is_nan[ind], axis=0 )
			self.sv_pos[i] = np.sum( instance_weights[ind] * target[ind]             , axis=0 )
			self.ss_pos[i] = np.sum( instance_weights[ind] * target[ind] ** 2        , axis=0 )

		if any(ind_missing):
			self.n_tot  -= np.sum( instance_weights[ind_missing] )
			self.k_tot  -= np.sum( instance_weights[ind_missing] * ~self.target_is_nan[ind_missing], axis=0 )
			self.sv_tot -= np.sum( instance_weights[ind_missing] * target[ind_missing]             , axis=0 )
			self.ss_tot -= np.sum( instance_weights[ind_missing] * target[ind_missing] ** 2        , axis=0 )

	def debug_total(self):
		print (self.n_tot)
		print (self.k_tot)
		print (self.sv_tot)
		print (self.ss_tot)
	
	# def gini_subset(self, index_temptive, indexes_included):
	# 	partition = np.sum(self.partitions[i] for i in indexes_included) +   self.partitions[index_temptive] 
	# 	partition_size = np.sum(self.partition_sizes[i] for i in indexes_included) +   self.partition_sizes[index_temptive]
	# 	# print (self.measure_heuristic(current_criterion, partition, partition_size))
	# 	return self.gini_total(self.current_criterion, partition, partition_size)

	def variance_subset(self, index_temptive, indexes_included):
		"""
		Returns L{self.variance_total} for partition variables constructed by the parameters.

		@param index_temptive  : The groups in the current partition.
		@param indexes_included: The groups to add to the partition.
		"""
		if indexes_included:
			n_pos  = self.n_pos [index_temptive] + np.sum(self.n_pos[indexes_included])
			k_pos  = self.k_pos [index_temptive] + np.sum(self.k_pos[indexes_included] , axis=0)
			sv_pos = self.sv_pos[index_temptive] + np.sum(self.sv_pos[indexes_included], axis=0)
			ss_pos = self.ss_pos[index_temptive] + np.sum(self.ss_pos[indexes_included], axis=0)
		else:
			n_pos  = self.n_pos [index_temptive]
			k_pos  = self.k_pos [index_temptive]
			sv_pos = self.sv_pos[index_temptive]
			ss_pos = self.ss_pos[index_temptive]
		partition_size = np.sum([self.partition_sizes[i] for i in indexes_included]) + self.partition_sizes[index_temptive]
		return self.variance_total(k_pos, n_pos, sv_pos, ss_pos)

	# def gini_total(self, partition, partition_size):
	# 	total_annotations = np.sum(self.total_classes)
	# 	positive_annotations = np.sum(partition,axis=0)
		
	# 	positive_gini = ((1.0 - np.sum([(a/positive_annotations) ** 2 for a in partition]))  * self.target_weights * positive_annotations) / self.targets_number 
	# 	# print (positive_gini)
	# 	difference = total_annotations - positive_annotations
	# 	gini_difference = ( (
	# 		1.0 - np.sum([
	# 			((v - partition[i]) / difference ) ** 2 for i,v in enumerate(self.total_classes)
	# 			]) 
	# 		)
	# 		* self.target_weights * difference
	# 	) / self.targets_number 
	# 	# print (gini_difference)
	# 	return self.ftest.test(total_annotations, self.current_criterion,  gini_difference + positive_gini)

	def variance_total(self, k_pos, n_pos, sv_pos , ss_pos):
		"""Returns the total variance if it was accepted by the F-test (-np.inf otherwise)."""
		variance = self.variance( n_pos, k_pos, sv_pos, ss_pos )
		return self.ftest.test(self.n_tot, self.current_criterion,  variance)

	def stop_criteria(self, indexes_included, index_temptive):
		"""True if and only if the current partition violates the min_instances restriction."""
		#### PUT THIS IN A SEPARATE CLASS 
		partition_size = np.nansum([self.n_pos[i] for i in indexes_included]) + self.n_pos[index_temptive]
		# print (partition_size)
		# print (self.n_tot)
		return partition_size < self.min_instances or self.n_tot - partition_size < self.min_instances
	
	def get_attribute_name(self, index):
		for name, i in self.indexing.items():
			if i == index:
				return name
