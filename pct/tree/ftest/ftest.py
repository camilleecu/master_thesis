import numpy as np
from scipy.stats import f

class FTest:
	"""Class containing the tools to perform an F-test for the significance of a split.
	
	Note that due to the normality assumption, this F-test is not a real statistical
	test (but more like a heuristic for when to stop splitting). See
		Hendrik Blockeel, Luc De Raedt, Jan Ramon
		Top-down induction of clustering trees
		U{https://arxiv.org/pdf/cs/0011032.pdf}
	for more details.
	"""

	"""Minimum improvement necessary over the previous criterion value"""
	minimum_improvement = 1e-9

	"""
	Significance levels for the F-test for which this class has built-in values.
	The last value, 0, is for adding a custom F-test (see L{set_value}).
	"""
	possible_values = [1, 0.1, 0.05, 0.01, 0.005, 0.001, 0]

	"""
	Critical p-values for each significance level in FTest.possible_values, for different degrees 
	of freedom in the denominator, starting from  1.
	(note that the degrees of freedom in the numerator is equal to 1, because we have 2 groups).
	The critical p-value is the minimal value for the teststatistic so that the F-test is accepted.
	"""
	critical_f_01 = [
		39.8161, 8.5264, 5.5225, 4.5369, 4.0804, 3.7636,
		3.61, 3.4596, 3.3489, 3.2761, 3.24, 3.1684, 3.1329, 3.0976,
		3.0625, 3.0625, 3.0276, 2.9929, 2.9929, 2.9584
	]
	critical_f_005 = [
		161.0, 18.5, 10.1, 7.71, 6.61, 5.99, 5.59, 5.32, 5.12, 4.96,
		4.84, 4.75, 4.67, 4.6, 4.54, 4.49, 4.45, 4.41, 4.38, 4.35,
		4.32, 4.3, 4.28, 4.26, 4.24, 4.23, 4.21, 4.2, 4.18, 4.17
	]
	critical_f_001 = [
		4052.0, 98.5, 34.1, 21.2, 16.3, 13.7, 12.2, 11.3, 10.6, 10.0,
		9.65, 9.33, 9.07, 8.86, 8.68, 8.53, 8.40, 8.29, 8.18, 8.1,
		8.02, 7.95, 7.88, 7.82, 7.77, 7.72, 7.68, 7.64, 7.6, 7.56
	]
	critical_f_0005 = [
		15876, 198.81, 55.5025, 31.36, 22.7529, 18.6624, 16.2409, 14.6689, 13.6161, 12.8164,
		12.25, 11.7649, 11.2896, 11.0889, 10.8241, 10.5625, 10.3684, 10.24, 10.0489, 9.9225,
		9.8596, 9.7344, 9.61, 9.5481, 9.4864, 9.4249, 9.3636, 9.3025, 9.2416, 9.1809
	]
	critical_f_0001 = [
		405769, 998.56, 166.41, 74.1321, 47.0596, 35.5216, 29.16, 25.4016, 22.8484, 21.0681,
		19.7136, 18.6624, 17.8084, 17.1396, 16.5649, 16.0801, 15.6025, 15.3664, 15.0544, 14.8225,
		14.5924, 14.3641, 14.2129, 13.9876, 13.8384, 13.7641, 13.6161, 13.4689, 13.3956, 13.3225
	]

	def __init__(self, value):
		"""Initializes this F-test with the given p-value.
		
		@param value: The p-value to use in the test.
		@raise ValueError: The given value equals 0 (the reserved value for custom F-tests).
		"""
		if value == 0:
			raise ValueError("This significance level is a reserved value for custom F-tests!")
		self.set_value(value)

	def set_value(self, value):
		"""Sets the significance to the given value.

		If the given value is close enough to one of the built-in levels in L{possible_values},
		also adjusts this FTests level to that built-in level.
		("close enough" meaning that the relative difference does not exceed 0.01)
		Otherwise, computes a new array of critical F-values for different degrees of freedom.
		"""
		self.value = value
		self.level = self.getLevelAndComputeArray(value)
	def getLevelAndComputeArray(self, significance):
		# clus/heuristic/ftest.java (called in clus/main/Settings.java)
		maxlevel = len(self.possible_values) - 1
		for level in range(maxlevel):
			# If the given value is close enough to an already computed value:
			if abs(significance - self.possible_values[level]) / self.possible_values[level] < 0.01:
				return level
		self.possible_values[-1] = significance
		self.initializeFTable(significance)
		return maxlevel
	def initializeFTable(self, sig):
		# clus/heuristic/ftest.java
		df = 3
		value = f.ppf(q=1-sig, dfn=1, dfd=3)
		limit = f.ppf(q=1-sig, dfn=1, dfd=1e5)
		values = []
		while (value-limit)/limit > 0.05:
			value = f.ppf(q=1-sig, dfn=1, dfd=df)
			values.append(value)
			df += 1
		self.ftest_limit = limit
		self.ftest_value = np.zeros(len(values) + 3) # probably for ease of indexing in df later
		self.ftest_value[3:] = values

	def test(self, number_rows, current_criterion, partition_criterion):
		"""
		Returns the variance reduction heuristic if the given split values lead to a 
		statistically significant split, or -np.inf otherwise.

		@param number_rows        : The number of instances (weighted) (n_tot).
		@param current_criterion  : The current criterion value (ss_tot).
		@param partition_criterion: The criterion after partitioning (ss_sum).
		"""
		# TODO This function was probably made to be more general than the variance
		# reduction alone (what the documentation is now suggesting). However, something
		# like information gain doesn't work here I think? Because new_criterion would be
		# negative then, in the way that it is now calculated...
		new_criterion = current_criterion - partition_criterion
		# if new_criterion < 0.01 and new_criterion!=0:
		# 	print (new_criterion) 
		# print (self.minimum_improvement)
		if new_criterion < self.minimum_improvement:
			return -np.inf
		if self.level == 0: # p-value = 1, always significant
			return new_criterion
		# print ("number rows\n", number_rows)
		n_2 = np.floor(number_rows - 2.0 + 0.5).astype(int)
		# print ("n_2" + str(n_2))
		if n_2 <= 0:
			# print ("nao rolou")
			return -np.inf
		else:
			if self.test2(current_criterion,partition_criterion,n_2):
				# print ("aceito")
				return new_criterion 
			else:
				# print ("nao rolou2")
				return -np.inf

	def test2(self, current_criterion, partition_criterion, n2):
		if self.level == 0:
			return True # TODO this should never run right?
		if current_criterion <= 0:
			return False
		elif partition_criterion == 0:
			return True
		f = n2 * (current_criterion - partition_criterion)/partition_criterion
		# print (n2)
		# print (current_criterion)
		# print (partition_criterion)

		cf = self.get_critical_f(n2)
		return f > cf # = Would f be accepted in a statistical test?

	def get_critical_f(self, df):
		"""Returns the critical F-value for the given degrees of freedom (for the denominator)."""
		# print (df)
		if self.level == 1: # 0.1
			if df <= 20:
				return self.critical_f_01[df - 1]
			elif df <= 30:
				return 2.9
			elif df <= 40:
				return 2.86
			elif df <= 120:
				return 2.79
			else:
				return 2.7 					 
		elif self.level == 2: # 0.05
			if df <= 30:
				return self.critical_f_005[df - 1]
			elif df <= 40:
				return 4.08
			elif df <= 60:
				return 4.00
			elif df <= 120: 
				return 3.92
			else:
				return 3.84
		elif self.level == 3: # 0.01
			if df <= 30:
				return self.critical_f_001[df - 1]
			elif df <= 40:
				return 7.31
			elif df <= 60:
				return 7.08
			elif df <= 120: 
				return 6.85
			else:
				return 6.63	
		elif self.level == 4 : # 0.005
			if df <= 30:
				return self.critical_f_0005[df - 1]
			elif df <= 40:
				return 8.82
			elif df <= 60:
				return 8.47
			elif df <= 120:
				return 8.18
			else:
				return 7.90
		elif self.level == 5:	# 0.001
			if df <= 30:
				return self.critical_f_0001[df - 1]
			elif df <= 40:
				return 12.60
			elif df <= 60:
				return 11.98
			elif df <= 120:
				return 11.36
			else:
				return 10.82
		elif self.level == 6:  # Custom F-test
			if df < len(self.ftest_value):
				return self.ftest_value[df]
			else:
				return self.ftest_limit