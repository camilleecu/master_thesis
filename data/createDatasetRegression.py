import pandas as pd
import numpy as np
import sys
n_attrs_categorical = 0
n_attrs_numerical = 3

n_targets = 2
missing_attr_categorical = False
missing_attr_numerical = True
missing_targets = True

df = pd.DataFrame()

choices_attrs = ['a','b','c','d']
choices_classes = ['class1','class2','class3']
if missing_attr_categorical:
	choices_attrs.append('?')

for i in range(n_attrs_categorical):
	df['attr_categorical' + str(i)] = np.random.choice(choices_attrs,100)

for i in range(n_attrs_numerical):
	a = np.random.randint(0,np.random.choice([10,20,30,40,50]),100).astype(float)
	if missing_attr_numerical:
		mask = np.random.choice([1, 0], a.shape, p=[.1, .9]).astype(bool)
		a[mask] = np.nan
	df['attr_numerical' + str(i)] = a

for i in range(n_targets):
	a = np.random.randint(0,20,100).astype(float)
	if missing_targets:
		mask = np.random.choice([1, 0], a.shape, p=[.1, .9]).astype(bool)
		a[mask] = np.nan
	df['target' + str(i)] = a
df.to_csv(sys.argv[1],index=False, na_rep = "NaN")